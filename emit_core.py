import numpy as np
from scipy.ndimage import convolve

# ----------------------------
# Core math helpers
# ----------------------------

def diffusion_kernel_3x3(dtype=np.float32):
    return np.array(
        [
            [0.05, 0.20, 0.05],
            [0.20, -1.0, 0.20],
            [0.05, 0.20, 0.05],
        ],
        dtype=dtype,
    )


def precompute_k2(h, w, dx=1.0, dtype=np.float32):
    ky = 2.0 * np.pi * np.fft.fftfreq(h, d=dx).astype(dtype)
    kx = 2.0 * np.pi * np.fft.fftfreq(w, d=dx).astype(dtype)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k2 = kx_grid * kx_grid + ky_grid * ky_grid
    k2[0, 0] = 1.0
    return k2


def poisson_potential_fft(source, k2):
    s_hat = np.fft.fft2(source)
    phi_hat = -s_hat / k2
    phi_hat[0, 0] = 0.0
    phi = np.fft.ifft2(phi_hat).real.astype(np.float32)
    return phi


def grad_centered(phi):
    dphi_dx = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) * 0.5
    dphi_dy = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) * 0.5
    return dphi_dx, dphi_dy


def divergence_centered(fx, fy):
    dfx_dx = (np.roll(fx, -1, axis=1) - np.roll(fx, 1, axis=1)) * 0.5
    dfy_dy = (np.roll(fy, -1, axis=0) - np.roll(fy, 1, axis=0)) * 0.5
    return dfx_dx + dfy_dy


# ----------------------------
# State init
# ----------------------------

def _add_gaussian(field, cx, cy, sigma, amplitude):
    r = int(max(3.0, sigma * 3.0))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    dist2 = (xx * xx + yy * yy).astype(np.float32)
    blob = np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32) * amplitude

    h, w = field.shape
    ys = (cy + yy) % h
    xs = (cx + xx) % w
    field[ys, xs] += blob


def make_state(h, w, seed=3):
    rng = np.random.default_rng(seed)
    u = np.ones((h, w), dtype=np.float32)
    v = np.zeros((h, w), dtype=np.float32)
    m = np.zeros((h, w), dtype=np.float32)
    t = np.ones((h, w), dtype=np.float32)
    w_field = np.zeros((h, w), dtype=np.float32)
    w_dot = np.zeros((h, w), dtype=np.float32)

    # Low-amplitude noise everywhere to avoid a hard square start.
    v += rng.random((h, w), dtype=np.float32) * 0.01

    # Add a few soft Gaussian seeds for pattern growth.
    for _ in range(6):
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))
        sigma = float(rng.uniform(3.0, 6.0))
        amp = float(rng.uniform(0.12, 0.22))
        _add_gaussian(v, cx, cy, sigma, amp)

    v = np.clip(v, 0.0, 1.0, out=v)
    u = 1.0 - v

    return {
        "U": u.astype(np.float32, copy=False),
        "V": v.astype(np.float32, copy=False),
        "M": m,
        "T": t,
        "W": w_field,
        "Wdot": w_dot,
    }


# ----------------------------
# Core update operator
# ----------------------------

def step(state, params, k2, kernel, dt=1.0, dx=1.0):
    u = state["U"]
    v = state["V"]
    m = state["M"]
    w_field = state.get("W")
    w_dot = state.get("Wdot")
    v_cap = float(params.get("v_cap", 5.0))
    m_cap = float(params.get("m_cap", 5.0))
    tscale = state.get("T")
    if tscale is None:
        tscale = np.ones_like(m, dtype=np.float32)
    tscale = np.clip(tscale, params["time_floor"], 1.0)
    tdt = dt * tscale
    continuum_mode = bool(params.get("continuum_mode", False))
    if continuum_mode:
        inv_dx = 1.0 / max(dx, 1e-9)
        inv_dx2 = inv_dx * inv_dx
    else:
        inv_dx = 1.0
        inv_dx2 = 1.0

    # Guard against NaNs/Infs from prior steps.
    np.nan_to_num(u, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    np.nan_to_num(v, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    np.nan_to_num(m, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # 1) Propagation / diffusion on the excitation field (V)
    lap_v = convolve(v, kernel, mode="wrap").astype(np.float32, copy=False)
    v = v + (params["D_E"] * lap_v * inv_dx2) * tdt
    v = np.clip(v, 0.0, v_cap, out=v)

    # 1b) Wave driver field (W) with velocity (Wdot)
    if w_field is not None and w_dot is not None:
        lap_w = convolve(w_field, kernel, mode="wrap").astype(np.float32, copy=False)
        w_dot = w_dot + (params["wave_c2"] * lap_w * inv_dx2 - params["wave_damp"] * w_dot) * tdt
        w_field = w_field + w_dot * tdt
        np.nan_to_num(w_field, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(w_dot, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # 2) Gray-Scott reaction-diffusion
    lap_u = convolve(u, kernel, mode="wrap").astype(np.float32, copy=False)
    lap_v = convolve(v, kernel, mode="wrap").astype(np.float32, copy=False)
    uvv = u * v * v

    u = u + (params["D_u"] * lap_u * inv_dx2 - uvv + params["F"] * (1.0 - u)) * tdt
    v = v + (params["D_v"] * lap_v * inv_dx2 + uvv - (params["F"] + params["k"]) * v) * tdt

    if w_field is not None:
        if continuum_mode:
            v = v + params["wave_couple"] * w_field * tdt
        else:
            v = v + params["wave_couple"] * w_field

    np.clip(u, 0.0, 1.0, out=u)
    v = np.clip(v, 0.0, v_cap, out=v)

    # 3) Threshold freeze: V -> M
    dM_map = np.zeros_like(m)
    hotspots = v > params["theta"]
    if hotspots.any():
        dM = params["alpha"] * v[hotspots] * tdt[hotspots]
        v[hotspots] -= dM
        m[hotspots] += dM
        dM_map[hotspots] = dM
    v = np.clip(v, 0.0, v_cap, out=v)
    m = np.clip(m, 0.0, m_cap, out=m)

    # 4) Secondary emission pulse from new mass
    if params["beta"] > 0.0:
        pulse = convolve(dM_map, kernel, mode="wrap").astype(np.float32, copy=False)
        pulse = np.maximum(pulse, 0.0, out=pulse)
        if continuum_mode:
            v = v + params["beta"] * pulse * tdt
        else:
            v = v + params["beta"] * pulse
        v = np.clip(v, 0.0, v_cap, out=v)

    # 5) Gravity: Poisson potential from mass, advect V downhill
    m = np.clip(m, 0.0, m_cap, out=m)
    m_mean = m.mean(dtype=np.float32)
    source = params["kappa"] * (m - m_mean)
    source = np.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
    phi = poisson_potential_fft(source, k2)

    gx, gy = grad_centered(phi)
    gx = -gx
    gy = -gy
    if continuum_mode:
        gx *= inv_dx
        gy *= inv_dx

    fx = v * gx
    fy = v * gy
    div_f = divergence_centered(fx, fy)
    if continuum_mode:
        div_f *= inv_dx
    v = v - (params["A_adv"] * div_f) * tdt
    v = np.clip(v, 0.0, v_cap, out=v)

    # 6) Mass release back into excitation (toy recycle term)
    if params["mass_release"] > 0.0:
        release = params["mass_release"] * m * tdt
        m = np.maximum(m - release, 0.0, out=m)
        v = v + release
        v = np.clip(v, 0.0, v_cap, out=v)
        m = np.clip(m, 0.0, m_cap, out=m)

    # 7) Collapse / cleanup
    v *= (1.0 - params["damping"] * tdt)
    v = np.clip(v, 0.0, v_cap, out=v)
    v[v < params["floor"]] = 0.0

    if params["mass_decay"] > 0.0:
        m *= (1.0 - params["mass_decay"] * tdt)
        m = np.clip(m, 0.0, m_cap, out=m)

    if params["D_M"] > 0.0:
        lap_m = convolve(m, kernel, mode="wrap").astype(np.float32, copy=False)
        m = m + (params["D_M"] * lap_m * inv_dx2) * tdt
        m = np.clip(m, 0.0, m_cap, out=m)

    # 8) Lazy evaluation (time dilation map for next tick)
    tscale = 1.0 / (1.0 + params["time_dilation"] * m)
    np.nan_to_num(tscale, copy=False, nan=1.0, posinf=1.0, neginf=1.0)
    tscale = np.clip(tscale, params["time_floor"], 1.0, out=tscale)

    state["U"] = u.astype(np.float32, copy=False)
    state["V"] = v.astype(np.float32, copy=False)
    state["M"] = m.astype(np.float32, copy=False)
    state["T"] = tscale.astype(np.float32, copy=False)
    if w_field is not None:
        state["W"] = w_field.astype(np.float32, copy=False)
    if w_dot is not None:
        state["Wdot"] = w_dot.astype(np.float32, copy=False)

    return state, phi, dM_map
