import numpy as np

from emit_core import diffusion_kernel_3x3, make_state, precompute_k2, step

DEFAULT_SIZE = 256
DT = 1.0
SEED = 3


def default_params():
    return {
        # EMIT coupling
        "D_E": 0.02,
        "kappa": 0.8,
        "A_adv": 0.20,
        "theta": 0.32,
        "alpha": 0.05,
        "beta": 0.12,
        "damping": 0.001,
        "floor": 0.00005,
        "mass_decay": 0.0002,
        "mass_release": 0.0006,
        "time_dilation": 0.8,
        "time_floor": 0.2,
        "D_M": 0.0,
        "v_cap": 5.0,
        "m_cap": 5.0,
        # Gray-Scott
        "D_u": 0.16,
        "D_v": 0.08,
        "F": 0.035,
        "k": 0.065,
        # UI
        "poke": 1.5,
        "poke_sigma": 3.0,
        # Metrics
        "bubble_thresh": 0.15,
        # Display
        "display_gamma": 1.0,
        # Wave driver
        "wave_c2": 0.8,
        "wave_damp": 0.25,
        "wave_couple": 0.01,
        # Components
        "min_component_area": 10,
        # Continuum toggle (legacy default False)
        "continuum_mode": False,
    }


PRESETS = [
    (
        "Turing Spots",
        {
            "D_u": 0.16,
            "D_v": 0.08,
            "F": 0.035,
            "k": 0.065,
            "D_E": 0.02,
            "kappa": 0.6,
            "A_adv": 0.18,
            "theta": 0.32,
            "alpha": 0.04,
            "beta": 0.10,
            "mass_release": 0.0008,
            "time_dilation": 0.7,
            "time_floor": 0.25,
            "v_cap": 5.0,
            "m_cap": 5.0,
        },
    ),
    (
        "Worms",
        {
            "D_u": 0.12,
            "D_v": 0.06,
            "F": 0.025,
            "k": 0.055,
            "D_E": 0.03,
            "kappa": 1.1,
            "A_adv": 0.30,
            "theta": 0.30,
            "alpha": 0.06,
            "beta": 0.14,
            "mass_release": 0.0006,
            "time_dilation": 0.9,
            "time_floor": 0.2,
            "v_cap": 5.0,
            "m_cap": 5.0,
        },
    ),
    (
        "Clusters",
        {
            "D_u": 0.19,
            "D_v": 0.09,
            "F": 0.040,
            "k": 0.062,
            "D_E": 0.02,
            "kappa": 1.4,
            "A_adv": 0.40,
            "theta": 0.36,
            "alpha": 0.08,
            "beta": 0.16,
            "mass_release": 0.0005,
            "time_dilation": 1.1,
            "time_floor": 0.18,
            "v_cap": 5.0,
            "m_cap": 5.0,
        },
    ),
]


class Simulation:
    def __init__(self, size=DEFAULT_SIZE, seed=SEED, dt=DT, params=None, dx=1.0):
        self.size = size
        self.seed = seed
        self.dt = float(dt)
        self.dx = float(dx)
        self.params = default_params()
        if params:
            self.params.update(params)
        self.kernel = diffusion_kernel_3x3()
        self.k2 = precompute_k2(size, size, dx=self.dx)
        self.state = make_state(size, size, seed=seed)
        self.last_phi = np.zeros((size, size), dtype=np.float32)
        self.last_dM_map = None

    def reset(self, seed=SEED):
        self.seed = seed
        self.state = make_state(self.size, self.size, seed=seed)
        self.last_phi[:] = 0.0
        self.last_dM_map = None

    def step(self):
        self.state, self.last_phi, self.last_dM_map = step(
            self.state, self.params, self.k2, self.kernel, dt=self.dt, dx=self.dx
        )
        return self.state, self.last_phi

    def set_dx(self, dx):
        self.dx = float(dx)
        self.k2 = precompute_k2(self.size, self.size, dx=self.dx)

    def apply_action(self, action):
        result = {
            "applied": False,
            "action_name": str(action.get("type")) if action else "unknown",
            "delta_V_total": 0.0,
            "delta_M_total": 0.0,
            "delta_U_total": 0.0,
            "delta_field_total": np.nan,
            "delta_field_l2": np.nan,
            "affected_field": None,
        }
        if not action:
            return result

        x = action.get("x")
        y = action.get("y")
        if x is None or y is None:
            return result

        strength = float(action.get("strength", self.params.get("poke", 1.0)))
        sigma = float(action.get("sigma", self.params.get("poke_sigma", 3.0)))
        radius = int(action.get("radius", max(2.0, sigma * 3.0)))
        action_type = str(action.get("type", "poke"))

        v_before = float(self.state["V"].sum())
        m_before = float(self.state["M"].sum())
        u_before = float(self.state["U"].sum())
        field_before = None

        h, w = self.state["V"].shape
        yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        dist2 = (xx * xx + yy * yy).astype(np.float32)
        ys = (y + yy) % h
        xs = (x + xx) % w

        internal_peak = None
        if action_type == "poke" and "Wdot" in self.state:
            norm = dist2 / (sigma * sigma + 1e-6)
            ring = (norm - 1.0) * np.exp(-0.5 * norm)
            ring = ring.astype(np.float32)
            field_before = self.state["Wdot"][ys, xs].copy()
            self.state["Wdot"][ys, xs] += strength * ring
            np.nan_to_num(
                self.state["Wdot"], copy=False, nan=0.0, posinf=0.0, neginf=0.0
            )
            internal_peak = float(ring.max())
            result["affected_field"] = "Wdot"
        else:
            blob = np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32)
            field_before = self.state["V"][ys, xs].copy()
            self.state["V"][ys, xs] += strength * blob
            self.state["V"][:] = np.clip(self.state["V"], 0.0, 1.0)
            self.state["U"][:] = np.clip(1.0 - self.state["V"], 0.0, 1.0)
            internal_peak = float(blob.max())
            result["affected_field"] = "V"

        v_after = float(self.state["V"].sum())
        m_after = float(self.state["M"].sum())
        u_after = float(self.state["U"].sum())

        delta_field_total = np.nan
        delta_field_l2 = np.nan
        if field_before is not None:
            if result["affected_field"] == "Wdot":
                field_after = self.state["Wdot"][ys, xs]
            else:
                field_after = self.state["V"][ys, xs]
            field_delta = field_after - field_before
            delta_field_total = float(field_delta.sum())
            delta_field_l2 = float(np.sqrt(np.sum(field_delta * field_delta)))

        result.update(
            {
                "applied": True,
                "action_name": action_type,
                "delta_V_total": v_after - v_before,
                "delta_M_total": m_after - m_before,
                "delta_U_total": u_after - u_before,
                "delta_field_total": delta_field_total,
                "delta_field_l2": delta_field_l2,
                "strength": strength,
                "radius": radius,
                "sigma": sigma,
                "internal_peak": internal_peak,
            }
        )
        return result

    def apply_preset(self, preset):
        self.params.update(preset)

    def preset_names(self):
        return [name for name, _ in PRESETS]
