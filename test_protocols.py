from __future__ import annotations

import math

from sim import PRESETS

DEFAULT_PRESET = "Turing Spots"
DEFAULT_SEED = 3
DEFAULT_STEPS = 2000
DEFAULT_POKE_FRAME = 50
DEFAULT_POKE_STRENGTH = 1.5
DEFAULT_POKE_SIGMA = 3.0
DEFAULT_POKE_RADIUS = 9
CONTINUUM_L = 1.0
CONTINUUM_N_REF = 256
CONTINUUM_DT_REF = 1.0


def _run_spec(
    *,
    preset_name=DEFAULT_PRESET,
    grid_size=256,
    dt=1.0,
    steps=DEFAULT_STEPS,
    seed=DEFAULT_SEED,
    poke_frame=DEFAULT_POKE_FRAME,
    poke_strength=DEFAULT_POKE_STRENGTH,
    poke_sigma=DEFAULT_POKE_SIGMA,
    poke_radius=DEFAULT_POKE_RADIUS,
    click_mode="poke",
    field="auto",
    snapshot_every=None,
    snapshot_frames=None,
    snapshot_mode=None,
    param_overrides=None,
    test_physical_L=None,
    dx=None,
    n_ref=None,
    scaling_rules=None,
    physical_T=None,
    continuum_mode=None,
    snapshot_times=None,
    save_png_snapshots=None,
    save_npz_snapshots=None,
    always_save_keyframes_png=None,
):
    return {
        "preset_name": preset_name,
        "grid_size": int(grid_size),
        "dt": float(dt),
        "steps": int(steps),
        "seed": int(seed),
        "poke_frame": int(poke_frame),
        "poke_strength": float(poke_strength),
        "poke_sigma": float(poke_sigma),
        "poke_radius": int(poke_radius),
        "click_mode": str(click_mode),
        "field": field,
        "snapshot_every": int(snapshot_every) if snapshot_every else None,
        "snapshot_frames": list(snapshot_frames) if snapshot_frames else None,
        "snapshot_mode": snapshot_mode,
        "param_overrides": dict(param_overrides) if param_overrides else None,
        "test_physical_L": float(test_physical_L) if test_physical_L is not None else None,
        "dx": float(dx) if dx is not None else None,
        "n_ref": int(n_ref) if n_ref is not None else None,
        "scaling_rules": dict(scaling_rules) if scaling_rules else None,
        "physical_T": float(physical_T) if physical_T is not None else None,
        "dt_ref": float(CONTINUUM_DT_REF) if test_physical_L is not None else None,
        "dx_ref": float(1.0 / CONTINUUM_N_REF) if test_physical_L is not None else None,
        "continuum_mode": bool(continuum_mode) if continuum_mode is not None else None,
        "snapshot_times": list(snapshot_times) if snapshot_times else None,
        "save_png_snapshots": bool(save_png_snapshots) if save_png_snapshots is not None else None,
        "save_npz_snapshots": bool(save_npz_snapshots) if save_npz_snapshots is not None else None,
        "always_save_keyframes_png": bool(always_save_keyframes_png) if always_save_keyframes_png is not None else None,
    }


def _preset_map():
    return {name: preset for name, preset in PRESETS}


def _scaled_params(preset_name, n, n_ref):
    base = _preset_map().get(preset_name, {})
    scale = float(n) / float(n_ref)
    diffusion_scale = scale * scale
    adv_scale = scale
    diffusion_keys = ["D_E", "D_u", "D_v"]
    advection_keys = ["A_adv", "kappa"]
    overrides = {}
    # Diffusion-like terms scale with dx^-2; advection/gradient-like scale with dx^-1.
    for key in diffusion_keys:
        if key in base:
            overrides[key] = float(base[key]) * diffusion_scale
    for key in advection_keys:
        if key in base:
            overrides[key] = float(base[key]) * adv_scale
    scaling_rules = {
        "diffusion_scale": f"(N/{n_ref})^2",
        "advection_scale": f"(N/{n_ref})",
        "diffusion_keys": diffusion_keys,
        "advection_keys": advection_keys,
        "params_scaled": True,
    }
    return overrides, scaling_rules


def _dt_scaled(dx, dx_ref, dt_ref):
    ratio = dx / dx_ref
    return float(dt_ref * (ratio * ratio))


def _poke_radius_scaled(radius_ref, n, n_ref):
    return max(2, int(round(radius_ref * (float(n) / float(n_ref)))))


def _poke_sigma_scaled(sigma_ref, n, n_ref):
    return max(2, int(round(sigma_ref * (float(n) / float(n_ref)))))


def _frames_from_times(times, dt):
    frames = []
    for t in times:
        frames.append(int(math.ceil(float(t) / float(dt))))
    return frames


TESTS = {
    "Test 1": {
        "label": "Test 1 dt sweep",
        "name": "Test_1_dt_sweep",
        "runs": [
            _run_spec(grid_size=256, dt=0.25),
            _run_spec(grid_size=256, dt=0.5),
            _run_spec(grid_size=256, dt=1.0),
            _run_spec(grid_size=256, dt=2.0),
        ],
    },
    "Test 2": {
        "label": "Test 2 N sweep",
        "name": "Test_2_N_sweep",
        "runs": [
            _run_spec(grid_size=128, dt=2.0),
            _run_spec(grid_size=192, dt=1.5),
            _run_spec(grid_size=256, dt=1.0),
            _run_spec(grid_size=512, dt=0.5),
        ],
    },
    "Test 3": {
        "label": "Test 3 symmetry",
        "name": "Test_3_symmetry",
        "runs": [
            _run_spec(grid_size=128, dt=2.0),
            _run_spec(grid_size=192, dt=1.5),
            _run_spec(grid_size=256, dt=1.0),
            _run_spec(grid_size=512, dt=0.5),
        ],
    },
    "Test 4": {
        "label": "Test 4 PDE snapshots",
        "name": "Test_4_PDE_snapshots",
        "runs": [
            _run_spec(grid_size=256, dt=1.0, snapshot_every=500),
        ],
    },
    "Test 5": {
        "label": "Test 5 scaling limit",
        "name": "Test_5_scaling",
        "runs": [
            _run_spec(grid_size=128, dt=1.0, poke_sigma=max(2, round(128 / 64)), poke_radius=max(2, round(128 / 64)) * 3, snapshot_frames=[50, 200, 800, -1], snapshot_mode="png"),
            _run_spec(grid_size=128, dt=0.5, poke_sigma=max(2, round(128 / 64)), poke_radius=max(2, round(128 / 64)) * 3, snapshot_frames=[50, 200, 800, -1], snapshot_mode="png"),
            _run_spec(grid_size=256, dt=1.0, poke_sigma=max(2, round(256 / 64)), poke_radius=max(2, round(256 / 64)) * 3, snapshot_frames=[50, 200, 800, -1], snapshot_mode="png"),
            _run_spec(grid_size=256, dt=0.5, poke_sigma=max(2, round(256 / 64)), poke_radius=max(2, round(256 / 64)) * 3, snapshot_frames=[50, 200, 800, -1], snapshot_mode="png"),
            _run_spec(grid_size=512, dt=0.5, steps=1500, poke_sigma=max(2, round(512 / 64)), poke_radius=max(2, round(512 / 64)) * 3, snapshot_frames=[50, 200, 800, -1], snapshot_mode="png"),
        ],
    },
    "Test 6": {
        "label": "Test 6 continuum dx-rescaled",
        "name": "Test_6_continuum_dx_rescaled",
        "runs": [
            _run_spec(
                grid_size=128,
                dt=1.0,
                steps=int(1000 / 1.0),
                poke_sigma=max(2, round(128 / 64)),
                poke_radius=max(2, round(128 / 64)) * 3,
                snapshot_frames=[50, 200, 800, -1],
                snapshot_mode="png",
                param_overrides=_scaled_params(DEFAULT_PRESET, 128, CONTINUUM_N_REF)[0],
                test_physical_L=CONTINUUM_L,
                dx=CONTINUUM_L / 128,
                n_ref=CONTINUUM_N_REF,
                scaling_rules=_scaled_params(DEFAULT_PRESET, 128, CONTINUUM_N_REF)[1],
                physical_T=1000.0,
                continuum_mode=True,
            ),
            _run_spec(
                grid_size=256,
                dt=1.0,
                steps=int(1000 / 1.0),
                poke_sigma=max(2, round(256 / 64)),
                poke_radius=max(2, round(256 / 64)) * 3,
                snapshot_frames=[50, 200, 800, -1],
                snapshot_mode="png",
                param_overrides=_scaled_params(DEFAULT_PRESET, 256, CONTINUUM_N_REF)[0],
                test_physical_L=CONTINUUM_L,
                dx=CONTINUUM_L / 256,
                n_ref=CONTINUUM_N_REF,
                scaling_rules=_scaled_params(DEFAULT_PRESET, 256, CONTINUUM_N_REF)[1],
                physical_T=1000.0,
                continuum_mode=True,
            ),
            _run_spec(
                grid_size=512,
                dt=1.0,
                steps=int(800 / 1.0),
                poke_sigma=max(2, round(512 / 64)),
                poke_radius=max(2, round(512 / 64)) * 3,
                snapshot_frames=[50, 200, 800, -1],
                snapshot_mode="png",
                param_overrides=_scaled_params(DEFAULT_PRESET, 512, CONTINUUM_N_REF)[0],
                test_physical_L=CONTINUUM_L,
                dx=CONTINUUM_L / 512,
                n_ref=CONTINUUM_N_REF,
                scaling_rules=_scaled_params(DEFAULT_PRESET, 512, CONTINUUM_N_REF)[1],
                physical_T=800.0,
                continuum_mode=True,
            ),
        ],
    },
    "Test 6B": {
        "label": "Test 6B CFL-stable dt~dx^2",
        "name": "Test_6B_CFL_dt_dx2",
        "runs": [
            _run_spec(
                grid_size=128,
                dt=_dt_scaled(1.0 / 128.0, 1.0 / CONTINUUM_N_REF, CONTINUUM_DT_REF),
                steps=int(math.ceil(1000.0 / _dt_scaled(1.0 / 128.0, 1.0 / CONTINUUM_N_REF, CONTINUUM_DT_REF))),
                poke_frame=int(math.ceil(50.0 / _dt_scaled(1.0 / 128.0, 1.0 / CONTINUUM_N_REF, CONTINUUM_DT_REF))),
                poke_sigma=_poke_sigma_scaled(DEFAULT_POKE_SIGMA, 128, CONTINUUM_N_REF),
                poke_radius=_poke_radius_scaled(DEFAULT_POKE_RADIUS, 128, CONTINUUM_N_REF),
                snapshot_frames=[50, 200, 800, -1],
                snapshot_mode="png",
                test_physical_L=CONTINUUM_L,
                dx=CONTINUUM_L / 128,
                n_ref=CONTINUUM_N_REF,
                physical_T=1000.0,
                scaling_rules={
                    "dt_scaled_rule": "dt~dx^2",
                    "params_scaled": False,
                    "radius_rule": "radius_ref_px*(N/256)",
                    "strength_rule": "constant",
                },
                continuum_mode=True,
            ),
            _run_spec(
                grid_size=256,
                dt=_dt_scaled(1.0 / 256.0, 1.0 / CONTINUUM_N_REF, CONTINUUM_DT_REF),
                steps=int(math.ceil(1000.0 / _dt_scaled(1.0 / 256.0, 1.0 / CONTINUUM_N_REF, CONTINUUM_DT_REF))),
                poke_frame=int(math.ceil(50.0 / _dt_scaled(1.0 / 256.0, 1.0 / CONTINUUM_N_REF, CONTINUUM_DT_REF))),
                poke_sigma=_poke_sigma_scaled(DEFAULT_POKE_SIGMA, 256, CONTINUUM_N_REF),
                poke_radius=_poke_radius_scaled(DEFAULT_POKE_RADIUS, 256, CONTINUUM_N_REF),
                snapshot_frames=[50, 200, 800, -1],
                snapshot_mode="png",
                test_physical_L=CONTINUUM_L,
                dx=CONTINUUM_L / 256,
                n_ref=CONTINUUM_N_REF,
                physical_T=1000.0,
                scaling_rules={
                    "dt_scaled_rule": "dt~dx^2",
                    "params_scaled": False,
                    "radius_rule": "radius_ref_px*(N/256)",
                    "strength_rule": "constant",
                },
                continuum_mode=True,
            ),
            _run_spec(
                grid_size=512,
                dt=_dt_scaled(1.0 / 512.0, 1.0 / CONTINUUM_N_REF, CONTINUUM_DT_REF),
                steps=int(math.ceil(1000.0 / _dt_scaled(1.0 / 512.0, 1.0 / CONTINUUM_N_REF, CONTINUUM_DT_REF))),
                poke_frame=int(math.ceil(50.0 / _dt_scaled(1.0 / 512.0, 1.0 / CONTINUUM_N_REF, CONTINUUM_DT_REF))),
                poke_sigma=_poke_sigma_scaled(DEFAULT_POKE_SIGMA, 512, CONTINUUM_N_REF),
                poke_radius=_poke_radius_scaled(DEFAULT_POKE_RADIUS, 512, CONTINUUM_N_REF),
                snapshot_frames=[50, 200, 800, -1],
                snapshot_mode="png",
                test_physical_L=CONTINUUM_L,
                dx=CONTINUUM_L / 512,
                n_ref=CONTINUUM_N_REF,
                physical_T=1000.0,
                scaling_rules={
                    "dt_scaled_rule": "dt~dx^2",
                    "params_scaled": False,
                    "radius_rule": "radius_ref_px*(N/256)",
                    "strength_rule": "constant",
                },
                continuum_mode=True,
            ),
        ],
    },
    "Test 7": {
        "label": "Test 7 dt invariance",
        "name": "Test_7_dt_invariance",
        "runs": [
            _run_spec(
                grid_size=256,
                dt=1.0,
                steps=int(1000.0 / 1.0),
                poke_frame=int(math.ceil(50.0 / 1.0)),
                poke_sigma=_poke_sigma_scaled(DEFAULT_POKE_SIGMA, 256, CONTINUUM_N_REF),
                poke_radius=_poke_radius_scaled(DEFAULT_POKE_RADIUS, 256, CONTINUUM_N_REF),
                snapshot_times=[0.0, 50.0, 200.0, 500.0, 1000.0],
                snapshot_frames=_frames_from_times([0.0, 50.0, 200.0, 500.0, 1000.0], 1.0),
                snapshot_mode="png",
                test_physical_L=CONTINUUM_L,
                dx=CONTINUUM_L / 256,
                n_ref=CONTINUUM_N_REF,
                physical_T=1000.0,
                scaling_rules={
                    "dt_scaled_rule": "dt~dx^2",
                    "params_scaled": False,
                    "radius_rule": "radius_ref_px*(N/256)",
                    "strength_rule": "constant",
                },
                continuum_mode=True,
                save_png_snapshots=True,
                save_npz_snapshots=True,
            ),
            _run_spec(
                grid_size=256,
                dt=0.5,
                steps=int(1000.0 / 0.5),
                poke_frame=int(math.ceil(50.0 / 0.5)),
                poke_sigma=_poke_sigma_scaled(DEFAULT_POKE_SIGMA, 256, CONTINUUM_N_REF),
                poke_radius=_poke_radius_scaled(DEFAULT_POKE_RADIUS, 256, CONTINUUM_N_REF),
                snapshot_times=[0.0, 50.0, 200.0, 500.0, 1000.0],
                snapshot_frames=_frames_from_times([0.0, 50.0, 200.0, 500.0, 1000.0], 0.5),
                snapshot_mode="png",
                test_physical_L=CONTINUUM_L,
                dx=CONTINUUM_L / 256,
                n_ref=CONTINUUM_N_REF,
                physical_T=1000.0,
                scaling_rules={
                    "dt_scaled_rule": "dt~dx^2",
                    "params_scaled": False,
                    "radius_rule": "radius_ref_px*(N/256)",
                    "strength_rule": "constant",
                },
                continuum_mode=True,
                save_png_snapshots=True,
                save_npz_snapshots=True,
            ),
            _run_spec(
                grid_size=256,
                dt=0.25,
                steps=int(1000.0 / 0.25),
                poke_frame=int(math.ceil(50.0 / 0.25)),
                poke_sigma=_poke_sigma_scaled(DEFAULT_POKE_SIGMA, 256, CONTINUUM_N_REF),
                poke_radius=_poke_radius_scaled(DEFAULT_POKE_RADIUS, 256, CONTINUUM_N_REF),
                snapshot_times=[0.0, 50.0, 200.0, 500.0, 1000.0],
                snapshot_frames=_frames_from_times([0.0, 50.0, 200.0, 500.0, 1000.0], 0.25),
                snapshot_mode="png",
                test_physical_L=CONTINUUM_L,
                dx=CONTINUUM_L / 256,
                n_ref=CONTINUUM_N_REF,
                physical_T=1000.0,
                scaling_rules={
                    "dt_scaled_rule": "dt~dx^2",
                    "params_scaled": False,
                    "radius_rule": "radius_ref_px*(N/256)",
                    "strength_rule": "constant",
                },
                continuum_mode=True,
                save_png_snapshots=True,
                save_npz_snapshots=True,
            ),
        ],
    },
    "Test 8": {
        "label": "Test 8 prediction plateau",
        "name": "Test_8_prediction_plateau",
        "runs": [
            _run_spec(
                grid_size=256,
                seed=seed,
                dt=1.0,
                steps=int(1000.0 / 1.0),
                poke_frame=int(math.ceil(50.0 / 1.0)),
                poke_sigma=_poke_sigma_scaled(DEFAULT_POKE_SIGMA, 256, CONTINUUM_N_REF),
                poke_radius=_poke_radius_scaled(DEFAULT_POKE_RADIUS, 256, CONTINUUM_N_REF),
                snapshot_times=[0.0, 50.0, 200.0, 500.0, 1000.0],
                snapshot_frames=_frames_from_times([0.0, 50.0, 200.0, 500.0, 1000.0], 1.0),
                snapshot_mode="png",
                test_physical_L=CONTINUUM_L,
                dx=CONTINUUM_L / 256,
                n_ref=CONTINUUM_N_REF,
                physical_T=1000.0,
                scaling_rules={
                    "dt_scaled_rule": "dt~dx^2",
                    "params_scaled": False,
                    "radius_rule": "radius_ref_px*(N/256)",
                    "strength_rule": "constant",
                },
                continuum_mode=True,
                save_png_snapshots=True,
                save_npz_snapshots=True,
            )
            for seed in [0, 1, 2, 3, 4]
        ],
    },
    "Test 9": {
        "label": "Test 9 effective PDE fit",
        "name": "Test_9_effective_PDE_fit",
        "runs": [
            _run_spec(
                grid_size=256,
                dt=1.0,
                steps=int(300.0 / 1.0),
                poke_frame=int(math.ceil(50.0 / 1.0)),
                poke_sigma=_poke_sigma_scaled(DEFAULT_POKE_SIGMA, 256, CONTINUUM_N_REF),
                poke_radius=_poke_radius_scaled(DEFAULT_POKE_RADIUS, 256, CONTINUUM_N_REF),
                snapshot_times=[40.0, 50.0, 60.0, 120.0, 300.0],
                snapshot_frames=_frames_from_times([40.0, 50.0, 60.0, 120.0, 300.0], 1.0),
                snapshot_mode="png",
                test_physical_L=CONTINUUM_L,
                dx=CONTINUUM_L / 256,
                n_ref=CONTINUUM_N_REF,
                physical_T=300.0,
                scaling_rules={
                    "dt_scaled_rule": "dt~dx^2",
                    "params_scaled": False,
                    "radius_rule": "radius_ref_px*(N/256)",
                    "strength_rule": "constant",
                },
                continuum_mode=True,
                save_png_snapshots=True,
                save_npz_snapshots=True,
            ),
        ],
    },
    "Test 10": {
        "label": "Test 10 stability N512",
        "name": "Test_10_stability_N512",
        "runs": [
            _run_spec(
                grid_size=512,
                seed=0,
                dt=_dt_scaled(1.0 / 512.0, 1.0 / CONTINUUM_N_REF, CONTINUUM_DT_REF),
                steps=int(math.ceil(1000.0 / _dt_scaled(1.0 / 512.0, 1.0 / CONTINUUM_N_REF, CONTINUUM_DT_REF))),
                poke_frame=50,
                poke_sigma=_poke_sigma_scaled(DEFAULT_POKE_SIGMA, 512, CONTINUUM_N_REF),
                poke_radius=_poke_radius_scaled(DEFAULT_POKE_RADIUS, 512, CONTINUUM_N_REF),
                test_physical_L=CONTINUUM_L,
                dx=CONTINUUM_L / 512,
                n_ref=CONTINUUM_N_REF,
                physical_T=1000.0,
                scaling_rules={
                    "dt_scaled_rule": "dt = dt_ref*(dx/dx_ref)^2",
                    "params_scaled": False,
                    "radius_rule": "radius_ref_px*(N/256)",
                    "strength_rule": "constant",
                },
                continuum_mode=True,
                save_png_snapshots=True,
                save_npz_snapshots=False,
                always_save_keyframes_png=True,
            ),
        ],
    },
    "Test 11": {
        "label": "Test 11 continuum dt invariance N256",
        "name": "Test_11_continuum_dt_invariance_N256",
        "runs": [
            _run_spec(
                grid_size=256,
                seed=0,
                dt=dt_value,
                steps=int(math.ceil(1000.0 / dt_value)),
                poke_frame=50,
                poke_sigma=_poke_sigma_scaled(DEFAULT_POKE_SIGMA, 256, CONTINUUM_N_REF),
                poke_radius=_poke_radius_scaled(DEFAULT_POKE_RADIUS, 256, CONTINUUM_N_REF),
                test_physical_L=CONTINUUM_L,
                dx=CONTINUUM_L / 256,
                n_ref=CONTINUUM_N_REF,
                physical_T=1000.0,
                scaling_rules={
                    "dt_scaled_rule": "dt = dt_ref*(dx/dx_ref)^2",
                    "params_scaled": False,
                    "radius_rule": "radius_ref_px*(N/256)",
                    "strength_rule": "constant",
                },
                continuum_mode=True,
                save_png_snapshots=True,
                save_npz_snapshots=False,
                always_save_keyframes_png=True,
            )
            for dt_value in [1.0, 0.5, 0.25]
        ],
    },
    "Test 12": {
        "label": "Test 12 refinement trend CFL",
        "name": "Test_12_refinement_trend_CFL",
        "runs": [
            _run_spec(
                grid_size=grid,
                seed=0,
                dt=_dt_scaled(1.0 / grid, 1.0 / CONTINUUM_N_REF, CONTINUUM_DT_REF),
                steps=int(math.ceil(1000.0 / _dt_scaled(1.0 / grid, 1.0 / CONTINUUM_N_REF, CONTINUUM_DT_REF))),
                poke_frame=50,
                poke_sigma=_poke_sigma_scaled(DEFAULT_POKE_SIGMA, grid, CONTINUUM_N_REF),
                poke_radius=_poke_radius_scaled(DEFAULT_POKE_RADIUS, grid, CONTINUUM_N_REF),
                test_physical_L=CONTINUUM_L,
                dx=CONTINUUM_L / grid,
                n_ref=CONTINUUM_N_REF,
                physical_T=1000.0,
                scaling_rules={
                    "dt_scaled_rule": "dt = dt_ref*(dx/dx_ref)^2",
                    "params_scaled": False,
                    "radius_rule": "radius_ref_px*(N/256)",
                    "strength_rule": "constant",
                },
                continuum_mode=True,
                save_png_snapshots=True,
                save_npz_snapshots=False,
                always_save_keyframes_png=True,
            )
            for grid in [128, 256, 512]
        ],
    },
}

TEST_ORDER = ["Off", "Test 1", "Test 2", "Test 3", "Test 4", "Test 5", "Test 6", "Test 6B", "Test 7", "Test 8", "Test 9", "Test 10", "Test 11", "Test 12"]
