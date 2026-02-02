# EMIT

## DOI (archived release)
Zenodo record (frozen archive, data + code + paper): **10.5281/zenodo.18457553**

## GitHub (active development)
Repository: https://github.com/neal2hawkins-hub/EMIT_SIM

## Reproduce figures/tables from logs
If you have run logs in `runs/` (or you downloaded them from the Zenodo archive), you can regenerate summary outputs:

```
python analysis/analyze_tests.py --runs_dir runs
```

Outputs are written under `analysis_outputs/` by default (see script flags/help).

## How to cite
- Zenodo DOI: **10.5281/zenodo.18457553**
- See `CITATION.cff` for a ready-made citation entry.

EMIT is a real-time, physically-inspired toy simulation built from continuous cellular automata ideas, Gray-Scott reaction-diffusion, and a Poisson-coupled long-range field. It is a research playground for emergence (waves, clustering, persistent structures) and is not calibrated to any detector or experimental system.

## What this model is useful for

- Emergence and stability studies in coupled nonlinear fields.
- Regime transitions under parameter variation.
- Effective large-scale diagnostics (H, Λeff, Ωm/ΩΛ) in simulation units.
- Robustness testing across presets and initial conditions.

## Requirements

- Python 3.10+
- numpy
- scipy
- matplotlib

Install:

```
pip install -r requirements.txt
```

## Run

```
python main.py
```

Optional size:

```
python main.py --size 512
```

## Browser version (interactive)

The web demo lives in `web/` and runs entirely in the browser with a Canvas renderer.

Quick start (local):

```
cd web
python -m http.server 8000
```

Then open `http://localhost:8000` in your browser.

Notes:
- The browser version uses a fast Jacobi solver for the potential (no FFT) and is a visual port of the model.
- It starts paused; the first click ignites the sim.

## Controls

- Sliders: tune diffusion, gravity coupling, advection, freeze, damping, and Gray-Scott parameters live.
- Buttons: Pause/Run, Reset, Preset cycling, View toggles, Mode (Poke/Seed), Save PNG, Zoom.
- Mouse: click or drag to inject a wave impulse in Poke mode or a Gaussian seed in Seed mode.
- Keyboard:
  - Space: pause/resume
  - R: reset
  - 1/2/3: energy, mass, potential views
  - S: save a PNG snapshot
  - P: toggle click mode (Poke/Seed)
  - Panels: drag monitors or plots to reposition them inside the window

## How to use

1) Start the sim with `python main.py`. It opens paused.  
2) Click inside the main view to ignite the first reaction (Poke mode by default).  
3) Use the Mode button or `P` key to switch between:
   - Poke: injects a ring impulse to produce outward waves.
   - Seed: injects a Gaussian blob (legacy behavior).
4) Use the View button or keys 1/2/3 to switch layers (Energy/Mass/Potential).  
5) Use sliders to tune behavior. A few starting tips:
   - For more persistent waves: lower `damping`, slightly raise `wave_couple`.
   - For more structure: raise `kappa` and `A_adv` modestly.
6) Save a snapshot with the Save PNG button or `S`.

## Layout tips (if the UI looks messy)

- All panels are draggable. Click and drag:
  - the main view,
  - sliders,
  - buttons,
  - each monitor card,
  - each plot panel.
- If buttons overlap sliders, drag the buttons panel downward or the sliders panel upward.
- If the monitor text is too tall, widen the monitor cards by dragging them apart.
- If zoom is stuck on, click the Zoom button again or press the toolbar zoom icon.

## What changed

- Added effective cosmology diagnostics: H(t), rho(t), rolling fit H^2≈Aρ+B, Λeff, Ω values, regime classification, and a de Sitter stability check.
- CSV logging per run with timestamped files in `runs/`.
- UI monitor panel extended to show new diagnostics live.

## Model notes (scientific honesty)

This system is a deliberately simplified dynamical toy model. It combines:
- A Gray-Scott reaction-diffusion pair (U, V) to form spots/stripes.
- A frozen mass field (M) created when excitation crosses a threshold.
- A Poisson-solved potential from M to create long-range attraction.

The interactions are inspired by generic PDEs and pattern formation. They support quantitative diagnostics and regime analysis in simulation units, but they are not calibrated to any physical experiment or detector and should not be interpreted as such.

## Parameter tips

- For classic Turing patterns, set `kappa=0`, `A_adv=0`, `alpha=0` (freeze off).
- For stronger clustering, increase `kappa` and `A_adv` modestly.
- If the system collapses into a single blob, reduce `alpha` or raise `theta`.
- `time_dilation` and `time_floor` slow local updates near mass (toy time dilation); keep `time_floor` > 0 for stability.

## What each monitor means

- preset: active parameter preset name.
- paused: whether the simulation is running.
- frame: current tick count.
- dt: timestep per tick.
- view: current render layer.
- V stats: min/max/mean/total of excitation field.
- M stats: min/max/mean/total of frozen mass field.
- P stats: min/max/mean of Poisson potential.
- T stats: min/mean of the time-dilation map.
- bubble area: count of pixels above the bubble threshold.
- bubble R: effective radius from area, `sqrt(A/pi)`.
- detrended r: radius minus a moving average.
- f_peak: dominant FFT frequency of detrended radius (cycles/tick).
- rho_in: mean mass density inside the bubble mask.
- H, H2: expansion rate and squared expansion rate from R(t).
- fit A, B, r2: rolling least squares for H^2≈Aρ+B and fit quality.
- Lambda_eff: 3*B in simulation units.
- Omega_m/L: fractional contributions from Aρ and B (clamped for display).
- regime: qualitative label from Ω values and fit quality.
- de_sitter: late-time constant-H check.
- log: CSV path for the current run.
- components: connected components in the bubble mask.
- boundary |gradP|: mean potential gradient magnitude on bubble boundary.

## Continuum validation steps

1) Test 6B stability (continuum mode):
   - Run Test 6B and confirm N512 completes without blowing up.
2) dt invariance at fixed N=256 (continuum mode):
   - Run N=256 with dt=1.0, 0.5, 0.25 and confirm curves match closely.
3) Refinement trend:
   - Compare RMSE for N128 vs N256 and N256 vs N512; expect RMSE to drop with N.

## Tests 7-9 quick run

1) Test 7 (dt invariance):
   - Select "Test 7 dt invariance" and Run.
2) Test 8 (prediction plateau):
   - Select "Test 8 prediction plateau" and Run.
3) Test 9 (effective PDE fit):
   - Select "Test 9 effective PDE fit" and Run.
4) Analysis:
   - `python analysis/analyze_tests.py --runs_dir runs`

## Tests 10-12 quick run

1) Test 10 (stability N512, CFL):
   - Select "Test 10 stability N512" and Run.
2) Test 11 (continuum dt invariance N256):
   - Select "Test 11 continuum dt invariance N256" and Run.
3) Test 12 (refinement trend CFL):
   - Select "Test 12 refinement trend CFL" and Run.
4) Analysis:
   - `python analysis/analyze_tests.py --runs_dir runs`
   - New outputs: `stability_test10_summary.json`, `dt_invariance_rmse_test11.csv`, `refinement_rmse_test12.csv`
5) Keyframe PNGs:
   - Each run saves `snapshots/frame_000000_key.png`, mid, and last automatically.


## Licensing
- **Code** in this repository is released under the MIT License (see `LICENSE`).
- The archived Zenodo record may include additional assets (e.g., manuscript/figures) distributed under the license specified on Zenodo.
