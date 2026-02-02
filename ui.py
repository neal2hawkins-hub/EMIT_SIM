import json
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

from metrics import MetricsTracker
from run_logger import RunLogger
from sim import PRESETS, Simulation
from test_protocols import TESTS, TEST_ORDER

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
    }
)

PLOT_UPDATE_EVERY = 10
SERIES_LEN = 2048
UI_LAYOUT = "legacy"
LOG_FFT_SERIES = True
FFT_LOG_MAX = 256


class DraggablePanels:
    def __init__(self, fig, groups):
        self.fig = fig
        self.groups = groups
        self.active_group = None
        self.start_fig = None
        self.start_positions = None
        self.cid_press = fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.cid_release = fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_motion = fig.canvas.mpl_connect("motion_notify_event", self.on_motion)

    def on_press(self, event):
        for group in self.groups:
            if event.inaxes in group:
                self.active_group = group
                self.start_positions = [ax.get_position() for ax in group]
                self.start_fig = self.fig.transFigure.inverted().transform((event.x, event.y))
                break

    def on_release(self, _event):
        self.active_group = None
        self.start_fig = None
        self.start_positions = None

    def on_motion(self, event):
        if self.active_group is None or self.start_fig is None:
            return
        x, y = self.fig.transFigure.inverted().transform((event.x, event.y))
        dx = x - self.start_fig[0]
        dy = y - self.start_fig[1]
        for ax, start_pos in zip(self.active_group, self.start_positions):
            pos = list(start_pos.bounds)
            pos[0] += dx
            pos[1] += dy
            pos[0] = min(max(pos[0], 0.02), 0.98 - pos[2])
            pos[1] = min(max(pos[1], 0.02), 0.98 - pos[3])
            ax.set_position(pos)
        self.fig.canvas.draw_idle()


class EmitUI:
    def __init__(self, sim):
        self.sim = sim
        self.metrics = MetricsTracker(series_len=SERIES_LEN, plot_update_every=PLOT_UPDATE_EVERY)
        self.paused = True
        self.started = False
        self.frame = 0
        self.preset_index = 0
        self.view_mode = "energy"
        self.mouse_down = False
        self.zoom_active = False
        self.click_mode = "poke"
        self.run_logger = None
        self._force_log_frame0 = False
        self._run_config = None
        self.test_mode = "Off"
        self.test_active = False
        self.test_run_index = 0
        self.test_frame_in_run = 0
        self.test_plan = []
        self.test_base_dir = None
        self.test_current_spec = None
        self.test_name = None
        self.test_options = list(TEST_ORDER)
        self._preset_map = {name: preset for name, preset in PRESETS}
        self.test_include_seed_in_run_id = False

        self.fig = plt.figure(figsize=(13, 8))
        self.fig.patch.set_facecolor("#f5f5f5")
        if hasattr(self.fig.canvas.manager, "set_window_title"):
            self.fig.canvas.manager.set_window_title("EMIT")

        gs = self.fig.add_gridspec(1, 2, width_ratios=[3.4, 1.6], wspace=0.25)
        self.ax_main = self.fig.add_subplot(gs[0, 0])
        gs_right = gs[0, 1].subgridspec(4, 1, height_ratios=[5.0, 0.9, 1.3, 1.1], hspace=0.35)

        self.ax_controls = self.fig.add_subplot(gs_right[0, 0])
        self.ax_controls.set_axis_off()
        self.ax_buttons = self.fig.add_subplot(gs_right[1, 0])
        self.ax_buttons.set_axis_off()
        self.ax_monitors = self.fig.add_subplot(gs_right[2, 0])
        self.ax_monitors.set_axis_off()
        gs_plots = gs_right[3, 0].subgridspec(2, 1, hspace=0.35)
        self.ax_r = self.fig.add_subplot(gs_plots[0, 0])
        self.ax_fft = self.fig.add_subplot(gs_plots[1, 0])

        self.img = self.ax_main.imshow(self.sim.state["V"], interpolation="nearest", cmap="inferno")
        self.ax_main.set_title("EMIT (Energy = V)")
        self.ax_main.set_axis_off()
        self.ax_main.set_aspect("equal", adjustable="box")

        self.sliders = {}
        self.control_axes = []
        self.button_axes = []
        self._build_controls()
        self._build_buttons()
        self._build_monitors()
        self._build_plots()
        self._bind_events()
        self._draggables = DraggablePanels(
            self.fig,
            [
                [self.ax_main],
                [self.ax_controls] + self.control_axes,
                [self.ax_buttons] + self.button_axes,
                [self.ax_monitors_left],
                [self.ax_monitors_right],
                [self.ax_r],
                [self.ax_fft],
            ],
        )

        self.metrics.start_run_log(PRESETS[self.preset_index][0])
        self._start_run_logger(PRESETS[self.preset_index][0])
        self.anim = FuncAnimation(self.fig, self.update, interval=16, blit=False)

    def _build_controls(self):
        slider_specs = [
            ("dt", 0.25, 3.0),
            ("D_E", 0.0, 0.2),
            ("kappa", 0.0, 3.0),
            ("A_adv", 0.0, 1.5),
            ("theta", 0.0, 1.0),
            ("alpha", 0.0, 0.5),
            ("beta", 0.0, 0.5),
            ("damping", 0.0, 0.02),
            ("floor", 0.0, 0.005),
            ("mass_decay", 0.0, 0.01),
            ("mass_release", 0.0, 0.01),
            ("time_dilation", 0.0, 3.0),
            ("time_floor", 0.0, 0.5),
            ("D_u", 0.0, 0.3),
            ("D_v", 0.0, 0.3),
            ("F", 0.0, 0.08),
            ("k", 0.0, 0.08),
            ("bubble_thresh", 0.0, 1.0),
            ("display_gamma", 0.5, 2.5),
            ("min_component_area", 5, 200),
        ]
        bbox = self.ax_controls.get_position()
        x0, y0, w, h = bbox.bounds
        slider_area_h = h * 0.96
        slider_top = y0 + h - 0.02
        slider_left = x0 + 0.04 * w
        slider_width = w * 0.85

        n = len(slider_specs)
        step = slider_area_h / n
        slider_h = step * 0.52
        for idx, (name, vmin, vmax) in enumerate(slider_specs):
            y = slider_top - (idx + 1) * step
            ax = self.fig.add_axes([slider_left, y, slider_width, slider_h])
            if name == "dt":
                valinit = self.sim.dt
                valstep = 0.25
            elif name == "min_component_area":
                valinit = self.sim.params[name]
                valstep = 1
            else:
                valinit = self.sim.params[name]
                valstep = None
            s = Slider(
                ax,
                name,
                vmin,
                vmax,
                valinit=valinit,
                valfmt="%.4f",
                valstep=valstep,
            )
            self.sliders[name] = s
            self.control_axes.append(ax)

        # button labels set in _build_buttons()

    def _build_buttons(self):
        bbox = self.ax_buttons.get_position()
        x0, y0, w, h = bbox.bounds
        btn_w = w * 0.22
        btn_h = h * 0.42
        gap = w * 0.02
        row1_y = y0 + h * 0.55
        row2_y = y0 + h * 0.08
        x_start = x0 + 0.01

        btn_pause_ax = self.fig.add_axes([x_start + 0 * (btn_w + gap), row1_y, btn_w, btn_h])
        btn_reset_ax = self.fig.add_axes([x_start + 1 * (btn_w + gap), row1_y, btn_w, btn_h])
        btn_preset_ax = self.fig.add_axes([x_start + 2 * (btn_w + gap), row1_y, btn_w, btn_h])
        btn_mode_ax = self.fig.add_axes([x_start + 3 * (btn_w + gap), row1_y, btn_w, btn_h])
        btn_view_ax = self.fig.add_axes([x_start + 0 * (btn_w + gap), row2_y, btn_w, btn_h])
        btn_save_ax = self.fig.add_axes([x_start + 1 * (btn_w + gap), row2_y, btn_w, btn_h])
        btn_zoom_ax = self.fig.add_axes([x_start + 2 * (btn_w + gap), row2_y, btn_w, btn_h])
        btn_test_ax = self.fig.add_axes([x_start + 3 * (btn_w + gap), row2_y, btn_w, btn_h])
        self.btn_pause = Button(btn_pause_ax, "Run")
        self.btn_reset = Button(btn_reset_ax, "Reset")
        self.btn_preset = Button(btn_preset_ax, "Preset")
        self.btn_mode = Button(btn_mode_ax, "Mode: Poke")
        self.btn_view = Button(btn_view_ax, "View")
        self.btn_save = Button(btn_save_ax, "Save PNG")
        self.btn_zoom = Button(btn_zoom_ax, "Zoom")
        self.btn_test = Button(btn_test_ax, "Test: Off")
        self.button_axes.extend(
            [
                btn_pause_ax,
                btn_reset_ax,
                btn_preset_ax,
                btn_mode_ax,
                btn_view_ax,
                btn_save_ax,
                btn_zoom_ax,
                btn_test_ax,
            ]
        )
        self.btn_preset.label.set_text(f"Preset: {PRESETS[self.preset_index][0]}")
        self.btn_view.label.set_text("View: Energy")

    def _build_monitors(self):
        bbox = self.ax_monitors.get_position()
        x0, y0, w, h = bbox.bounds
        self.ax_monitors_left = self.fig.add_axes([x0, y0, w * 0.48, h])
        self.ax_monitors_right = self.fig.add_axes([x0 + w * 0.52, y0, w * 0.48, h])
        self.ax_monitors_left.set_axis_off()
        self.ax_monitors_right.set_axis_off()

        self.monitor_text = self.ax_monitors_left.text(
            0.0,
            1.0,
            "",
            va="top",
            ha="left",
            family="monospace",
            bbox={"boxstyle": "round", "facecolor": "#ffffff", "edgecolor": "#dddddd"},
        )
        self.monitor_text_right = self.ax_monitors_right.text(
            0.0,
            1.0,
            "",
            va="top",
            ha="left",
            family="monospace",
            bbox={"boxstyle": "round", "facecolor": "#ffffff", "edgecolor": "#dddddd"},
        )

    def _build_plots(self):
        self.ax_r.set_title("R(t)")
        self.ax_r.set_xlabel("samples")
        self.ax_r.set_ylabel("radius (px)")
        self.line_r, = self.ax_r.plot([], [], color="tab:blue")

        self.ax_fft.set_title("FFT |r(t)|")
        self.ax_fft.set_xlabel("freq")
        self.ax_fft.set_ylabel("magnitude")
        self.line_fft, = self.ax_fft.plot([], [], color="tab:orange")

    def _bind_events(self):
        self.btn_pause.on_clicked(self._on_pause)
        self.btn_reset.on_clicked(self._on_reset)
        self.btn_preset.on_clicked(self._on_preset)
        self.btn_mode.on_clicked(self._on_mode)
        self.btn_view.on_clicked(self._on_view)
        self.btn_save.on_clicked(self._on_save)
        self.btn_zoom.on_clicked(self._on_zoom)
        self.btn_test.on_clicked(self._on_test)

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _sync_params(self):
        for key, slider in self.sliders.items():
            if key == "dt":
                self.sim.dt = float(slider.val)
            elif key == "min_component_area":
                self.sim.params[key] = int(slider.val)
            else:
                self.sim.params[key] = float(slider.val)

    def _current_params(self):
        params = {"dt": float(self.sim.dt)}
        for key, slider in self.sliders.items():
            if key == "dt":
                params["dt"] = float(self.sim.dt)
            elif key == "min_component_area":
                params[key] = int(slider.val)
            else:
                params[key] = float(slider.val)
        if "min_component_area" in params:
            params["min_area"] = params["min_component_area"]
        for key in ("v_cap", "m_cap"):
            if key in self.sim.params:
                params[key] = float(self.sim.params[key])
        return params

    def _param_keys(self):
        base = [
            "dt",
            "D_E",
            "alpha",
            "beta",
            "theta",
            "kappa",
            "A_adv",
            "damping",
            "mass_decay",
            "time_dilation",
            "time_floor",
            "bubble_thresh",
            "min_area",
            "v_cap",
            "m_cap",
        ]
        slider_keys = [key for key in self.sliders.keys() if key not in base]
        return base + slider_keys

    def _start_run_logger(self, preset_name, output_dir="runs", run_id_prefix=None, config=None):
        if self.run_logger is not None:
            self.run_logger.close()
        if run_id_prefix is None:
            run_id = time.strftime("%Y%m%d_%H%M%S")
            run_dir = None
        else:
            run_id = str(run_id_prefix)
            run_dir = os.path.join(output_dir, run_id)
        self.run_logger = RunLogger(
            output_dir=output_dir,
            run_id=run_id,
            preset_name=preset_name,
            log_every=self.metrics.log_every,
            startup_frames=100,
            param_keys=self._param_keys(),
            run_dir=run_dir,
        )
        if config is None:
            config = {
                "run_id": run_id,
                "preset_name": preset_name,
                "grid_size": self.sim.size,
                "seed": self.sim.seed,
                "version": "unknown",
                "dt": float(self.sim.dt),
                "params": self._current_params(),
                "invariant_fit_c": 0.0,
                "dx": float(self.sim.dx),
                "continuum_mode": bool(self.sim.params.get("continuum_mode", False)),
            }
        self._run_config = dict(config)
        self.run_logger.write_config(config)

    def _log_frame_if_needed(self, metrics):
        if self.run_logger is None:
            return
        if metrics is not None and self._run_config is not None:
            fit_c = metrics.get("invariant_fit_c")
            if fit_c is not None and np.isfinite(fit_c):
                if self._run_config.get("invariant_fit_c") is None:
                    self._run_config["invariant_fit_c"] = float(fit_c)
                elif self._run_config.get("invariant_fit_c") == 0.0 and fit_c != 0.0:
                    self._run_config["invariant_fit_c"] = float(fit_c)
                self.run_logger.write_config(self._run_config)
        params = self._current_params()
        t_value = self.frame * self.sim.dt
        fft_freqs = None
        fft_mag = None
        fft_peak_mag = np.nan
        fft_n = 0
        if self.metrics.last_fft is not None and self.metrics.last_freqs is not None:
            fft_n = int(len(self.metrics.last_fft))
            if fft_n > 0:
                fft_peak_mag = float(np.max(self.metrics.last_fft))
            if LOG_FFT_SERIES:
                freqs = self.metrics.last_freqs
                mag = self.metrics.last_fft
                if freqs is not None and mag is not None and len(freqs) == len(mag) and len(freqs) > 0:
                    if len(freqs) > FFT_LOG_MAX:
                        idx = np.linspace(0, len(freqs) - 1, FFT_LOG_MAX).astype(int)
                        freqs = freqs[idx]
                        mag = mag[idx]
                    fft_freqs = json.dumps([float(x) for x in freqs])
                    fft_mag = json.dumps([float(x) for x in mag])

        extras = {
            "r_series_last": float(self.metrics.r_series.values()[-1]) if len(self.metrics.r_series) > 0 else np.nan,
            "r_series_len": int(len(self.metrics.r_series)),
            "fft_n": fft_n,
            "fft_peak_mag": fft_peak_mag,
            "fft_freqs": fft_freqs if fft_freqs is not None else "",
            "fft_mag": fft_mag if fft_mag is not None else "",
        }
        self.run_logger.log_frame(
            self.frame,
            t_value,
            params=params,
            metrics=metrics,
            extras=extras,
        )

    def _make_action_from_event(self, event):
        if event.xdata is None or event.ydata is None:
            return None
        x = int(event.xdata)
        y = int(event.ydata)
        strength = float(self.sim.params.get("poke", 1.0))
        sigma = float(self.sim.params.get("poke_sigma", 3.0))
        radius = int(max(2.0, sigma * 3.0))
        action_type = "poke" if self.click_mode == "poke" else "seed"
        if action_type == "poke" and "Wdot" in self.sim.state:
            field = "Wdot"
        else:
            field = "V"
        return {
            "type": action_type,
            "x": x,
            "y": y,
            "radius": radius,
            "strength": strength,
            "sigma": sigma,
            "field": field,
        }

    def _apply_action_from_event(self, event, source):
        action = self._make_action_from_event(event)
        if action is None:
            return
        result = self.sim.apply_action(action)
        if self.run_logger is not None:
            t_value = self.frame * self.sim.dt
            if source == "click":
                payload = {
                    "x": action["x"],
                    "y": action["y"],
                    "button": getattr(event, "button", None),
                    "modifiers": getattr(event, "modifiers", None),
                    "strength": action["strength"],
                    "radius": action["radius"],
                    "mode": self.click_mode,
                    "triggered_action": bool(result.get("applied")),
                }
                self.run_logger.log_event(self.frame, t_value, "click", payload)
            if result.get("applied"):
                payload = {}
                payload.update(action)
                payload.update(result)
                payload["source"] = source
                payload["params"] = self._current_params()
                self.run_logger.log_event(self.frame, t_value, "action", payload)

    def _on_pause(self, _event):
        if self.test_mode != "Off" and not self.test_active:
            self._start_test_runner()
            return
        self.paused = not self.paused
        self.btn_pause.label.set_text("Pause" if not self.paused else "Run")

    def _on_reset(self, _event):
        if self.test_active:
            self._stop_test_runner("reset")
        self.sim.reset()
        self.frame = 0
        self.started = False
        self.paused = True
        self.btn_pause.label.set_text("Run")
        self.metrics.start_run_log(PRESETS[self.preset_index][0])
        self._start_run_logger(PRESETS[self.preset_index][0])

    def _on_preset(self, _event):
        if self.test_active:
            self._stop_test_runner("preset change")
        self.preset_index = (self.preset_index + 1) % len(PRESETS)
        name, preset = PRESETS[self.preset_index]
        self.sim.apply_preset(preset)
        for key, slider in self.sliders.items():
            if key in preset:
                slider.set_val(preset[key])
        self.btn_preset.label.set_text(f"Preset: {name}")
        self.metrics.start_run_log(name)
        self._start_run_logger(name)

    def _on_mode(self, _event):
        self.click_mode = "seed" if self.click_mode == "poke" else "poke"
        label = "Mode: Seed" if self.click_mode == "seed" else "Mode: Poke"
        self.btn_mode.label.set_text(label)

    def _on_view(self, _event):
        if self.view_mode == "energy":
            self.view_mode = "mass"
            self.btn_view.label.set_text("View: Mass")
            self.ax_main.set_title("EMIT (Mass)")
            self.img.set_cmap("magma")
        elif self.view_mode == "mass":
            self.view_mode = "potential"
            self.btn_view.label.set_text("View: Potential")
            self.ax_main.set_title("EMIT (Potential)")
            self.img.set_cmap("coolwarm")
        elif self.view_mode == "potential":
            self.view_mode = "composite"
            self.btn_view.label.set_text("View: Composite")
            self.ax_main.set_title("EMIT (Composite)")
        else:
            self.view_mode = "energy"
            self.btn_view.label.set_text("View: Energy")
            self.ax_main.set_title("EMIT (Energy = V)")
            self.img.set_cmap("inferno")

    def _on_save(self, _event):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.fig.savefig(f"emit_snapshot_{ts}.png", dpi=150)

    def _on_zoom(self, _event):
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if toolbar is None:
            return
        toolbar.zoom()
        self.zoom_active = not self.zoom_active
        self.btn_zoom.label.set_text("Zoom On" if self.zoom_active else "Zoom")

    def _on_test(self, _event):
        if self.test_active:
            print("Test runner active. Stop or wait for completion to change tests.")
            return
        if self.test_mode not in self.test_options:
            self.test_mode = "Off"
        idx = (self.test_options.index(self.test_mode) + 1) % len(self.test_options)
        self.test_mode = self.test_options[idx]
        if self.test_mode == "Off":
            label = "Test: Off"
        else:
            label = f"Test: {TESTS[self.test_mode]['label']}"
        self.btn_test.label.set_text(label)

    def _on_press(self, event):
        if event.inaxes != self.ax_main:
            return
        if self.test_active:
            return
        if not self.started:
            self.started = True
            self.paused = False
            self.btn_pause.label.set_text("Pause")
            self._force_log_frame0 = True
        self.mouse_down = True
        self._sync_params()
        self._apply_action_from_event(event, source="click")

    def _on_release(self, _event):
        self.mouse_down = False

    def _on_move(self, event):
        if self.mouse_down:
            if self.test_active:
                return
            self._sync_params()
            self._apply_action_from_event(event, source="drag")

    def _on_key(self, event):
        if event.key == " ":
            self._on_pause(None)
        elif event.key in ("r", "R"):
            self._on_reset(None)
        elif event.key == "1":
            self.view_mode = "energy"
            self.btn_view.label.set_text("View: Energy")
            self.ax_main.set_title("EMIT (Energy = V)")
            self.img.set_cmap("inferno")
        elif event.key == "2":
            self.view_mode = "mass"
            self.btn_view.label.set_text("View: Mass")
            self.ax_main.set_title("EMIT (Mass)")
            self.img.set_cmap("magma")
        elif event.key == "3":
            self.view_mode = "potential"
            self.btn_view.label.set_text("View: Potential")
            self.ax_main.set_title("EMIT (Potential)")
            self.img.set_cmap("coolwarm")
        elif event.key in ("s", "S"):
            ts = time.strftime("%Y%m%d_%H%M%S")
            self.fig.savefig(f"emit_snapshot_{ts}.png", dpi=150)
        elif event.key in ("p", "P"):
            self._on_mode(None)

    def _add_energy(self, event):
        self._apply_action_from_event(event, source="click")

    def _reset_metrics(self):
        self.metrics.close_run_log()
        self.metrics = MetricsTracker(series_len=SERIES_LEN, plot_update_every=PLOT_UPDATE_EVERY)

    def _stop_test_runner(self, reason):
        if not self.test_active:
            return
        if self.run_logger is not None:
            self.run_logger.close()
            self.run_logger = None
        self.test_active = False
        self.test_plan = []
        self.test_run_index = 0
        self.test_frame_in_run = 0
        self.test_current_spec = None
        self.test_base_dir = None
        self.paused = True
        self.btn_pause.label.set_text("Run")
        print(f"Test runner stopped ({reason}).")

    def _start_test_runner(self):
        test_def = TESTS.get(self.test_mode)
        if test_def is None:
            print(f"Unknown test mode: {self.test_mode}")
            return
        self.test_name = test_def["name"]
        self.test_plan = [dict(spec) for spec in test_def["runs"]]
        for spec in self.test_plan:
            spec["_poke_done"] = False
        seen = set()
        include_seed = False
        for spec in self.test_plan:
            key = (spec.get("grid_size"), spec.get("dt"))
            if key in seen:
                include_seed = True
                break
            seen.add(key)
        self.test_include_seed_in_run_id = include_seed
        self.test_run_index = 0
        self.test_frame_in_run = 0
        self.test_active = True
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.test_base_dir = os.path.join("runs", f"{stamp}_{self.test_name}")
        os.makedirs(self.test_base_dir, exist_ok=True)
        self._start_test_run(self.test_plan[0])

    def _ensure_sim_size(self, grid_size, seed, dt, dx):
        if self.sim.size == grid_size:
            if dx is not None and abs(self.sim.dx - dx) > 1e-12:
                self.sim.set_dx(dx)
            return
        self.sim = Simulation(size=grid_size, seed=seed, dt=dt, dx=dx if dx is not None else 1.0)
        self.img.set_data(self.sim.state["V"])

    def _start_test_run(self, spec):
        self.test_current_spec = spec
        self.frame = 0
        self.test_frame_in_run = 0
        self.started = True
        self.paused = False
        self.btn_pause.label.set_text("Pause")
        self._force_log_frame0 = True

        grid_size = int(spec["grid_size"])
        seed = int(spec["seed"])
        dt = float(spec["dt"])
        dx = spec.get("dx")
        preset_name = spec["preset_name"]

        self._ensure_sim_size(grid_size, seed, dt, dx)
        self.sim.reset(seed=seed)
        preset = self._preset_map.get(preset_name)
        if preset is not None:
            self.sim.apply_preset(preset)
            for key, slider in self.sliders.items():
                if key in preset:
                    slider.set_val(preset[key])
        for idx, (name, _) in enumerate(PRESETS):
            if name == preset_name:
                self.preset_index = idx
                self.btn_preset.label.set_text(f"Preset: {name}")
                break

        self.sliders["dt"].set_val(dt)
        self._sync_params()

        continuum_mode = bool(spec.get("continuum_mode", False))
        self.sim.params["continuum_mode"] = continuum_mode

        overrides = spec.get("param_overrides")
        if overrides:
            for key, value in overrides.items():
                self.sim.params[key] = float(value)
                if key in self.sliders:
                    self.sliders[key].set_val(float(value))

        self.click_mode = spec.get("click_mode", "poke")
        self.btn_mode.label.set_text("Mode: Seed" if self.click_mode == "seed" else "Mode: Poke")

        self._reset_metrics()
        self.metrics.start_run_log(preset_name)

        run_id = f"N{grid_size}_dt{dt}"
        if self.test_include_seed_in_run_id and spec.get("seed") is not None:
            run_id = f"{run_id}_seed{seed}"
        config = {
            "run_id": run_id,
            "test_name": self.test_name,
            "test_label": TESTS[self.test_mode]["label"],
            "run_index": self.test_run_index,
            "preset_name": preset_name,
            "grid_size": grid_size,
            "seed": seed,
            "dt": dt,
            "steps": int(spec["steps"]),
            "poke_frame": int(spec["poke_frame"]),
            "poke_strength": float(spec["poke_strength"]),
            "poke_radius": int(spec["poke_radius"]),
            "poke_sigma": float(spec["poke_sigma"]),
            "field": spec.get("field", "auto"),
            "click_mode": self.click_mode,
            "snapshot_every": spec.get("snapshot_every"),
            "snapshot_frames": spec.get("snapshot_frames"),
            "snapshot_mode": spec.get("snapshot_mode"),
            "snapshot_times": spec.get("snapshot_times"),
            "save_png_snapshots": spec.get("save_png_snapshots"),
            "save_npz_snapshots": spec.get("save_npz_snapshots"),
            "always_save_keyframes_png": spec.get("always_save_keyframes_png"),
            "keyframe_frames": None,
            "params": self._current_params(),
            "invariant_fit_c": 0.0,
            "test_physical_L": spec.get("test_physical_L"),
            "dx": spec.get("dx"),
            "dx_ref": spec.get("dx_ref"),
            "dt_ref": spec.get("dt_ref"),
            "dt_scaled_rule": None if not spec.get("scaling_rules") else spec.get("scaling_rules", {}).get("dt_scaled_rule"),
            "params_scaled": None if not spec.get("scaling_rules") else spec.get("scaling_rules", {}).get("params_scaled"),
            "radius_rule": None if not spec.get("scaling_rules") else spec.get("scaling_rules", {}).get("radius_rule"),
            "strength_rule": None if not spec.get("scaling_rules") else spec.get("scaling_rules", {}).get("strength_rule"),
            "N_ref": spec.get("n_ref"),
            "scaling_rules": spec.get("scaling_rules"),
            "rescaled_params": spec.get("param_overrides"),
            "physical_T": spec.get("physical_T"),
            "continuum_mode": bool(spec.get("continuum_mode", False)),
        }
        self._start_run_logger(
            preset_name,
            output_dir=self.test_base_dir,
            run_id_prefix=run_id,
            config=config,
        )
        if config.get("always_save_keyframes_png"):
            total_frames = int(spec.get("steps", 0))
            mid_frame = total_frames // 2 if total_frames > 0 else 0
            self._run_config["keyframe_frames"] = [0, mid_frame, max(0, total_frames - 1)]
            self.run_logger.write_config(self._run_config)

    def _finish_test_run(self):
        if self.run_logger is not None:
            self.run_logger.close()
            self.run_logger = None

    def _finish_test_runner(self):
        self.test_active = False
        self.test_current_spec = None
        self.test_frame_in_run = 0
        self.paused = True
        self.btn_pause.label.set_text("Run")
        preset_name = PRESETS[self.preset_index][0]
        self.metrics.start_run_log(preset_name)
        self._start_run_logger(preset_name)
        print(f"Done. Test outputs in: {self.test_base_dir}")

    def _advance_test_run(self):
        self._finish_test_run()
        self.test_run_index += 1
        if self.test_run_index >= len(self.test_plan):
            self._finish_test_runner()
            return
        self._start_test_run(self.test_plan[self.test_run_index])

    def _maybe_apply_test_poke(self):
        if not self.test_active or self.test_current_spec is None:
            return
        if self.test_current_spec.get("_poke_done"):
            return
        if self.frame != int(self.test_current_spec["poke_frame"]):
            return
        field = self.test_current_spec.get("field", "auto")
        if field == "auto":
            field = "Wdot" if "Wdot" in self.sim.state else "V"
        x = self.sim.size // 2
        y = self.sim.size // 2
        action = {
            "type": "poke",
            "x": x,
            "y": y,
            "radius": int(self.test_current_spec["poke_radius"]),
            "strength": float(self.test_current_spec["poke_strength"]),
            "sigma": float(self.test_current_spec["poke_sigma"]),
            "field": field,
        }
        result = self.sim.apply_action(action)
        self.test_current_spec["_poke_done"] = True
        if self.run_logger is not None and result.get("applied"):
            t_value = self.frame * self.sim.dt
            payload = {}
            payload.update(action)
            payload.update(result)
            payload["source"] = "test"
            payload["params"] = self._current_params()
            self.run_logger.log_event(self.frame, t_value, "action", payload)

    def _maybe_save_snapshot(self):
        if not self.test_active or self.test_current_spec is None:
            return
        keyframe_enabled = bool(self.test_current_spec.get("always_save_keyframes_png"))
        if keyframe_enabled:
            total_frames = int(self.test_current_spec.get("steps", 0))
            mid_frame = total_frames // 2 if total_frames > 0 else 0
            keyframes = {0, mid_frame, max(0, total_frames - 1)}
            if self.frame in keyframes and self.run_logger is not None:
                snapshots_dir = os.path.join(self.run_logger.run_dir, "snapshots")
                os.makedirs(snapshots_dir, exist_ok=True)
                n_val = self.sim.size
                dt_val = self.sim.dt
                dx_val = self.sim.dx
                tag = f"N{n_val}_dt{dt_val:.2f}_dx{dx_val:.5f}"
                gamma = float(self.sim.params.get("display_gamma", 1.0))
                gamma = max(gamma, 1e-6)
                v = self.sim.state.get("V")
                if v is not None:
                    v_norm = v / (v.max() + 1e-6)
                    v_disp = np.power(v_norm, 1.0 / gamma)
                    path = os.path.join(
                        snapshots_dir, f"frame_{self.frame:06d}_key_{tag}_energy.png"
                    )
                    plt.imsave(path, v_disp, cmap="inferno")
                m = self.sim.state.get("M")
                if m is not None:
                    m_norm = m / (m.max() + 1e-6)
                    m_disp = np.power(m_norm, 1.0 / gamma)
                    path = os.path.join(
                        snapshots_dir, f"frame_{self.frame:06d}_key_{tag}_mass.png"
                    )
                    plt.imsave(path, m_disp, cmap="magma")
                phi = self.sim.last_phi
                if phi is not None:
                    path = os.path.join(
                        snapshots_dir, f"frame_{self.frame:06d}_key_{tag}_phi.png"
                    )
                    plt.imsave(path, phi, cmap="coolwarm")
        snap_frames = self.test_current_spec.get("snapshot_frames")
        snap_mode = self.test_current_spec.get("snapshot_mode")
        save_png = self.test_current_spec.get("save_png_snapshots")
        save_npz = self.test_current_spec.get("save_npz_snapshots")
        if snap_frames:
            if self.run_logger is None:
                return
            if any(frame == -1 for frame in snap_frames):
                steps = int(self.test_current_spec.get("steps", 0))
                snap_frames = [steps if frame == -1 else int(frame) for frame in snap_frames]
            if self.frame not in snap_frames:
                return
            out_dir = os.path.join(self.run_logger.run_dir, "snapshots")
            os.makedirs(out_dir, exist_ok=True)
            if save_png or snap_mode == "png":
                path = os.path.join(out_dir, f"snap_{self.frame:07d}.png")
                v = self.sim.state.get("V")
                plt.imsave(path, v, cmap="inferno")
            if save_npz:
                npz_dir = os.path.join(self.run_logger.run_dir, "snapshots_npz")
                os.makedirs(npz_dir, exist_ok=True)
                t_value = self.frame * self.sim.dt
                npz_path = os.path.join(npz_dir, f"frame_{self.frame:07d}_t{t_value:.3f}.npz")
                payload = {
                    "V": self.sim.state.get("V"),
                    "M": self.sim.state.get("M"),
                    "phi": self.sim.last_phi,
                    "frame": self.frame,
                    "t": t_value,
                    "dt": self.sim.dt,
                    "dx": self.sim.dx,
                    "params": self._current_params(),
                }
                if "W" in self.sim.state:
                    payload["W"] = self.sim.state.get("W")
                if "Wdot" in self.sim.state:
                    payload["Wdot"] = self.sim.state.get("Wdot")
                np.savez_compressed(npz_path, **payload)
            return
        snap_every = self.test_current_spec.get("snapshot_every")
        if not snap_every:
            return
        if self.frame <= 0 or (self.frame % snap_every) != 0:
            return
        if self.run_logger is None:
            return
        out_dir = os.path.join(self.run_logger.run_dir, "snapshots")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"snap_{self.frame:07d}.npz")
        np.savez(
            path,
            V=self.sim.state.get("V"),
            M=self.sim.state.get("M"),
            phi=self.sim.last_phi,
            frame=self.frame,
            dt=self.sim.dt,
            params=self._current_params(),
        )

    def _test_status_line(self):
        if self.test_mode == "Off":
            return "test: off"
        if not self.test_active or self.test_current_spec is None:
            label = TESTS.get(self.test_mode, {}).get("label", self.test_mode)
            return f"test: {label} (idle)"
        spec = self.test_current_spec
        total = len(self.test_plan)
        run_idx = self.test_run_index + 1
        return (
            f"TEST: {run_idx}/{total} "
            f"N={spec['grid_size']} dt={spec['dt']} "
            f"frame={self.frame}/{spec['steps']}"
        )

    def _update_monitors(self, metrics):
        def fmt(value, digits=4):
            if value is None or not np.isfinite(value):
                return "NA"
            return f"{value:.{digits}f}"

        def fmt_sci(value):
            if value is None or not np.isfinite(value):
                return "NA"
            return f"{value:.3e}"

        left_lines = [
            self._test_status_line(),
            f"preset: {PRESETS[self.preset_index][0]}",
            f"paused: {self.paused}",
            f"frame: {self.frame}",
            f"dt: {self.sim.dt:.3f}",
            f"view: {self.view_mode}",
            f"bubble area: {metrics['area']}",
            f"bubble R: {metrics['radius']:.2f}",
            f"rho_in: {fmt(metrics.get('rho'))}",
            f"H: {fmt(metrics.get('h'))}  H2: {fmt(metrics.get('h2'))}",
            f"detrended r: {metrics['r_det_last']:.4f}",
            f"f_peak: {metrics['f_peak']:.6f}",
            f"fft_n: {len(self.metrics.r_series)}",
            f"df: {1.0/(max(1,len(self.metrics.r_series))*self.sim.dt):.9f}",
            f"fit A: {fmt_sci(metrics.get('fit_a'))}",
            f"fit B: {fmt_sci(metrics.get('fit_b'))}",
            f"fit r2: {fmt(metrics.get('fit_r2'), digits=3)}",
            f"Lambda_eff: {fmt_sci(metrics.get('lambda_eff'))}",
            f"Omega_m/L: {fmt(metrics.get('omega_m_disp'), digits=3)} / {fmt(metrics.get('omega_l_disp'), digits=3)}",
            f"regime: {metrics.get('regime', 'NA')}",
            f"de_sitter: {'pass' if metrics.get('de_sitter') else 'fail'}",
            f"log: {metrics.get('log_path') or 'off'}",
        ]

        right_lines = [
            f"click: {self.click_mode}",
            f"wall s: {fmt(metrics.get('wall_elapsed'))}",
            f"sim s: {fmt(metrics.get('sim_elapsed'))}",
            f"fps: {fmt(metrics.get('fps_inst'), digits=1)}  ema: {fmt(metrics.get('fps_ema'), digits=1)}",
            f"V total: {fmt(metrics.get('v_total'), digits=2)}",
            f"dV/dt w: {fmt_sci(metrics.get('dE_dt_wall'))}",
            f"dV/dt s: {fmt_sci(metrics.get('dE_dt_sim'))}",
            f"dV/dt avg: {fmt_sci(metrics.get('dE_dt_wall_avg'))}",
            f"M total: {fmt(metrics.get('m_total'), digits=2)}",
            f"dM/dt w: {fmt_sci(metrics.get('dM_dt_wall'))}",
            f"dM/dt s: {fmt_sci(metrics.get('dM_dt_sim'))}",
            f"dM/dt avg: {fmt_sci(metrics.get('dM_dt_wall_avg'))}",
            f"dR/dt w: {fmt_sci(metrics.get('dR_dt_wall'))}",
            f"dR/dt s: {fmt_sci(metrics.get('dR_dt_sim'))}",
            f"dR/dt avg: {fmt_sci(metrics.get('dR_dt_wall_avg'))}",
            f"components: {metrics.get('components_raw', 0)} / {metrics.get('components_filtered', 0)}",
            f"min_area: {self.sim.params.get('min_component_area')}",
            f"boundary |gradP|: {metrics['pressure']:.4f}",
        ]
        self.monitor_text.set_text("\n".join(left_lines))
        self.monitor_text_right.set_text("\n".join(right_lines))

    def _update_plots(self):
        series = self.metrics.r_series.values()
        if len(series) > 1:
            xs = np.arange(len(series))
            self.line_r.set_data(xs, series)
            self.ax_r.set_xlim(0, max(10, len(series)))
            self.ax_r.set_ylim(0, max(1.0, float(series.max()) * 1.1))

        if self.metrics.last_fft is not None and self.metrics.last_freqs is not None:
            self.line_fft.set_data(self.metrics.last_freqs, self.metrics.last_fft)
            self.ax_fft.set_xlim(0, float(self.metrics.last_freqs.max()))
            self.ax_fft.set_ylim(0, float(self.metrics.last_fft.max()) * 1.05 + 1e-6)

    def update(self, _frame):
        self._sync_params()
        end_test_run = False

        if self._force_log_frame0:
            metrics0 = self.metrics.update(
                self.sim.state,
                self.sim.last_phi,
                self.sim.dt,
                0,
                self.sim.params["bubble_thresh"],
                self.sim.state["V"],
                self.sim.params.get("min_component_area", 30),
                dM_map=self.sim.last_dM_map,
            )
            self._log_frame_if_needed(metrics0)
            self._force_log_frame0 = False

        if not self.paused:
            if self.test_active:
                self._maybe_apply_test_poke()
            self.sim.step()
            self.frame += 1
            self.test_frame_in_run = self.frame
            if (
                self.test_active
                and self.test_current_spec is not None
                and self.frame >= int(self.test_current_spec["steps"])
            ):
                end_test_run = True

        gamma = float(self.sim.params.get("display_gamma", 1.0))
        gamma = max(gamma, 1e-6)
        view_field = None

        if self.view_mode == "energy":
            v = self.sim.state["V"]
            v_norm = v / (v.max() + 1e-6)
            v_disp = np.power(v_norm, 1.0 / gamma)
            self.img.set_data(v_disp)
            view_field = v_disp
        elif self.view_mode == "mass":
            m = self.sim.state["M"]
            m_norm = m / (m.max() + 1e-6)
            m_disp = np.power(m_norm, 1.0 / gamma)
            self.img.set_data(m_disp)
            view_field = m_disp
        elif self.view_mode == "potential":
            self.img.set_data(self.sim.last_phi)
            view_field = np.abs(self.sim.last_phi)
        else:
            v = self.sim.state["V"]
            m = self.sim.state["M"]
            v_norm = v / (v.max() + 1e-6)
            m_norm = m / (m.max() + 1e-6)
            v_norm = np.power(v_norm, 1.0 / gamma)
            m_norm = np.power(m_norm, 1.0 / gamma)
            rgb = np.zeros((v.shape[0], v.shape[1], 3), dtype=np.float32)
            rgb[..., 0] = np.clip(m_norm * 1.2, 0.0, 1.0)
            rgb[..., 1] = np.clip(v_norm * 1.2, 0.0, 1.0)
            rgb[..., 2] = np.clip(v_norm + 0.3 * m_norm, 0.0, 1.0)
            self.img.set_data(rgb)
            view_field = (rgb[..., 0] + rgb[..., 1] + rgb[..., 2]) / 3.0

        if self.started:
            metrics = self.metrics.update(
                self.sim.state,
                self.sim.last_phi,
                self.sim.dt,
                self.frame,
                self.sim.params["bubble_thresh"],
                view_field if view_field is not None else self.sim.state["V"],
                self.sim.params.get("min_component_area", 30),
                dM_map=self.sim.last_dM_map,
            )
            self._update_monitors(metrics)
            self._log_frame_if_needed(metrics)
            if (self.frame % PLOT_UPDATE_EVERY) == 0:
                self._update_plots()
            if self.test_active:
                self._maybe_save_snapshot()
        else:
            self.monitor_text.set_text(
                "\n".join(
                    [
                        f"preset: {PRESETS[self.preset_index][0]}",
                        "paused: True",
                        "frame: 0",
                        f"dt: {self.sim.dt:.3f}",
                        f"view: {self.view_mode}",
                        "waiting for first poke...",
                    ]
                )
            )
            self.monitor_text_right.set_text(
                "\n".join(
                    [
                        f"click: {self.click_mode}",
                        "wall s: NA",
                        "sim s: NA",
                        "fps: NA  ema: NA",
                    ]
                )
            )
            self.line_r.set_data([], [])
            self.line_fft.set_data([], [])
            self.ax_r.set_xlim(0, 1)
            self.ax_r.set_ylim(0, 1)
            self.ax_fft.set_xlim(0, 1)
            self.ax_fft.set_ylim(0, 1)

        if end_test_run:
            self._advance_test_run()

        return (self.img,)

    def close(self):
        if self.run_logger is not None:
            self.run_logger.close()
        self.metrics.close_run_log()


def run(sim):
    ui = EmitUI(sim)
    plt.show()
    ui.close()
    if ui.run_logger is not None:
        print(
            "Run summary:",
            f"total_frames_logged={ui.run_logger.total_frames_logged}",
            f"total_clicks={ui.run_logger.total_clicks}",
            f"total_actions={ui.run_logger.total_actions}",
            f"output_dir={ui.run_logger.run_dir}",
        )
