import atexit
import csv
import json
import os

import numpy as np


def _sanitize_name(name):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name))


class RunLogger:
    def __init__(
        self,
        output_dir,
        run_id,
        preset_name,
        log_every=10,
        startup_frames=100,
        param_keys=None,
        run_dir=None,
    ):
        self.output_dir = output_dir
        self.run_id = str(run_id)
        self.preset_name = str(preset_name)
        self.log_every = int(log_every)
        self.startup_frames = int(startup_frames)
        self.param_keys = list(param_keys) if param_keys is not None else None

        if run_dir:
            self.run_dir = run_dir
        else:
            safe_preset = _sanitize_name(self.preset_name)
            self.run_dir = os.path.join(self.output_dir, f"{self.run_id}_{safe_preset}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.frames_path = os.path.join(self.run_dir, "frames.csv")
        self.events_path = os.path.join(self.run_dir, "events.csv")
        self.config_path = os.path.join(self.run_dir, "config.json")

        self.total_clicks = 0
        self.total_actions = 0
        self.actions_this_run = 0
        self.total_frames_logged = 0

        self._frames_file = None
        self._frames_writer = None
        self._frames_header = None
        self._events_file = None
        self._events_writer = None
        self._last_logged_frame = None
        self._last_components_m_filtered = None

        self._metric_columns = [
            "v_min",
            "v_max",
            "v_mean",
            "V_total",
            "m_min",
            "m_max",
            "m_mean",
            "M_total",
            "radius_est",
            "I2_total",
            "anisotropy_M",
            "anisotropy_M_smoothed_s8",
            "anisotropy_M_smoothed_s16",
            "p_min",
            "p_max",
            "p_mean",
            "area",
            "components_M_raw",
            "components_M_filtered",
            "components_M_filtered_threshold_used",
            "largest_M_area",
            "second_largest_M_area",
            "merge_event",
            "decay_event",
            "collapse_sum_dM",
            "pressure_sum_grad_phi",
            "mean_T",
            "min_T",
            "max_T",
            "H",
            "R",
            "rho",
            "H2",
            "fit_A",
            "fit_B",
            "Lambda_eff",
            "Omega_m",
            "Omega_L",
            "Omega_m_disp",
            "Omega_L_disp",
            "fit_r2",
            "regime",
            "de_sitter",
            "log_path",
            "wall_elapsed",
            "sim_elapsed",
            "fps_inst",
            "components_raw",
            "components_filtered",
            "pressure",
            "f_peak",
            "r_det_last",
            "fps_ema",
            "dM_dt_wall",
            "dE_dt_wall",
            "dR_dt_wall",
            "dM_dt_sim",
            "dE_dt_sim",
            "dR_dt_sim",
            "dM_dt_wall_avg",
            "dE_dt_wall_avg",
            "dR_dt_wall_avg",
        ]

        self._metric_map = {
            "v_min": "v_min",
            "v_max": "v_max",
            "v_mean": "v_mean",
            "V_total": "v_total",
            "m_min": "m_min",
            "m_max": "m_max",
            "m_mean": "m_mean",
            "M_total": "m_total",
            "radius_est": "radius_est",
            "I2_total": "i2_total",
            "anisotropy_M": "anisotropy_m",
            "anisotropy_M_smoothed_s8": "anisotropy_m_s8",
            "anisotropy_M_smoothed_s16": "anisotropy_m_s16",
            "p_min": "p_min",
            "p_max": "p_max",
            "p_mean": "p_mean",
            "area": "area",
            "components_M_raw": "components_m_raw",
            "components_M_filtered": "components_m_filtered",
            "components_M_filtered_threshold_used": "components_m_filtered_threshold_used",
            "largest_M_area": "largest_m_area",
            "second_largest_M_area": "second_largest_m_area",
            "collapse_sum_dM": "collapse_sum_dM",
            "pressure_sum_grad_phi": "pressure_sum_grad_phi",
            "mean_T": "t_mean",
            "min_T": "t_min",
            "max_T": "t_max",
            "H": "h",
            "R": "radius",
            "rho": "rho",
            "H2": "h2",
            "fit_A": "fit_a",
            "fit_B": "fit_b",
            "Lambda_eff": "lambda_eff",
            "Omega_m": "omega_m",
            "Omega_L": "omega_l",
            "Omega_m_disp": "omega_m_disp",
            "Omega_L_disp": "omega_l_disp",
            "fit_r2": "fit_r2",
            "regime": "regime",
            "de_sitter": "de_sitter",
            "log_path": "log_path",
            "wall_elapsed": "wall_elapsed",
            "sim_elapsed": "sim_elapsed",
            "fps_inst": "fps_inst",
            "components_raw": "components_raw",
            "components_filtered": "components_filtered",
            "pressure": "pressure",
            "f_peak": "f_peak",
            "r_det_last": "r_det_last",
            "fps_ema": "fps_ema",
            "dM_dt_wall": "dM_dt_wall",
            "dE_dt_wall": "dE_dt_wall",
            "dR_dt_wall": "dR_dt_wall",
            "dM_dt_sim": "dM_dt_sim",
            "dE_dt_sim": "dE_dt_sim",
            "dR_dt_sim": "dR_dt_sim",
            "dM_dt_wall_avg": "dM_dt_wall_avg",
            "dE_dt_wall_avg": "dE_dt_wall_avg",
            "dR_dt_wall_avg": "dR_dt_wall_avg",
            "decay_event": "decay_event",
        }

        atexit.register(self.close)

    def write_config(self, config):
        try:
            with open(self.config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2)
        except OSError:
            pass

    def should_log_frame(self, frame_idx):
        if frame_idx == 0:
            return True
        if frame_idx < self.startup_frames:
            return True
        return (frame_idx % self.log_every) == 0

    def _ensure_frames_writer(self, params, extras):
        if self._frames_writer is not None:
            return
        if self.param_keys is None:
            self.param_keys = list(params.keys()) if params else []
        extra_keys = list(extras.keys()) if extras else []
        self._extra_keys = extra_keys
        header = (
            [
                "frame",
                "t",
                "preset_name",
                "run_id",
                "total_clicks",
                "total_actions",
                "actions_this_run",
            ]
            + self.param_keys
            + self._metric_columns
            + self._extra_keys
        )
        self._frames_file = open(self.frames_path, "w", newline="", encoding="utf-8")
        self._frames_writer = csv.writer(self._frames_file)
        self._frames_writer.writerow(header)
        self._frames_file.flush()
        self._frames_header = header

    def _ensure_events_writer(self):
        if self._events_writer is not None:
            return
        self._events_file = open(self.events_path, "w", newline="", encoding="utf-8")
        self._events_writer = csv.writer(self._events_file)
        self._events_writer.writerow(
            ["frame", "t", "preset_name", "run_id", "event_type", "payload"]
        )
        self._events_file.flush()

    def _format_value(self, value):
        if value is None:
            return np.nan
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return np.nan
            return float(value)
        if isinstance(value, (int, np.integer)):
            return int(value)
        return value

    def log_frame(self, frame_idx, t, params, metrics, extras=None):
        if not self.should_log_frame(frame_idx):
            return False
        if self._last_logged_frame == frame_idx:
            return False

        params = params or {}
        metrics = metrics or {}
        extras = extras or {}
        self._ensure_frames_writer(params, extras)

        components_filtered = metrics.get("components_m_filtered")
        merge_event = np.nan
        if components_filtered is not None and np.isfinite(components_filtered):
            if (
                self._last_components_m_filtered is not None
                and np.isfinite(self._last_components_m_filtered)
                and components_filtered < self._last_components_m_filtered
            ):
                merge_event = 1
            else:
                merge_event = 0
            self._last_components_m_filtered = components_filtered

        decay_event = metrics.get("decay_event", np.nan)

        row = [
            frame_idx,
            t,
            self.preset_name,
            self.run_id,
            self.total_clicks,
            self.total_actions,
            self.actions_this_run,
        ]
        for key in self.param_keys:
            row.append(self._format_value(params.get(key)))
        for col in self._metric_columns:
            if col == "merge_event":
                row.append(self._format_value(merge_event))
            elif col == "decay_event":
                row.append(self._format_value(decay_event))
            else:
                metric_key = self._metric_map.get(col)
                row.append(self._format_value(metrics.get(metric_key)))
        for key in self._extra_keys:
            row.append(self._format_value(extras.get(key)))

        self._frames_writer.writerow(row)
        self._frames_file.flush()
        self._last_logged_frame = frame_idx
        self.total_frames_logged += 1
        return True

    def log_event(self, frame_idx, t, event_type, payload):
        if payload is None:
            payload = {}
        if "affected_field" not in payload:
            payload["affected_field"] = None
        if "delta_field_total" not in payload:
            payload["delta_field_total"] = np.nan
        if "delta_field_l2" not in payload:
            payload["delta_field_l2"] = np.nan
        self._ensure_events_writer()
        if event_type == "click":
            self.total_clicks += 1
        if event_type == "action":
            self.total_actions += 1
            self.actions_this_run += 1
        try:
            payload_str = json.dumps(payload, separators=(",", ":"), default=str)
        except (TypeError, ValueError):
            payload_str = "{}"
        self._events_writer.writerow(
            [frame_idx, t, self.preset_name, self.run_id, event_type, payload_str]
        )
        self._events_file.flush()

    def close(self):
        if self._frames_file is not None:
            try:
                self._frames_file.flush()
            finally:
                self._frames_file.close()
        if self._events_file is not None:
            try:
                self._events_file.flush()
            finally:
                self._events_file.close()
        self._frames_file = None
        self._frames_writer = None
        self._events_file = None
        self._events_writer = None
