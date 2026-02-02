import atexit
import csv
import os
import time
from collections import deque

import numpy as np
from scipy.ndimage import binary_erosion, label


def _grad_mag(phi):
    dphi_dx = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) * 0.5
    dphi_dy = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) * 0.5
    return np.sqrt(dphi_dx * dphi_dx + dphi_dy * dphi_dy, dtype=np.float32)


class RingBuffer:
    def __init__(self, size):
        self.size = int(size)
        self.data = np.zeros(self.size, dtype=np.float32)
        self.index = 0
        self.full = False

    def append(self, value):
        self.data[self.index] = value
        self.index = (self.index + 1) % self.size
        if self.index == 0:
            self.full = True

    def values(self):
        if self.full:
            return np.concatenate((self.data[self.index :], self.data[: self.index]))
        return self.data[: self.index]

    def __len__(self):
        return self.size if self.full else self.index


class MetricsTracker:
    def __init__(
        self,
        series_len=512,
        plot_update_every=10,
        fit_win=256,
        fit_update_every=10,
        ds_win=256,
        ds_min_valid=100,
        ds_tol=0.05,
        log_every=10,
        log_enabled=True,
        log_dir="runs",
    ):
        self.series_len = int(series_len)
        self.plot_update_every = int(plot_update_every)
        self.fit_win = int(fit_win)
        self.fit_update_every = int(fit_update_every)
        self.ds_win = int(ds_win)
        self.ds_min_valid = int(ds_min_valid)
        self.ds_tol = float(ds_tol)
        self.log_every = int(log_every)
        self.log_enabled = bool(log_enabled)
        self.log_dir = log_dir
        self.r_series = RingBuffer(self.series_len)
        self.t_series = RingBuffer(self.series_len)
        self.h_series = RingBuffer(self.series_len)
        self.rho_series = RingBuffer(self.series_len)
        self.h2_series = RingBuffer(self.series_len)
        self.last_fft = None
        self.last_freqs = None
        self.last_metrics = {}
        self._warning_counter = 0
        self._warning_every = 200
        self._log_file = None
        self._log_writer = None
        self.log_path = None
        self._last_wall = None
        self._wall_start = None
        self._fps_ema = None
        self._prev_m_total = None
        self._prev_v_total = None
        self._prev_r = None
        self._rate_buffer = deque(maxlen=200)
        self._coords_cache = None
        self._coords_shape = None
        self._invariant_samples = []
        self.invariant_fit_c = None
        atexit.register(self.close_run_log)

    def start_run_log(self, preset_name):
        if not self.log_enabled:
            return
        self.close_run_log()
        os.makedirs(self.log_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        safe_preset = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in preset_name)
        filename = f"emit_{stamp}_{safe_preset}.csv"
        self.log_path = os.path.join(self.log_dir, filename)
        self._log_file = open(self.log_path, "w", newline="", encoding="utf-8")
        self._log_writer = csv.writer(self._log_file)
        self._log_writer.writerow(
            [
                "t",
                "R",
                "H",
                "rho",
                "H2",
                "A",
                "B",
                "Lambda_eff",
                "Omega_m",
                "Omega_L",
                "fit_r2",
                "regime",
                "de_sitter_pass",
                "fps_ema",
                "dM_dt_wall",
            ]
        )
        self._log_file.flush()
        self._last_wall = None
        self._wall_start = None
        self._fps_ema = None
        self._prev_m_total = None
        self._prev_v_total = None
        self._prev_r = None
        self._rate_buffer.clear()
        self._invariant_samples = []
        self.invariant_fit_c = None

    def close_run_log(self):
        if self._log_file is not None:
            try:
                self._log_file.flush()
            finally:
                self._log_file.close()
        self._log_file = None
        self._log_writer = None

    def _box_blur(self, arr, radius):
        if radius <= 0:
            return arr
        pad = int(radius)
        size = 2 * pad + 1
        padded = np.pad(arr, pad_width=pad, mode="wrap")
        csum = padded.cumsum(axis=0).cumsum(axis=1)
        csum = np.pad(csum, pad_width=((1, 0), (1, 0)), mode="constant", constant_values=0.0)
        h, w = arr.shape
        total = (
            csum[size : size + h, size : size + w]
            - csum[0:h, size : size + w]
            - csum[size : size + h, 0:w]
            + csum[0:h, 0:w]
        )
        return total / float(size * size)

    def _anisotropy(self, weights, eps=1e-12):
        if self._coords_cache is None:
            return np.nan
        wsum = float(np.sum(weights))
        if wsum <= eps or wsum < 5:
            return np.nan
        xs, ys = self._coords_cache
        xbar = float(np.sum(weights * xs) / wsum)
        ybar = float(np.sum(weights * ys) / wsum)
        dx = xs - xbar
        dy = ys - ybar
        ixx = float(np.sum(weights * dx * dx) / wsum)
        iyy = float(np.sum(weights * dy * dy) / wsum)
        ixy = float(np.sum(weights * dx * dy) / wsum)
        trace = ixx + iyy
        if trace <= eps:
            return np.nan
        disc = max(trace * trace - 4.0 * (ixx * iyy - ixy * ixy), 0.0)
        l1 = 0.5 * (trace + np.sqrt(disc))
        l2 = 0.5 * (trace - np.sqrt(disc))
        return float((l1 - l2) / (l1 + l2 + eps))

    def update(self, state, phi, dt, frame, bubble_thresh, view_field, min_area, dM_map=None):
        v = np.nan_to_num(state["V"], nan=0.0, posinf=0.0, neginf=0.0)
        m = np.nan_to_num(state["M"], nan=0.0, posinf=0.0, neginf=0.0)
        t = np.nan_to_num(
            state.get("T", np.ones_like(m, dtype=np.float32)),
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        )

        v_min, v_max = float(v.min()), float(v.max())
        m_min, m_max = float(m.min()), float(m.max())
        v_mean, m_mean = float(v.mean()), float(m.mean())
        v_total, m_total = float(v.sum()), float(m.sum())
        p_min, p_max = float(phi.min()), float(phi.max())
        p_mean = float(phi.mean())
        t_min, t_mean, t_max = float(t.min()), float(t.mean()), float(t.max())

        mask = m > bubble_thresh
        if mask.sum() == 0:
            mask = v > bubble_thresh
        area = int(mask.sum())
        radius = float(np.sqrt(area / np.pi)) if area > 0 else 0.0
        radius_est = radius
        rho = float(m[mask].mean()) if area > 0 else np.nan

        time_value = frame * dt
        self.r_series.append(radius)
        self.t_series.append(time_value)
        self.rho_series.append(rho if np.isfinite(rho) else np.nan)

        do_expensive = (frame % self.plot_update_every) == 0
        do_fit = (frame % self.fit_update_every) == 0
        components_raw = self.last_metrics.get("components_raw", 0)
        components_filtered = self.last_metrics.get("components_filtered", 0)
        components_m_raw = self.last_metrics.get("components_m_raw", 0)
        components_m_filtered = self.last_metrics.get("components_m_filtered", 0)
        largest_m_area = self.last_metrics.get("largest_m_area", 0)
        second_largest_m_area = self.last_metrics.get("second_largest_m_area", 0)
        anisotropy_m = self.last_metrics.get("anisotropy_m", np.nan)
        anisotropy_m_s8 = self.last_metrics.get("anisotropy_m_s8", np.nan)
        anisotropy_m_s16 = self.last_metrics.get("anisotropy_m_s16", np.nan)
        pressure = self.last_metrics.get("pressure", 0.0)
        pressure_sum_grad_phi = self.last_metrics.get("pressure_sum_grad_phi", np.nan)
        collapse_sum_dM = self.last_metrics.get("collapse_sum_dM", np.nan)
        components_m_filtered_threshold_used = self.last_metrics.get(
            "components_m_filtered_threshold_used", np.nan
        )
        f_peak = self.last_metrics.get("f_peak", 0.0)
        r_det_last = self.last_metrics.get("r_det_last", 0.0)
        h_val = self.last_metrics.get("h", np.nan)
        h2_val = self.last_metrics.get("h2", np.nan)
        fit_a = self.last_metrics.get("fit_a", np.nan)
        fit_b = self.last_metrics.get("fit_b", np.nan)
        fit_r2 = self.last_metrics.get("fit_r2", np.nan)
        lambda_eff = self.last_metrics.get("lambda_eff", np.nan)
        omega_m = self.last_metrics.get("omega_m", np.nan)
        omega_l = self.last_metrics.get("omega_l", np.nan)
        omega_m_disp = self.last_metrics.get("omega_m_disp", np.nan)
        omega_l_disp = self.last_metrics.get("omega_l_disp", np.nan)
        regime = self.last_metrics.get("regime", "unstable/unknown")
        de_sitter = self.last_metrics.get("de_sitter", False)
        invariant_fit_c = self.last_metrics.get("invariant_fit_c", self.invariant_fit_c)
        i2_total = self.last_metrics.get("i2_total", np.nan)

        if do_expensive:
            view_mask = view_field > bubble_thresh
            labeled_view, count = label(view_mask)
            components_raw = int(count)
            if count > 0:
                sizes = np.bincount(labeled_view.ravel())
                components_filtered = int(np.sum(sizes[1:] >= max(1, int(min_area))))
            else:
                components_filtered = 0

        if do_expensive:
            m_mask = m > bubble_thresh
            labeled_m, count_m = label(m_mask)
            components_m_raw = int(count_m)
            if count_m > 0:
                sizes_m = np.bincount(labeled_m.ravel())
                valid_sizes = sizes_m[1:]
                min_area_val = max(1, int(min_area))
                components_m_filtered = int(np.sum(valid_sizes >= min_area_val))
                components_m_filtered_threshold_used = float(min_area_val)
                if valid_sizes.size > 0:
                    sorted_sizes = np.sort(valid_sizes)
                    largest_m_area = int(sorted_sizes[-1])
                    second_largest_m_area = int(sorted_sizes[-2]) if sorted_sizes.size > 1 else 0
                else:
                    largest_m_area = 0
                    second_largest_m_area = 0
            else:
                components_m_filtered = 0
                components_m_filtered_threshold_used = float(max(1, int(min_area)))
                largest_m_area = 0
                second_largest_m_area = 0

        if do_expensive:
            if self._coords_shape != m.shape:
                yy, xx = np.indices(m.shape, dtype=np.float32)
                self._coords_cache = (xx, yy)
                self._coords_shape = m.shape
            m_max = float(m.max())
            if m_max > 1e-9:
                m_thresh = 0.2 * m_max
                mask = m > m_thresh
                anisotropy_m = self._anisotropy(mask.astype(np.float32))
                m_s8 = self._box_blur(m, 8)
                m_s16 = self._box_blur(m, 16)
                m_s8_max = float(m_s8.max())
                m_s16_max = float(m_s16.max())
                if m_s8_max > 1e-9:
                    mask_s8 = m_s8 > (0.2 * m_s8_max)
                    anisotropy_m_s8 = self._anisotropy(mask_s8.astype(np.float32))
                else:
                    anisotropy_m_s8 = np.nan
                if m_s16_max > 1e-9:
                    mask_s16 = m_s16 > (0.2 * m_s16_max)
                    anisotropy_m_s16 = self._anisotropy(mask_s16.astype(np.float32))
                else:
                    anisotropy_m_s16 = np.nan
            else:
                anisotropy_m = np.nan
                anisotropy_m_s8 = np.nan
                anisotropy_m_s16 = np.nan

        if dM_map is not None:
            collapse_sum_dM = float(np.nan_to_num(dM_map, nan=0.0).sum())

        grad_mag = None
        if do_expensive and phi is not None:
            grad_mag = _grad_mag(phi)
            pressure_sum_grad_phi = float(grad_mag.sum())

        if do_expensive and area > 0:
            eroded = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
            boundary = mask & ~eroded
            if boundary.any():
                if grad_mag is None:
                    grad_mag = _grad_mag(phi)
                pressure = float(grad_mag[boundary].mean())

        if do_expensive and len(self.r_series) >= 16:
            series = self.r_series.values()
            n = len(series)
            window = max(5, n // 5)
            kernel = np.ones(window, dtype=np.float32) / float(window)
            trend = np.convolve(series, kernel, mode="same")
            r_det = series - trend
            r_det_last = float(r_det[-1])

            win = np.hanning(n).astype(np.float32)
            fft_vals = np.fft.rfft(r_det * win)
            freqs = np.fft.rfftfreq(n, d=dt)
            mag = np.abs(fft_vals)
            if mag.size > 1:
                peak_idx = 1 + int(np.argmax(mag[1:]))
                f_peak = float(freqs[peak_idx])
            else:
                f_peak = 0.0

            self.last_fft = mag
            self.last_freqs = freqs

        if len(self.r_series) >= 3:
            series = self.r_series.values()
            if len(series) >= 5:
                kernel = np.ones(5, dtype=np.float32) / 5.0
                smooth = np.convolve(series, kernel, mode="same")
            else:
                smooth = series
            dRdt = (smooth[-1] - smooth[-3]) / (2.0 * dt)
            if radius > 1e-6 and np.isfinite(dRdt):
                h_val = float(dRdt / max(radius, 1e-6))
                h2_val = float(h_val * h_val)
            else:
                h_val = np.nan
                h2_val = np.nan
        self.h_series.append(h_val if np.isfinite(h_val) else np.nan)
        self.h2_series.append(h2_val if np.isfinite(h2_val) else np.nan)

        if do_fit and len(self.h2_series) >= 8:
            rho_vals = self.rho_series.values()
            h2_vals = self.h2_series.values()
            n = min(len(rho_vals), len(h2_vals), self.fit_win)
            if n >= 8:
                rho_win = rho_vals[-n:]
                h2_win = h2_vals[-n:]
                valid = np.isfinite(rho_win) & np.isfinite(h2_win) & (h2_win > 0.0)
                if np.count_nonzero(valid) >= 5:
                    x = rho_win[valid]
                    y = h2_win[valid]
                    A = np.vstack([x, np.ones_like(x)]).T
                    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                    fit_a = float(coeffs[0])
                    fit_b = float(coeffs[1])
                    y_pred = fit_a * x + fit_b
                    ss_res = float(np.sum((y - y_pred) ** 2))
                    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                    fit_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
                else:
                    fit_a = np.nan
                    fit_b = np.nan
                    fit_r2 = np.nan

        if np.isfinite(fit_b):
            lambda_eff = float(3.0 * fit_b)
        else:
            lambda_eff = np.nan

        if np.isfinite(h2_val) and h2_val > 0 and np.isfinite(fit_a) and np.isfinite(fit_b):
            omega_m = float((fit_a * rho) / h2_val) if np.isfinite(rho) else np.nan
            omega_l = float(fit_b / h2_val)
            omega_m_disp = float(np.clip(omega_m, 0.0, 1.0))
            omega_l_disp = float(np.clip(omega_l, 0.0, 1.0))
        else:
            omega_m = np.nan
            omega_l = np.nan
            omega_m_disp = np.nan
            omega_l_disp = np.nan

        if not np.isfinite(fit_r2) or fit_r2 < 0.2:
            regime = "unstable/unknown"
        elif np.isfinite(omega_l) and omega_l > 0.7:
            regime = "lambda-like"
        elif np.isfinite(omega_m) and omega_m > 0.7:
            regime = "matter-like"
        else:
            regime = "mixed"

        h_vals = self.h_series.values()
        if len(h_vals) >= min(self.ds_win, len(h_vals)):
            h_win = h_vals[-min(self.ds_win, len(h_vals)) :]
            valid = np.isfinite(h_win)
            if np.count_nonzero(valid) >= self.ds_min_valid:
                mean_h = float(np.mean(h_win[valid]))
                std_h = float(np.std(h_win[valid]))
                if mean_h > 0 and std_h / abs(mean_h) < self.ds_tol:
                    de_sitter = True
                else:
                    de_sitter = False
            else:
                de_sitter = False
        else:
            de_sitter = False

        wall_now = time.perf_counter()
        if self._wall_start is None:
            self._wall_start = wall_now
        wall_elapsed = wall_now - self._wall_start
        prev_wall = self._last_wall
        if prev_wall is None:
            fps_inst = np.nan
            dt_wall = np.nan
        else:
            dt_wall = max(1e-6, wall_now - prev_wall)
            fps_inst = 1.0 / dt_wall
        self._last_wall = wall_now
        if self._fps_ema is None and np.isfinite(fps_inst):
            self._fps_ema = fps_inst
        elif np.isfinite(fps_inst):
            self._fps_ema = 0.95 * self._fps_ema + 0.05 * fps_inst

        sim_elapsed = frame * dt
        dM_dt_wall = np.nan
        dE_dt_wall = np.nan
        dR_dt_wall = np.nan
        dM_dt_sim = np.nan
        dE_dt_sim = np.nan
        dR_dt_sim = np.nan
        if self._prev_m_total is not None and np.isfinite(dt_wall):
            dM_dt_wall = (m_total - self._prev_m_total) / dt_wall
            dE_dt_wall = (v_total - self._prev_v_total) / dt_wall
            dR_dt_wall = (radius - self._prev_r) / dt_wall
            dM_dt_sim = (m_total - self._prev_m_total) / max(1e-6, dt)
            dE_dt_sim = (v_total - self._prev_v_total) / max(1e-6, dt)
            dR_dt_sim = (radius - self._prev_r) / max(1e-6, dt)
        self._prev_m_total = m_total
        self._prev_v_total = v_total
        self._prev_r = radius

        decay_event = int((m_total < 1e-6) or (components_m_raw == 0))

        if 50 <= frame <= 200:
            self._invariant_samples.append((v_total, m_total))
        if frame >= 200 and self.invariant_fit_c is None:
            if len(self._invariant_samples) >= 10:
                samples = np.array(self._invariant_samples, dtype=np.float64)
                v_vals = samples[:, 0]
                m_vals = samples[:, 1]
                v_mean = float(np.mean(v_vals))
                m_mean = float(np.mean(m_vals))
                cov_vm = float(np.mean((v_vals - v_mean) * (m_vals - m_mean)))
                var_m = float(np.mean((m_vals - m_mean) ** 2))
                if var_m > 1e-9:
                    self.invariant_fit_c = -cov_vm / var_m
                else:
                    self.invariant_fit_c = 0.0
            else:
                self.invariant_fit_c = 0.0
            invariant_fit_c = self.invariant_fit_c
        if self.invariant_fit_c is not None:
            i2_total = v_total + self.invariant_fit_c * m_total

        self._rate_buffer.append(
            (wall_now, dM_dt_wall, dE_dt_wall, dR_dt_wall)
        )
        if self._rate_buffer:
            valid = np.array([[x[1], x[2], x[3]] for x in self._rate_buffer], dtype=np.float32)
            dM_dt_wall_avg = float(np.nanmean(valid[:, 0]))
            dE_dt_wall_avg = float(np.nanmean(valid[:, 1]))
            dR_dt_wall_avg = float(np.nanmean(valid[:, 2]))
        else:
            dM_dt_wall_avg = np.nan
            dE_dt_wall_avg = np.nan
            dR_dt_wall_avg = np.nan

        self.last_metrics = {
            "v_min": v_min,
            "v_max": v_max,
            "v_mean": v_mean,
            "v_total": v_total,
            "m_min": m_min,
            "m_max": m_max,
            "m_mean": m_mean,
            "m_total": m_total,
            "p_min": p_min,
            "p_max": p_max,
            "p_mean": p_mean,
            "t_min": t_min,
            "t_mean": t_mean,
            "t_max": t_max,
            "area": area,
            "radius": radius,
            "radius_est": radius_est,
            "rho": rho,
            "h": h_val,
            "h2": h2_val,
            "r_det_last": r_det_last,
            "components_raw": components_raw,
            "components_filtered": components_filtered,
            "components_m_raw": components_m_raw,
            "components_m_filtered": components_m_filtered,
            "largest_m_area": largest_m_area,
            "second_largest_m_area": second_largest_m_area,
            "components_m_filtered_threshold_used": components_m_filtered_threshold_used,
            "anisotropy_m": anisotropy_m,
            "anisotropy_m_s8": anisotropy_m_s8,
            "anisotropy_m_s16": anisotropy_m_s16,
            "pressure": pressure,
            "pressure_sum_grad_phi": pressure_sum_grad_phi,
            "collapse_sum_dM": collapse_sum_dM,
            "f_peak": f_peak,
            "fit_a": fit_a,
            "fit_b": fit_b,
            "fit_r2": fit_r2,
            "lambda_eff": lambda_eff,
            "omega_m": omega_m,
            "omega_l": omega_l,
            "omega_m_disp": omega_m_disp,
            "omega_l_disp": omega_l_disp,
            "regime": regime,
            "de_sitter": de_sitter,
            "log_path": self.log_path,
            "wall_elapsed": wall_elapsed,
            "sim_elapsed": sim_elapsed,
            "fps_inst": fps_inst,
            "fps_ema": self._fps_ema if self._fps_ema is not None else np.nan,
            "dM_dt_wall": dM_dt_wall,
            "dE_dt_wall": dE_dt_wall,
            "dR_dt_wall": dR_dt_wall,
            "dM_dt_sim": dM_dt_sim,
            "dE_dt_sim": dE_dt_sim,
            "dR_dt_sim": dR_dt_sim,
            "dM_dt_wall_avg": dM_dt_wall_avg,
            "dE_dt_wall_avg": dE_dt_wall_avg,
            "dR_dt_wall_avg": dR_dt_wall_avg,
            "decay_event": decay_event,
            "invariant_fit_c": invariant_fit_c,
            "i2_total": i2_total,
        }

        if (
            self.log_enabled
            and self._log_writer is not None
            and (frame % self.log_every) == 0
        ):
            self._log_writer.writerow(
                [
                    time_value,
                    radius,
                    h_val,
                    rho,
                    h2_val,
                    fit_a,
                    fit_b,
                    lambda_eff,
                    omega_m,
                    omega_l,
                    fit_r2,
                    regime,
                    de_sitter,
                    self._fps_ema if self._fps_ema is not None else np.nan,
                    dM_dt_wall,
                ]
            )
            self._log_file.flush()

        if do_expensive and components_m_raw > 0 and largest_m_area < max(1, int(min_area)):
            self._warning_counter += 1
            if (self._warning_counter % self._warning_every) == 0:
                print(
                    "Warning: largest M component area below min_area for extended period.",
                    f"largest_M_area={largest_m_area}",
                    f"min_area={min_area}",
                )

        return self.last_metrics
