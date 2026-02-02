import argparse
import csv
import json
import os
import time

import numpy as np
import matplotlib.pyplot as plt


COMPACT_COLUMNS = [
    "run_id",
    "t",
    "R",
    "area",
    "radius_est",
    "components_filtered",
    "V_total",
    "M_total",
    "I2_total",
    "anisotropy_M",
    "anisotropy_M_smoothed_s8",
    "anisotropy_M_smoothed_s16",
]


def _load_frames_csv(path):
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return rows


def _load_events_csv(path):
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return rows


def _find_run_dirs(runs_root):
    run_dirs = []
    for root, _dirs, files in os.walk(runs_root):
        if "config.json" in files and "frames.csv" in files and "events.csv" in files:
            run_dirs.append(root)
    return sorted(run_dirs)


def _safe_float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _linear_slope(x, y):
    if len(x) < 5:
        return np.nan
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 5:
        return np.nan
    x = x[mask]
    y = y[mask]
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    denom = np.sum((x - x_mean) ** 2)
    if denom <= 0:
        return np.nan
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _plot_test1_stability(out_path, rows):
    dt_vals = []
    area_frac = []
    for row in rows:
        if row.get("test_name") != "Test_1_dt_sweep":
            continue
        if row.get("grid_size") != 256:
            continue
        dt_vals.append(row.get("dt", np.nan))
        area_frac.append(row.get("area_frac_end", np.nan))
    if not dt_vals:
        return
    dt_vals = np.asarray(dt_vals, dtype=np.float64)
    area_frac = np.asarray(area_frac, dtype=np.float64)
    order = np.argsort(dt_vals)
    plt.figure(figsize=(5.5, 3.5))
    plt.plot(dt_vals[order], area_frac[order], marker="o")
    plt.xlabel("dt")
    plt.ylabel("area / N^2")
    plt.title("Test 1 stability boundary (N=256)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_scaling(out_path, run_items, mode="R"):
    plt.figure(figsize=(6, 3.5))
    for item in run_items:
        t = item["t"]
        if len(t) == 0:
            continue
        t_end = np.nanmax(t)
        if not np.isfinite(t_end) or t_end <= 0:
            continue
        t_norm = t / t_end
        if mode == "R":
            series = item["R"] / float(item["grid_size"])
            ylabel = "R / N"
        else:
            series = item["area"] / float(item["grid_size"] ** 2)
            ylabel = "area / N^2"
        label = f"N={item['grid_size']} dt={item['dt']}"
        plt.plot(t_norm, series, label=label, linewidth=1)
    plt.xlabel("t / T_end")
    plt.ylabel(ylabel)
    plt.title(f"Test 5 scaling: {ylabel}")
    plt.legend(loc="best", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_anisotropy(out_path, run_items, grid_size):
    plt.figure(figsize=(6, 3.5))
    for item in run_items:
        if item["grid_size"] != grid_size:
            continue
        t = item["t"]
        label = f"dt={item['dt']}"
        plt.plot(t, item["anisotropy_M"], label=f"raw {label}", linewidth=1)
        plt.plot(t, item["anisotropy_M_smoothed_s8"], label=f"s8 {label}", linewidth=1)
        plt.plot(t, item["anisotropy_M_smoothed_s16"], label=f"s16 {label}", linewidth=1)
    plt.xlabel("time")
    plt.ylabel("anisotropy")
    plt.title(f"Anisotropy vs time (N={grid_size})")
    plt.legend(loc="best", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_invariant(out_path, item, slope):
    t = item["t"]
    i2 = item["I2_total"]
    plt.figure(figsize=(6, 3.5))
    plt.plot(t, i2, label="I2_total", linewidth=1)
    if np.isfinite(slope):
        plt.title(f"I2 drift slope={slope:.3e}")
    else:
        plt.title("I2 drift")
    plt.xlabel("time")
    plt.ylabel("I2_total")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_overlay(out_path, items, y_key, y_label, title):
    plt.figure(figsize=(6, 3.5))
    for item in items:
        t = item["t"]
        if len(t) == 0:
            continue
        if "seed" in item:
            label = f"seed={item['seed']}"
        elif "grid_size" in item:
            label = f"N={item['grid_size']} dt={item.get('dt')}"
        elif "dt" in item:
            label = f"dt={item['dt']}"
        else:
            label = "series"
        plt.plot(t, item[y_key], label=label, linewidth=1)
    plt.xlabel("time")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc="best", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _rmse_aligned(t_a, y_a, t_b, y_b):
    if len(t_a) < 5 or len(t_b) < 5:
        return np.nan
    t_min = max(np.nanmin(t_a), np.nanmin(t_b))
    t_max = min(np.nanmax(t_a), np.nanmax(t_b))
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        return np.nan
    t_common = np.linspace(t_min, t_max, num=200)
    y_a_interp = np.interp(t_common, t_a, y_a)
    y_b_interp = np.interp(t_common, t_b, y_b)
    diff = y_a_interp - y_b_interp
    return float(np.sqrt(np.mean(diff * diff)))


def main():
    parser = argparse.ArgumentParser(description="Analyze EMIT test runs")
    parser.add_argument("--runs_dir", default="runs", help="Root runs directory")
    parser.add_argument("--out_dir", default="analysis_outputs", help="Output directory")
    parser.add_argument("--only_test", default=None, help="Filter by test number, e.g. 5")
    args = parser.parse_args()

    run_dirs = _find_run_dirs(args.runs_dir)
    if not run_dirs:
        print(f"No runs found under: {args.runs_dir}")
        return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(args.out_dir, stamp)
    _ensure_dir(out_root)
    plots_dir = os.path.join(out_root, "plots")
    _ensure_dir(plots_dir)
    _ensure_dir(os.path.join(plots_dir, "test1"))
    _ensure_dir(os.path.join(plots_dir, "scaling"))
    _ensure_dir(os.path.join(plots_dir, "anisotropy"))
    _ensure_dir(os.path.join(plots_dir, "invariants"))
    _ensure_dir(os.path.join(plots_dir, "continuum"))
    _ensure_dir(os.path.join(plots_dir, "continuum6b"))
    _ensure_dir(os.path.join(plots_dir, "test7"))
    _ensure_dir(os.path.join(plots_dir, "test8"))
    _ensure_dir(os.path.join(plots_dir, "test9"))
    _ensure_dir(os.path.join(plots_dir, "test10"))
    _ensure_dir(os.path.join(plots_dir, "test11"))
    _ensure_dir(os.path.join(plots_dir, "test12"))

    runs_summary = []
    compact_rows = []
    event_rate_rows = []
    test5_items = []
    test6_items = []
    test6b_items = []
    rmse_rows = []
    rmse_rows_6b = []
    test7_items = []
    test8_items = []
    test9_runs = []
    dt_rmse_rows = []
    test10_items = []
    test11_items = []
    test12_items = []
    test11_rmse_rows = []
    test12_rmse_rows = []

    for run_dir in run_dirs:
        config_path = os.path.join(run_dir, "config.json")
        frames_path = os.path.join(run_dir, "frames.csv")
        events_path = os.path.join(run_dir, "events.csv")
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)

        test_name = config.get("test_name")
        if args.only_test:
            prefix = f"Test_{args.only_test}_"
            if not (test_name and test_name.startswith(prefix)):
                continue

        frames = _load_frames_csv(frames_path)
        events = _load_events_csv(events_path)

        n = _safe_int(config.get("grid_size"))
        dt = _safe_float(config.get("dt"))
        seed = _safe_int(config.get("seed"))
        preset = config.get("preset_name")
        run_id = config.get("run_id")
        invariant_fit_c = _safe_float(config.get("invariant_fit_c"), 0.0)

        t_vals = np.array([_safe_float(r.get("t")) for r in frames], dtype=np.float64)
        r_vals = np.array([_safe_float(r.get("R")) for r in frames], dtype=np.float64)
        area_vals = np.array([_safe_float(r.get("area")) for r in frames], dtype=np.float64)
        radius_est_vals = np.array([_safe_float(r.get("radius_est")) for r in frames], dtype=np.float64)
        comp_vals = np.array([_safe_float(r.get("components_filtered")) for r in frames], dtype=np.float64)
        anis = np.array([_safe_float(r.get("anisotropy_M")) for r in frames], dtype=np.float64)
        anis_s8 = np.array([_safe_float(r.get("anisotropy_M_smoothed_s8")) for r in frames], dtype=np.float64)
        anis_s16 = np.array([_safe_float(r.get("anisotropy_M_smoothed_s16")) for r in frames], dtype=np.float64)
        i2 = np.array([_safe_float(r.get("I2_total")) for r in frames], dtype=np.float64)

        t_end = float(np.nanmax(t_vals)) if len(t_vals) else np.nan
        r_end = float(r_vals[-1]) if len(r_vals) else np.nan
        area_end = float(area_vals[-1]) if len(area_vals) else np.nan
        area_frac_end = area_end / float(n * n) if n else np.nan
        components_end = float(comp_vals[-1]) if len(comp_vals) else np.nan
        anis_end = float(anis[-1]) if len(anis) else np.nan
        anis_s8_end = float(anis_s8[-1]) if len(anis_s8) else np.nan
        anis_s16_end = float(anis_s16[-1]) if len(anis_s16) else np.nan

        drift_slope = np.nan
        if len(t_vals) > 10:
            start = int(len(t_vals) * 0.75)
            drift_slope = _linear_slope(t_vals[start:], i2[start:])

        runs_summary.append(
            {
                "test_name": test_name,
                "run_id": run_id,
                "grid_size": n,
                "dt": dt,
                "seed": seed,
                "preset": preset,
                "frames_end": len(frames),
                "t_end": t_end,
                "R_end": r_end,
                "area_end": area_end,
                "area_frac_end": area_frac_end,
                "components_end": components_end,
                "anisotropy_end": anis_end,
                "anisotropy_s8_end": anis_s8_end,
                "anisotropy_s16_end": anis_s16_end,
                "invariant_fit_c": invariant_fit_c,
                "I2_drift_slope": drift_slope,
            }
        )

        for row in frames:
            compact_rows.append(
                {
                    "run_id": run_id,
                    "t": row.get("t"),
                    "R": row.get("R"),
                    "area": row.get("area"),
                    "radius_est": row.get("radius_est"),
                    "components_filtered": row.get("components_filtered"),
                    "V_total": row.get("V_total"),
                    "M_total": row.get("M_total"),
                    "I2_total": row.get("I2_total"),
                    "anisotropy_M": row.get("anisotropy_M"),
                    "anisotropy_M_smoothed_s8": row.get("anisotropy_M_smoothed_s8"),
                    "anisotropy_M_smoothed_s16": row.get("anisotropy_M_smoothed_s16"),
                }
            )

        event_count = len(events)
        actions = len([e for e in events if e.get("event_type") == "action"])
        clicks = len([e for e in events if e.get("event_type") == "click"])
        rate = event_count / t_end if t_end and t_end > 0 else np.nan
        event_rate_rows.append(
            {
                "run_id": run_id,
                "test_name": test_name,
                "grid_size": n,
                "dt": dt,
                "events": event_count,
                "actions": actions,
                "clicks": clicks,
                "events_per_sim_second": rate,
            }
        )

        if test_name == "Test_5_scaling":
            test5_items.append(
                {
                    "grid_size": n,
                    "dt": dt,
                    "t": t_vals,
                    "R": r_vals,
                    "area": area_vals,
                    "I2_total": i2,
                    "anisotropy_M": anis,
                    "anisotropy_M_smoothed_s8": anis_s8,
                    "anisotropy_M_smoothed_s16": anis_s16,
                }
            )

        if test_name == "Test_5_scaling":
            inv_path = os.path.join(plots_dir, "invariants", f"invariant_N{n}_dt{dt}.png")
            _plot_invariant(inv_path, {"t": t_vals, "I2_total": i2}, drift_slope)

        if test_name == "Test_6_continuum_dx_rescaled":
            if np.all(~np.isfinite(radius_est_vals)):
                radius_est_vals = np.sqrt(np.maximum(area_vals, 0.0) / np.pi)
            test6_items.append(
                {
                    "grid_size": n,
                    "dt": dt,
                    "t": t_vals,
                    "R": r_vals,
                    "area": area_vals,
                    "radius_est": radius_est_vals,
                    "I2_total": i2,
                    "anisotropy_M_smoothed_s16": anis_s16,
                }
            )

        if test_name == "Test_6B_CFL_dt_dx2":
            if np.all(~np.isfinite(radius_est_vals)):
                radius_est_vals = np.sqrt(np.maximum(area_vals, 0.0) / np.pi)
            test6b_items.append(
                {
                    "grid_size": n,
                    "dt": dt,
                    "t": t_vals,
                    "R": r_vals,
                    "area": area_vals,
                    "radius_est": radius_est_vals,
                    "I2_total": i2,
                    "anisotropy_M_smoothed_s16": anis_s16,
                }
            )

        if test_name == "Test_7_dt_invariance":
            if np.all(~np.isfinite(radius_est_vals)):
                radius_est_vals = np.sqrt(np.maximum(area_vals, 0.0) / np.pi)
            test7_items.append(
                {
                    "dt": dt,
                    "t": t_vals,
                    "radius_phys": radius_est_vals * float(config.get("dx", 1.0)),
                    "i2_phys": i2 * float(config.get("dx", 1.0)) ** 2,
                    "anisotropy_raw": anis,
                }
            )

        if test_name == "Test_8_prediction_plateau":
            if np.all(~np.isfinite(radius_est_vals)):
                radius_est_vals = np.sqrt(np.maximum(area_vals, 0.0) / np.pi)
            test8_items.append(
                {
                    "seed": seed,
                    "t": t_vals,
                    "i2_phys": i2 * float(config.get("dx", 1.0)) ** 2,
                    "anisotropy_raw": anis,
                }
            )

        if test_name == "Test_9_effective_PDE_fit":
            test9_runs.append(
                {
                    "run_dir": run_dir,
                    "dx": float(config.get("dx", 1.0)),
                    "dt": dt,
                }
            )
        if test_name == "Test_10_stability_N512":
            if np.all(~np.isfinite(radius_est_vals)):
                radius_est_vals = np.sqrt(np.maximum(area_vals, 0.0) / np.pi)
            test10_items.append(
                {
                    "t": t_vals,
                    "radius_phys": radius_est_vals * float(config.get("dx", 1.0)),
                    "i2_phys": i2 * float(config.get("dx", 1.0)) ** 2,
                    "anisotropy_s16": anis_s16,
                    "V_total": np.array([_safe_float(r.get("V_total")) for r in frames], dtype=np.float64),
                    "M_total": np.array([_safe_float(r.get("M_total")) for r in frames], dtype=np.float64),
                    "I2_total": i2,
                    "radius_est": radius_est_vals,
                    "dt": dt,
                    "dx": float(config.get("dx", 1.0)),
                    "physical_T": float(config.get("physical_T", np.nan)),
                    "frames": len(frames),
                }
            )
        if test_name == "Test_11_continuum_dt_invariance_N256":
            if np.all(~np.isfinite(radius_est_vals)):
                radius_est_vals = np.sqrt(np.maximum(area_vals, 0.0) / np.pi)
            test11_items.append(
                {
                    "dt": dt,
                    "t": t_vals,
                    "radius_phys": radius_est_vals * float(config.get("dx", 1.0)),
                    "i2_phys": i2 * float(config.get("dx", 1.0)) ** 2,
                    "anisotropy_s16": anis_s16,
                    "I2_total": i2,
                }
            )
        if test_name == "Test_12_refinement_trend_CFL":
            if np.all(~np.isfinite(radius_est_vals)):
                radius_est_vals = np.sqrt(np.maximum(area_vals, 0.0) / np.pi)
            test12_items.append(
                {
                    "grid_size": n,
                    "t": t_vals,
                    "radius_phys": radius_est_vals * float(config.get("dx", 1.0)),
                    "i2_phys": i2 * float(config.get("dx", 1.0)) ** 2,
                    "anisotropy_s16": anis_s16,
                    "I2_total": i2,
                }
            )

    runs_summary_path = os.path.join(out_root, "runs_summary.csv")
    with open(runs_summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs_summary[0].keys()))
        writer.writeheader()
        writer.writerows(runs_summary)

    frames_compact_path = os.path.join(out_root, "frames_compact.csv")
    with open(frames_compact_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPACT_COLUMNS)
        writer.writeheader()
        writer.writerows(compact_rows)

    event_rate_path = os.path.join(out_root, "event_rate_summary.csv")
    with open(event_rate_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(event_rate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(event_rate_rows)

    if rmse_rows:
        rmse_path = os.path.join(out_root, "continuum_scaling_rmse.csv")
        with open(rmse_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rmse_rows[0].keys()))
            writer.writeheader()
            writer.writerows(rmse_rows)

    if rmse_rows_6b:
        rmse_path = os.path.join(out_root, "continuum_scaling_rmse_test6B.csv")
        with open(rmse_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rmse_rows_6b[0].keys()))
            writer.writeheader()
            writer.writerows(rmse_rows_6b)

    if test7_items:
        _plot_overlay(
            os.path.join(plots_dir, "test7", "R_phys_vs_t.png"),
            [
                {**item, "series": item["radius_phys"]}
                for item in test7_items
            ],
            "series",
            "R_phys",
            "Test 7 dt invariance: R_phys vs t",
        )
        _plot_overlay(
            os.path.join(plots_dir, "test7", "I2_phys_vs_t.png"),
            [
                {**item, "series": item["i2_phys"]}
                for item in test7_items
            ],
            "series",
            "I2_phys",
            "Test 7 dt invariance: I2_phys vs t",
        )
        _plot_overlay(
            os.path.join(plots_dir, "test7", "anisotropy_raw_vs_t.png"),
            [
                {**item, "series": item["anisotropy_raw"]}
                for item in test7_items
            ],
            "series",
            "anisotropy",
            "Test 7 dt invariance: anisotropy vs t",
        )
        for i in range(len(test7_items)):
            for j in range(i + 1, len(test7_items)):
                a = test7_items[i]
                b = test7_items[j]
                dt_rmse_rows.append(
                    {
                        "dt_a": a["dt"],
                        "dt_b": b["dt"],
                        "rmse_R_phys": _rmse_aligned(a["t"], a["radius_phys"], b["t"], b["radius_phys"]),
                        "rmse_I2_phys": _rmse_aligned(a["t"], a["i2_phys"], b["t"], b["i2_phys"]),
                        "rmse_anisotropy": _rmse_aligned(a["t"], a["anisotropy_raw"], b["t"], b["anisotropy_raw"]),
                    }
                )
        if dt_rmse_rows:
            out_path = os.path.join(out_root, "dt_invariance_rmse_test7.csv")
            with open(out_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(dt_rmse_rows[0].keys()))
                writer.writeheader()
                writer.writerows(dt_rmse_rows)

    if test8_items:
        _plot_overlay(
            os.path.join(plots_dir, "test8", "I2_phys_by_seed.png"),
            [
                {**item, "series": item["i2_phys"]}
                for item in test8_items
            ],
            "series",
            "I2_phys",
            "Test 8 prediction: I2_phys by seed",
        )
        summary_rows = []
        cv_vals = []
        pass_count = 0
        for item in test8_items:
            t = item["t"]
            i2p = item["i2_phys"]
            mask = (t >= 600.0) & (t <= 1000.0)
            if np.count_nonzero(mask) < 5:
                mean_val = np.nan
                std_val = np.nan
                cv_val = np.nan
                slope = np.nan
            else:
                mean_val = float(np.nanmean(i2p[mask]))
                std_val = float(np.nanstd(i2p[mask]))
                cv_val = std_val / mean_val if mean_val != 0 else np.nan
                slope = _linear_slope(t[mask], i2p[mask])
            window_dur = 400.0
            slope_thresh = 1e-3 * mean_val / window_dur if np.isfinite(mean_val) else np.nan
            passed = bool(np.isfinite(cv_val) and np.isfinite(slope) and cv_val <= 0.10 and abs(slope) <= slope_thresh)
            if passed:
                pass_count += 1
            cv_vals.append(cv_val)
            summary_rows.append(
                {
                    "seed": item["seed"],
                    "mean_I2_phys": mean_val,
                    "std_I2_phys": std_val,
                    "cv_I2_phys": cv_val,
                    "slope_I2_phys": slope,
                    "passed": passed,
                }
            )
        if summary_rows:
            out_path = os.path.join(out_root, "prediction_test8_summary.csv")
            with open(out_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)
            passrate_path = os.path.join(out_root, "prediction_test8_passrate.json")
            with open(passrate_path, "w", encoding="utf-8") as handle:
                json.dump({"passed": pass_count, "total": len(summary_rows)}, handle, indent=2)
        if cv_vals:
            plt.figure(figsize=(5.0, 3.5))
            plt.hist([v for v in cv_vals if np.isfinite(v)], bins=8, color="tab:blue", alpha=0.8)
            plt.xlabel("cv_I2_phys")
            plt.ylabel("count")
            plt.title("Test 8: CV histogram")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "test8", "cv_hist.png"), dpi=160)
            plt.close()

    if test9_runs:
        pde_rows = []
        scatter_done = False
        for run in test9_runs:
            npz_dir = os.path.join(run["run_dir"], "snapshots_npz")
            if not os.path.isdir(npz_dir):
                continue
            files = sorted([f for f in os.listdir(npz_dir) if f.endswith(".npz")])
            if len(files) < 2:
                continue
            dx = float(run["dx"])
            for idx in range(len(files) - 1):
                path_a = os.path.join(npz_dir, files[idx])
                path_b = os.path.join(npz_dir, files[idx + 1])
                data_a = np.load(path_a)
                data_b = np.load(path_b)
                v_a = data_a["V"].astype(np.float32)
                v_b = data_b["V"].astype(np.float32)
                t_a = float(data_a.get("t", 0.0))
                t_b = float(data_b.get("t", 0.0))
                dt_local = max(1e-9, t_b - t_a)
                dVdt = (v_b - v_a) / dt_local
                lap = (
                    np.roll(v_a, 1, axis=0)
                    + np.roll(v_a, -1, axis=0)
                    + np.roll(v_a, 1, axis=1)
                    + np.roll(v_a, -1, axis=1)
                    - 4.0 * v_a
                ) / (dx * dx)
                x1 = lap.ravel()
                x2 = v_a.ravel()
                y = dVdt.ravel()
                A = np.vstack([x1, x2, np.ones_like(x1)]).T
                coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                d_eff, a_eff, c0 = [float(x) for x in coeffs]
                y_pred = A @ coeffs
                resid = y - y_pred
                rmse = float(np.sqrt(np.mean(resid * resid)))
                ss_res = float(np.sum(resid * resid))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
                pde_rows.append(
                    {
                        "t_start": t_a,
                        "t_end": t_b,
                        "D_eff": d_eff,
                        "a_eff": a_eff,
                        "c0": c0,
                        "rmse": rmse,
                        "r2": r2,
                    }
                )
                if not scatter_done:
                    idxs = np.random.default_rng(0).choice(len(y), size=min(5000, len(y)), replace=False)
                    plt.figure(figsize=(5.5, 3.5))
                    plt.scatter(y[idxs], y_pred[idxs], s=4, alpha=0.5)
                    plt.xlabel("observed dV/dt")
                    plt.ylabel("predicted dV/dt")
                    plt.title("Test 9: predicted vs observed")
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, "test9", "pred_vs_obs.png"), dpi=160)
                    plt.close()
                    scatter_done = True
        if pde_rows:
            out_path = os.path.join(out_root, "pde_fit_test9.csv")
            with open(out_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(pde_rows[0].keys()))
                writer.writeheader()
                writer.writerows(pde_rows)

    if test10_items:
        item = test10_items[0]
        summary = {
            "completed_frames": item["frames"],
            "dt": item["dt"],
            "dx": item["dx"],
            "physical_T": item["physical_T"],
            "V_total_min": float(np.nanmin(item["V_total"])) if len(item["V_total"]) else np.nan,
            "V_total_max": float(np.nanmax(item["V_total"])) if len(item["V_total"]) else np.nan,
            "M_total_min": float(np.nanmin(item["M_total"])) if len(item["M_total"]) else np.nan,
            "M_total_max": float(np.nanmax(item["M_total"])) if len(item["M_total"]) else np.nan,
            "radius_est_min": float(np.nanmin(item["radius_est"])) if len(item["radius_est"]) else np.nan,
            "radius_est_max": float(np.nanmax(item["radius_est"])) if len(item["radius_est"]) else np.nan,
            "anisotropy_s16_min": float(np.nanmin(item["anisotropy_s16"])) if len(item["anisotropy_s16"]) else np.nan,
            "anisotropy_s16_max": float(np.nanmax(item["anisotropy_s16"])) if len(item["anisotropy_s16"]) else np.nan,
            "I2_total_min": float(np.nanmin(item["I2_total"])) if len(item["I2_total"]) else np.nan,
            "I2_total_max": float(np.nanmax(item["I2_total"])) if len(item["I2_total"]) else np.nan,
            "any_nan_flags": bool(np.any(~np.isfinite(item["radius_est"])) or np.any(~np.isfinite(item["I2_total"])) or np.any(~np.isfinite(item["anisotropy_s16"]))),
        }
        with open(os.path.join(out_root, "stability_test10_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        t_vals = item["t"]
        plt.figure(figsize=(6, 3.5))
        plt.plot(t_vals, item["radius_phys"], label="radius_phys", linewidth=1)
        plt.plot(t_vals, item["i2_phys"], label="I2_phys", linewidth=1)
        plt.plot(t_vals, item["anisotropy_s16"], label="anisotropy_s16", linewidth=1)
        plt.xlabel("time")
        plt.ylabel("value")
        plt.title("Test 10 stability series")
        plt.legend(loc="best", fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "test10", "series.png"), dpi=160)
        plt.close()

    if test11_items:
        _plot_overlay(
            os.path.join(plots_dir, "test11", "overlay_radius_phys.png"),
            [
                {**item, "series": item["radius_phys"]}
                for item in test11_items
            ],
            "series",
            "radius_phys",
            "Test 11: radius_phys overlay",
        )
        _plot_overlay(
            os.path.join(plots_dir, "test11", "overlay_I2_phys.png"),
            [
                {**item, "series": item["i2_phys"]}
                for item in test11_items
            ],
            "series",
            "I2_phys",
            "Test 11: I2_phys overlay",
        )
        _plot_overlay(
            os.path.join(plots_dir, "test11", "overlay_anisotropy_s16.png"),
            [
                {**item, "series": item["anisotropy_s16"]}
                for item in test11_items
            ],
            "series",
            "anisotropy_s16",
            "Test 11: anisotropy_s16 overlay",
        )
        for i in range(len(test11_items)):
            for j in range(i + 1, len(test11_items)):
                a = test11_items[i]
                b = test11_items[j]
                t_common = np.linspace(0.0, min(np.nanmax(a["t"]), np.nanmax(b["t"])), num=1001)
                a_r = np.interp(t_common, a["t"], a["radius_phys"])
                b_r = np.interp(t_common, b["t"], b["radius_phys"])
                a_i2p = np.interp(t_common, a["t"], a["i2_phys"])
                b_i2p = np.interp(t_common, b["t"], b["i2_phys"])
                a_an = np.interp(t_common, a["t"], a["anisotropy_s16"])
                b_an = np.interp(t_common, b["t"], b["anisotropy_s16"])
                a_i2 = np.interp(t_common, a["t"], a["I2_total"])
                b_i2 = np.interp(t_common, b["t"], b["I2_total"])
                dt_rmse = {
                    "dt_a": a["dt"],
                    "dt_b": b["dt"],
                    "rmse_radius_phys": _rmse_aligned(t_common, a_r, t_common, b_r),
                    "rmse_I2_phys": _rmse_aligned(t_common, a_i2p, t_common, b_i2p),
                    "rmse_aniso_s16": _rmse_aligned(t_common, a_an, t_common, b_an),
                    "rmse_I2_total": _rmse_aligned(t_common, a_i2, t_common, b_i2),
                }
                test11_rmse_rows.append(dt_rmse)
        if test11_rmse_rows:
            out_path = os.path.join(out_root, "dt_invariance_rmse_test11.csv")
            with open(out_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(test11_rmse_rows[0].keys()))
                writer.writeheader()
                writer.writerows(test11_rmse_rows)

    if test12_items:
        _plot_overlay(
            os.path.join(plots_dir, "test12", "overlay_radius_phys.png"),
            [
                {**item, "series": item["radius_phys"]}
                for item in test12_items
            ],
            "series",
            "radius_phys",
            "Test 12: radius_phys overlay",
        )
        _plot_overlay(
            os.path.join(plots_dir, "test12", "overlay_I2_phys.png"),
            [
                {**item, "series": item["i2_phys"]}
                for item in test12_items
            ],
            "series",
            "I2_phys",
            "Test 12: I2_phys overlay",
        )
        _plot_overlay(
            os.path.join(plots_dir, "test12", "overlay_anisotropy_s16.png"),
            [
                {**item, "series": item["anisotropy_s16"]}
                for item in test12_items
            ],
            "series",
            "anisotropy_s16",
            "Test 12: anisotropy_s16 overlay",
        )
        by_n = {item["grid_size"]: item for item in test12_items}
        ref = by_n.get(256)
        if ref:
            for n_key in [128, 512]:
                item = by_n.get(n_key)
                if not item:
                    continue
                t_common = np.linspace(0.0, min(np.nanmax(ref["t"]), np.nanmax(item["t"])), num=1001)
                a_r = np.interp(t_common, item["t"], item["radius_phys"])
                b_r = np.interp(t_common, ref["t"], ref["radius_phys"])
                a_i2p = np.interp(t_common, item["t"], item["i2_phys"])
                b_i2p = np.interp(t_common, ref["t"], ref["i2_phys"])
                a_an = np.interp(t_common, item["t"], item["anisotropy_s16"])
                b_an = np.interp(t_common, ref["t"], ref["anisotropy_s16"])
                a_i2 = np.interp(t_common, item["t"], item["I2_total"])
                b_i2 = np.interp(t_common, ref["t"], ref["I2_total"])
                test12_rmse_rows.append(
                    {
                        "N_a": n_key,
                        "N_ref": 256,
                        "rmse_radius_phys": _rmse_aligned(t_common, a_r, t_common, b_r),
                        "rmse_I2_phys": _rmse_aligned(t_common, a_i2p, t_common, b_i2p),
                        "rmse_aniso_s16": _rmse_aligned(t_common, a_an, t_common, b_an),
                        "rmse_I2_total": _rmse_aligned(t_common, a_i2, t_common, b_i2),
                    }
                )
        if test12_rmse_rows:
            out_path = os.path.join(out_root, "refinement_rmse_test12.csv")
            with open(out_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(test12_rmse_rows[0].keys()))
                writer.writeheader()
                writer.writerows(test12_rmse_rows)

    if rmse_rows and rmse_rows_6b:
        compare_rows = []
        def _find_pair(rows, pair):
            for row in rows:
                if row.get("pair") == pair:
                    return row
            return None
        base = _find_pair(rmse_rows, "N256_vs_N512")
        alt = _find_pair(rmse_rows_6b, "N256_vs_N512")
        if base and alt:
            compare_rows.append(
                {
                    "pair": "N256_vs_N512",
                    "rmse_r_norm_test6": base.get("rmse_r_norm"),
                    "rmse_r_norm_test6B": alt.get("rmse_r_norm"),
                    "rmse_area_frac_test6": base.get("rmse_area_frac"),
                    "rmse_area_frac_test6B": alt.get("rmse_area_frac"),
                    "improved_r_norm": _safe_float(alt.get("rmse_r_norm")) < _safe_float(base.get("rmse_r_norm")),
                    "improved_area_frac": _safe_float(alt.get("rmse_area_frac")) < _safe_float(base.get("rmse_area_frac")),
                }
            )
        if compare_rows:
            compare_path = os.path.join(out_root, "continuum_compare_test6_vs_test6B.csv")
            with open(compare_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(compare_rows[0].keys()))
                writer.writeheader()
                writer.writerows(compare_rows)

    _plot_test1_stability(os.path.join(plots_dir, "test1", "stability_boundary.png"), runs_summary)

    if test5_items:
        _plot_scaling(os.path.join(plots_dir, "scaling", "R_over_N.png"), test5_items, mode="R")
        _plot_scaling(os.path.join(plots_dir, "scaling", "area_over_N2.png"), test5_items, mode="area")
        for grid_size in sorted({item["grid_size"] for item in test5_items}):
            path = os.path.join(plots_dir, "anisotropy", f"anisotropy_N{grid_size}.png")
            _plot_anisotropy(path, test5_items, grid_size)

    if test6_items:
        _plot_overlay(
            os.path.join(plots_dir, "continuum", "radius_over_N_vs_t.png"),
            [
                {**item, "radius_over_N": item["radius_est"] / float(item["grid_size"])}
                for item in test6_items
            ],
            "radius_over_N",
            "sqrt(area/pi) / N",
            "Test 6 continuum: radius/N vs t",
        )
        _plot_overlay(
            os.path.join(plots_dir, "continuum", "area_over_N2_vs_t.png"),
            [
                {**item, "area_over_N2": item["area"] / float(item["grid_size"] ** 2)}
                for item in test6_items
            ],
            "area_over_N2",
            "area / N^2",
            "Test 6 continuum: area/N^2 vs t",
        )
        _plot_overlay(
            os.path.join(plots_dir, "continuum", "anisotropy_s16_vs_t.png"),
            test6_items,
            "anisotropy_M_smoothed_s16",
            "anisotropy s16",
            "Test 6 continuum: anisotropy s16 vs t",
        )
        _plot_overlay(
            os.path.join(plots_dir, "continuum", "I2_total_vs_t.png"),
            test6_items,
            "I2_total",
            "I2_total",
            "Test 6 continuum: I2_total vs t",
        )

        by_n = {}
        for item in test6_items:
            by_n.setdefault(item["grid_size"], []).append(item)
        def _pick_dt(items):
            return sorted(items, key=lambda x: x["dt"])[0] if items else None
        n128 = _pick_dt(by_n.get(128, []))
        n256 = _pick_dt(by_n.get(256, []))
        n512 = _pick_dt(by_n.get(512, []))
        if n128 and n256:
            rmse_r = _rmse_aligned(n128["t"], n128["radius_est"] / 128.0, n256["t"], n256["radius_est"] / 256.0)
            rmse_a = _rmse_aligned(n128["t"], n128["area"] / (128.0 * 128.0), n256["t"], n256["area"] / (256.0 * 256.0))
            rmse_aniso = _rmse_aligned(n128["t"], n128["anisotropy_M_smoothed_s16"], n256["t"], n256["anisotropy_M_smoothed_s16"])
            rmse_i2 = _rmse_aligned(n128["t"], n128["I2_total"], n256["t"], n256["I2_total"])
            rmse_rows.append(
                {
                    "pair": "N128_vs_N256",
                    "rmse_r_norm": rmse_r,
                    "rmse_area_frac": rmse_a,
                    "rmse_anisotropy_s16": rmse_aniso,
                    "rmse_I2_total": rmse_i2,
                }
            )
        if n256 and n512:
            rmse_r = _rmse_aligned(n256["t"], n256["radius_est"] / 256.0, n512["t"], n512["radius_est"] / 512.0)
            rmse_a = _rmse_aligned(n256["t"], n256["area"] / (256.0 * 256.0), n512["t"], n512["area"] / (512.0 * 512.0))
            rmse_aniso = _rmse_aligned(n256["t"], n256["anisotropy_M_smoothed_s16"], n512["t"], n512["anisotropy_M_smoothed_s16"])
            rmse_i2 = _rmse_aligned(n256["t"], n256["I2_total"], n512["t"], n512["I2_total"])
            rmse_rows.append(
                {
                    "pair": "N256_vs_N512",
                    "rmse_r_norm": rmse_r,
                    "rmse_area_frac": rmse_a,
                    "rmse_anisotropy_s16": rmse_aniso,
                    "rmse_I2_total": rmse_i2,
                }
            )

    if test6b_items:
        _plot_overlay(
            os.path.join(plots_dir, "continuum6b", "area_over_N2_vs_t.png"),
            [
                {**item, "area_over_N2": item["area"] / float(item["grid_size"] ** 2)}
                for item in test6b_items
            ],
            "area_over_N2",
            "area / N^2",
            "Test 6B CFL: area/N^2 vs t",
        )
        _plot_overlay(
            os.path.join(plots_dir, "continuum6b", "radius_over_N_vs_t.png"),
            [
                {**item, "radius_over_N": item["radius_est"] / float(item["grid_size"])}
                for item in test6b_items
            ],
            "radius_over_N",
            "sqrt(area/pi) / N",
            "Test 6B CFL: radius/N vs t",
        )
        _plot_overlay(
            os.path.join(plots_dir, "continuum6b", "anisotropy_s16_vs_t.png"),
            test6b_items,
            "anisotropy_M_smoothed_s16",
            "anisotropy s16",
            "Test 6B CFL: anisotropy s16 vs t",
        )
        _plot_overlay(
            os.path.join(plots_dir, "continuum6b", "I2_total_vs_t.png"),
            test6b_items,
            "I2_total",
            "I2_total",
            "Test 6B CFL: I2_total vs t",
        )

        by_n_6b = {}
        for item in test6b_items:
            by_n_6b.setdefault(item["grid_size"], []).append(item)
        def _pick_dt(items):
            return sorted(items, key=lambda x: x["dt"])[0] if items else None
        n128 = _pick_dt(by_n_6b.get(128, []))
        n256 = _pick_dt(by_n_6b.get(256, []))
        n512 = _pick_dt(by_n_6b.get(512, []))
        if n128 and n256:
            rmse_r = _rmse_aligned(n128["t"], n128["radius_est"] / 128.0, n256["t"], n256["radius_est"] / 256.0)
            rmse_a = _rmse_aligned(n128["t"], n128["area"] / (128.0 * 128.0), n256["t"], n256["area"] / (256.0 * 256.0))
            rmse_aniso = _rmse_aligned(n128["t"], n128["anisotropy_M_smoothed_s16"], n256["t"], n256["anisotropy_M_smoothed_s16"])
            rmse_i2 = _rmse_aligned(n128["t"], n128["I2_total"], n256["t"], n256["I2_total"])
            rmse_rows_6b.append(
                {
                    "pair": "N128_vs_N256",
                    "rmse_r_norm": rmse_r,
                    "rmse_area_frac": rmse_a,
                    "rmse_anisotropy_s16": rmse_aniso,
                    "rmse_I2_total": rmse_i2,
                }
            )
        if n256 and n512:
            rmse_r = _rmse_aligned(n256["t"], n256["radius_est"] / 256.0, n512["t"], n512["radius_est"] / 512.0)
            rmse_a = _rmse_aligned(n256["t"], n256["area"] / (256.0 * 256.0), n512["t"], n512["area"] / (512.0 * 512.0))
            rmse_aniso = _rmse_aligned(n256["t"], n256["anisotropy_M_smoothed_s16"], n512["t"], n512["anisotropy_M_smoothed_s16"])
            rmse_i2 = _rmse_aligned(n256["t"], n256["I2_total"], n512["t"], n512["I2_total"])
            rmse_rows_6b.append(
                {
                    "pair": "N256_vs_N512",
                    "rmse_r_norm": rmse_r,
                    "rmse_area_frac": rmse_a,
                    "rmse_anisotropy_s16": rmse_aniso,
                    "rmse_I2_total": rmse_i2,
                }
            )

    print(f"Outputs written to: {out_root}")


if __name__ == "__main__":
    main()
