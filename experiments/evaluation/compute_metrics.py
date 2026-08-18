"""
compute_metrics.py
==================
Walk a Hydra multirun sweep directory and compute per-run and aggregated
evaluation metrics across all (scenario, planner/config) combinations.

Metrics computed
----------------
  final_nrmse
      Last value of the precomputed ``rmse_history`` stored in history.pkl.
      Consistent with the key used in plot_metrics.py.

  final_norm_trace_reduction
      Last value of the precomputed ``normalizedTraceReduction_history``.
      Consistent with the key used in plot_metrics.py.

  fraction_in_domain
      Fraction of visited waypoints that lie inside [0, W] × [0, H]:

          FID = steps_inside / (T + 1)

      Using a fraction (rather than raw step count) makes runs of different
      lengths comparable.

Output
------
  <out_dir>/metrics_per_run.csv      – one row per (scenario, config)
  <out_dir>/metrics_aggregated.csv   – mean ± 95% CI per config
  <out_dir>/metrics_aggregated.json  – same data, machine-readable

Usage
-----
    # Auto-discover the most recent sweep under the default dir:
    python compute_metrics.py

    # Specify a sweep root or timestamp subdir:
    python compute_metrics.py --sweep_dir results/evaluation/sweep/2026-04-01_12-00-00

    # Override domain size (must match what was used during evaluation):
    python compute_metrics.py --domain_size 15 15

    # Custom output directory:
    python compute_metrics.py --out_dir results/metrics
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def final_nrmse(history: dict) -> float:
    """
    Read the last value of the precomputed rmse_history stored in history.pkl.
    Consistent with the key name used in plot_metrics.py.
    """
    vals = history.get("rmse_history")
    if vals is None or len(vals) == 0:
        return float("nan")
    return float(vals[-1])


def final_norm_trace_reduction(history: dict) -> float:
    """
    Read the last value of the precomputed normalizedTraceReduction_history.
    Consistent with the key name used in plot_metrics.py.
    """
    vals = history.get("normalizedTraceReduction_history")
    if vals is None or len(vals) == 0:
        return float("nan")
    return float(vals[-1])


def fraction_in_domain(position_history, domain_size) -> float:
    """
    Fraction of waypoints (including the starting position) that lie inside
    the closed domain [0, W] × [0, H].
    """
    W, H = domain_size
    pos = np.array(position_history)   # (T+1, 2)
    inside = (
        (pos[:, 0] >= 0) & (pos[:, 0] <= W) &
        (pos[:, 1] >= 0) & (pos[:, 1] <= H)
    )
    return float(inside.mean())


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def resolve_sweep_root(sweep_dir: Path) -> Path:
    """
    If sweep_dir already contains history.pkl files at the expected depth
    (*/*/history.pkl), return it directly.  Otherwise descend into the
    lexicographically latest subdirectory (auto-selects most recent timestamp).
    Mirrors resolve_sweep_root in plot_metrics.py.
    """
    if any(sweep_dir.glob("*/*/history.pkl")):
        return sweep_dir
    subdirs = sorted(d for d in sweep_dir.iterdir() if d.is_dir())
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {sweep_dir}")
    return subdirs[-1]


# ---------------------------------------------------------------------------
# Confidence interval (non-parametric bootstrap, two-sided 95%)
# ---------------------------------------------------------------------------

def mean_ci95(
    values: list[float],
    n_bootstrap: int = 10_000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """
    Returns (mean, lower_bound, upper_bound) using a percentile bootstrap.

    For each replicate, resample *with replacement* from ``values`` and record
    the resample mean.  The 2.5th and 97.5th percentiles of that distribution
    form the two-sided 95% CI.

    This makes no distributional assumption (unlike the t-interval) and is
    appropriate for the small scenario counts typical in sweep evaluations.

    Edge cases
    ----------
    * n == 0  →  (nan, nan, nan)
    * n == 1  →  (value, value, value)  — CI is degenerate but non-crashing

    Parameters
    ----------
    values      : raw per-scenario metric values (NaNs are dropped)
    n_bootstrap : number of bootstrap replicates (default 10 000)
    rng         : optional numpy Generator for reproducibility
    """
    arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if n == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])

    if rng is None:
        rng = np.random.default_rng()

    # (n_bootstrap, n) resample matrix — one vectorised call
    resamples = rng.choice(arr, size=(n_bootstrap, n), replace=True)
    boot_means = resamples.mean(axis=1)

    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute(sweep_dir: Path, domain_size: tuple[float, float]) -> pd.DataFrame:
    """
    Walk the sweep directory and return a DataFrame with one row per
    (scenario, config) and columns for each metric.
    """
    sweep_root = resolve_sweep_root(sweep_dir)
    pkl_files = sorted(sweep_root.glob("*/*/history.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No history.pkl files found under: {sweep_root}")

    records = []
    for pkl_path in pkl_files:
        # Use relative_to so parsing is independent of absolute path depth
        rel_parts = pkl_path.relative_to(sweep_root).parts  # (scenario, config, history.pkl)
        if len(rel_parts) != 3:
            print(f"  Skipping unexpected path structure: {pkl_path}", file=sys.stderr)
            continue
        scenario = rel_parts[0]   # e.g. "scenario_0"
        config   = rel_parts[1]   # e.g. "diffusion", "bo"

        try:
            history = joblib.load(pkl_path)
        except Exception as exc:
            print(f"  WARN: could not load {pkl_path}: {exc}", file=sys.stderr)
            continue

        pos_hist = history.get("position_history")
        if pos_hist is None:
            print(f"  WARN: missing 'position_history' in {pkl_path}, skipping.", file=sys.stderr)
            continue

        records.append({
            "scenario":                   scenario,
            "config":                     config,
            "final_nrmse":                final_nrmse(history),
            "final_norm_trace_reduction":  final_norm_trace_reduction(history),
            "fraction_in_domain":         fraction_in_domain(pos_hist, domain_size),
        })
        print(f"  OK  {scenario}/{config}")

    return pd.DataFrame(records)


def aggregate(
    df: pd.DataFrame,
    n_bootstrap: int = 10_000,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Group by config and compute mean + 95% bootstrap CI for each metric.
    Returns a flat DataFrame ready for CSV export.

    A single Generator (seeded for reproducibility) is shared across all
    configs and metrics so results are stable between runs.
    """
    rng = np.random.default_rng(seed)
    metric_cols = ["final_nrmse", "final_norm_trace_reduction", "fraction_in_domain"]
    rows = []
    for config, grp in df.groupby("config"):
        row = {"config": config, "n_scenarios": len(grp)}
        for col in metric_cols:
            m, lo, hi = mean_ci95(grp[col].tolist(), n_bootstrap=n_bootstrap, rng=rng)
            row[f"{col}_mean"]     = m
            row[f"{col}_ci95_lo"]  = lo
            row[f"{col}_ci95_hi"]  = hi
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute evaluation metrics over a Hydra multirun sweep."
    )
    parser.add_argument(
        "--sweep_dir",
        type=Path,
        default=Path("results/evaluation/sweep"),
        help="Hydra sweep root or specific timestamp subdir.",
    )
    parser.add_argument(
        "--domain_size",
        type=float,
        nargs=2,
        default=[15.0, 15.0],
        metavar=("W", "H"),
        help="Physical domain size used during evaluation (default: 15 15).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Directory for output files. Defaults to <sweep_dir>/metrics/.",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=10_000,
        metavar="B",
        help="Number of bootstrap replicates for 95%% CI (default: 10000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="S",
        help="Random seed for bootstrap RNG (omit for non-deterministic).",
    )
    args = parser.parse_args()

    domain_size = tuple(args.domain_size)

    sweep_root = resolve_sweep_root(args.sweep_dir)
    out_dir = args.out_dir or sweep_root
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sweep root : {sweep_root}")
    print(f"Domain     : {domain_size}")
    print(f"Output dir : {out_dir}")
    print(f"Bootstrap  : B={args.n_bootstrap}, seed={args.seed}")
    print()

    # ------------------------------------------------------------------
    # Per-run metrics
    # ------------------------------------------------------------------
    try:
        df_runs = compute(sweep_root, domain_size)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if df_runs.empty:
        print("ERROR: no valid runs found.", file=sys.stderr)
        sys.exit(1)

    per_run_path = out_dir / "metrics_per_run.csv"
    df_runs.to_csv(per_run_path, index=False, float_format="%.6f")
    print(f"\nPer-run CSV      → {per_run_path}")

    # ------------------------------------------------------------------
    # Aggregated: mean + 95% CI per config
    # ------------------------------------------------------------------
    df_agg = aggregate(df_runs, n_bootstrap=args.n_bootstrap, seed=args.seed)

    agg_csv_path = out_dir / "metrics_aggregated.csv"
    df_agg.to_csv(agg_csv_path, index=False, float_format="%.6f")
    print(f"Aggregated CSV   → {agg_csv_path}")

    # JSON: nested dict  {config: {metric: {mean, ci95_lo, ci95_hi}}}
    metric_cols = ["final_nrmse", "final_norm_trace_reduction", "fraction_in_domain"]
    json_out = {}
    for _, row in df_agg.iterrows():
        config = row["config"]
        json_out[config] = {"n_scenarios": int(row["n_scenarios"])}
        for col in metric_cols:
            json_out[config][col] = {
                "mean":    round(float(row[f"{col}_mean"]),    6),
                "ci95_lo": round(float(row[f"{col}_ci95_lo"]), 6),
                "ci95_hi": round(float(row[f"{col}_ci95_hi"]), 6),
            }

    agg_json_path = out_dir / "metrics_aggregated.json"
    with open(agg_json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"Aggregated JSON  → {agg_json_path}")

    # ------------------------------------------------------------------
    # Pretty-print summary table
    # ------------------------------------------------------------------
    print("\n── Aggregated metrics (mean [95% CI]) ──────────────────────────────────────────────")
    col_w = max(len(c) for c in df_agg["config"]) + 2
    header = (
        f"{'Config':<{col_w}}  {'Final NRMSE':>24}"
        f"  {'Norm. Trace Red.':>24}  {'Frac. in Domain':>24}"
    )
    print(header)
    print("─" * len(header))
    for _, row in df_agg.iterrows():
        def fmt(col):
            m, lo, hi = row[f"{col}_mean"], row[f"{col}_ci95_lo"], row[f"{col}_ci95_hi"]
            return f"{m:.4f} [{lo:.4f}, {hi:.4f}]"
        print(
            f"{row['config']:<{col_w}}  "
            f"{fmt('final_nrmse'):>24}  "
            f"{fmt('final_norm_trace_reduction'):>24}  "
            f"{fmt('fraction_in_domain'):>24}"
        )
    print()


if __name__ == "__main__":
    main()