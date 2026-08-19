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

# Add project src and experiments directories to system path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import resolve_sweep_root
from evaluation_utils import compute, aggregate


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
        default=Path("results/conditioning_ablation/sweep"),
        help="Hydra sweep root or specific timestamp subdir.",
    )
    parser.add_argument(
        "--domain_size",
        type=float,
        nargs=2,
        default=[300.0, 300.0],
        metavar=("W", "H"),
        help="Physical domain size used during evaluation (default: 300 300).",
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

    sweep_root = resolve_sweep_root(args.sweep_dir)
    domain_size = tuple(args.domain_size)
    try:
        from omegaconf import OmegaConf
        cfg_path = sweep_root / ".hydra" / "config.yaml"
        if cfg_path.exists():
            cfg = OmegaConf.load(cfg_path)
            domain_size = tuple(cfg.get("domain_size", domain_size))
    except Exception:
        pass
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
    print(f"\nPer-run CSV      -> {per_run_path}")

    # ------------------------------------------------------------------
    # Aggregated: mean + 95% CI per config
    # ------------------------------------------------------------------
    df_agg = aggregate(df_runs, n_bootstrap=args.n_bootstrap, seed=args.seed)

    agg_csv_path = out_dir / "metrics_aggregated.csv"
    df_agg.to_csv(agg_csv_path, index=False, float_format="%.6f")
    print(f"Aggregated CSV   -> {agg_csv_path}")

    # JSON: nested dict  {config: {metric: {mean, ci95_lo, ci95_hi}}}
    metric_cols = ["final_nrmse", "fraction_in_domain"]
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
    print(f"Aggregated JSON  -> {agg_json_path}")

    # ------------------------------------------------------------------
    # Pretty-print summary table
    # ------------------------------------------------------------------
    print("\n== Aggregated metrics (mean [95% CI]) ==============================================")
    col_w = max(len(c) for c in df_agg["config"]) + 2
    header = (
        f"{'Config':<{col_w}}  {'Final NRMSE':>24}"
        f"  {'Frac. in Domain':>24}"
    )
    print(header)
    print("-" * len(header))
    for _, row in df_agg.iterrows():
        def fmt(col):
            m, lo, hi = row[f"{col}_mean"], row[f"{col}_ci95_lo"], row[f"{col}_ci95_hi"]
            return f"{m:.4f} [{lo:.4f}, {hi:.4f}]"
        print(
            f"{row['config']:<{col_w}}  "
            f"{fmt('final_nrmse'):>24}  "
            f"{fmt('fraction_in_domain'):>24}"
        )
    print()


if __name__ == "__main__":
    main()