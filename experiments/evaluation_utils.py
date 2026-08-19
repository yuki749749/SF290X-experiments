import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import joblib
import matplotlib.patches as mpatches

# Add project src directory to system path to import resolve_sweep_root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils import resolve_sweep_root


# ===========================================================================
# Metric Computation Helpers (compute_metrics.py)
# ===========================================================================

def final_nrmse(history: dict) -> float:
    """
    Read the last value of the precomputed rmse_history stored in history.pkl.
    Consistent with the key name used in plot_metrics.py.
    """
    vals = history.get("rmse_history")
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


def mean_ci95(
    values: list[float],
    n_bootstrap: int = 10_000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """
    Returns (mean, lower_bound, upper_bound) using a percentile bootstrap.
    """
    arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if n == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])

    if rng is None:
        rng = np.random.default_rng()

    # (n_bootstrap, n) resample matrix
    resamples = rng.choice(arr, size=(n_bootstrap, n), replace=True)
    boot_means = resamples.mean(axis=1)

    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


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
    """
    rng = np.random.default_rng(seed)
    metric_cols = ["final_nrmse", "fraction_in_domain"]
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


# ===========================================================================
# Metric/Result Discovery & Loading Helpers (plot_metrics.py)
# ===========================================================================

def discover_results(sweep_root: Path) -> dict[str, list[dict]]:
    """
    Walk sweep_root and collect history dicts keyed by planner name.

    Expected layout:
        sweep_root/scenario_<i>/<planner>/history.pkl
    """
    data: dict[str, list[dict]] = {}
    pkl_files = sorted(sweep_root.glob("*/*/history.pkl"))

    if not pkl_files:
        raise FileNotFoundError(f"No history.pkl files found under {sweep_root}")

    for pkl_path in pkl_files:
        rel_parts = pkl_path.relative_to(sweep_root).parts
        if len(rel_parts) != 3:
            print(f"  Skipping unexpected path structure: {pkl_path}", file=sys.stderr)
            continue
        planner_name = rel_parts[-2]   # e.g. 'bo', 'lawnmower', 'diffusion'
        history = joblib.load(pkl_path)
        data.setdefault(planner_name, []).append(history)

    return data


def extract_metric(runs: list[dict], key: str) -> np.ndarray:
    """
    Stack metric histories → (n_runs, T).
    Shorter runs are padded to max length by repeating their last value.
    """
    arrays = [[float(v) for v in h[key]] for h in runs]
    max_len = max(len(a) for a in arrays)
    padded = [a + [a[-1]] * (max_len - len(a)) for a in arrays]
    return np.array(padded)


def mean_ci(matrix: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, lower_95, upper_95) across axis=0."""
    mu  = matrix.mean(axis=0)
    sem = matrix.std(axis=0, ddof=min(1, matrix.shape[0] - 1)) / np.sqrt(matrix.shape[0])
    return mu, mu - z * sem, mu + z * sem


def mean_ci_bootstrap(
    matrix: np.ndarray,
    n_bootstrap: int = 10_000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (mean, lower_95, upper_95) across axis=0 using a percentile
    bootstrap. Each column (time step) is resampled independently.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = matrix.shape[0]
    mu = matrix.mean(axis=0)

    if n == 1:
        return mu, mu.copy(), mu.copy()

    # Resample scenario indices: (n_bootstrap, n)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    # Boot means: (n_bootstrap, T)
    boot_means = matrix[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5], axis=0)
    return mu, lo, hi


# ===========================================================================
# Visual & Grid Helpers (plot_comparison.py, plot_trajectory.py)
# ===========================================================================

def to_grid(arr, side: int) -> np.ndarray:
    """Flat tensor / ndarray → (side, side) numpy array."""
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    return np.asarray(arr).reshape(side, side)


def infer_side(flat_arr) -> int:
    N = len(np.asarray(flat_arr).flatten())
    side = int(round(np.sqrt(N)))
    assert side * side == N, f"Evaluation grid is not square (N={N})."
    return side


def add_domain_box(ax, domain_size, pad: float) -> None:
    """Dashed white rectangle at the domain boundary; expand axis limits."""
    W, H = domain_size
    rect = mpatches.Rectangle(
        (0, 0), W, H,
        linewidth=1.0,
        edgecolor="white",
        facecolor="none",
        linestyle="--",
        zorder=5,
    )
    ax.add_patch(rect)
    ax.set_xlim(-pad, W + pad)
    ax.set_ylim(-pad, H + pad)


def overlay_trajectory(ax, pos_seq: np.ndarray, domain_size, inside_mask) -> None:
    """Draw trajectory line, in-domain dots, out-of-domain dots, start star."""
    # Full path line
    ax.plot(
        pos_seq[:, 0], pos_seq[:, 1],
        color="white", lw=1.5, alpha=0.6, zorder=6,
    )
    # Out-of-domain waypoints (orange)
    out_mask = ~inside_mask
    if out_mask.any():
        ax.scatter(
            pos_seq[out_mask, 0], pos_seq[out_mask, 1],
            color="orange", s=6, zorder=7, linewidths=0,
        )
    # In-domain waypoints (white)
    ax.scatter(
        pos_seq[inside_mask, 0], pos_seq[inside_mask, 1],
        color="white", s=3, zorder=7, linewidths=0,
    )
    # Start marker
    ax.scatter(
        pos_seq[0, 0], pos_seq[0, 1],
        marker="*", s=60, color="yellow",
        edgecolors="black", linewidths=0.4, zorder=8,
    )


def inside_mask(pos_seq: np.ndarray, domain_size) -> np.ndarray:
    W, H = domain_size
    return (
        (pos_seq[:, 0] >= 0) & (pos_seq[:, 0] <= W) &
        (pos_seq[:, 1] >= 0) & (pos_seq[:, 1] <= H)
    )


def find_history_files(sweep_root: Path) -> list[Path]:
    hits = sorted(sweep_root.glob("*/*/history.pkl"))
    if hits:
        return hits
    subdirs = sorted(d for d in sweep_root.iterdir() if d.is_dir())
    if subdirs:
        hits = sorted(subdirs[-1].glob("*/*/history.pkl"))
    return hits


def label_from_path(pkl_path: Path) -> str:
    parts = pkl_path.parts
    planner  = parts[-2]
    scenario = parts[-3]
    return f"{scenario} / {planner}"


def find_and_group(sweep_root: Path, config_order: list[str]) -> dict[str, list[Path]]:
    """
    Returns {scenario_key: [pkl_cfg1, pkl_cfg2, pkl_cfg3, pkl_cfg4]}
    sorted consistently by config name so order is reproducible.
    """
    hits = sorted(sweep_root.glob("*/*/history.pkl"))
    if not hits:
        # Try one level deeper (timestamp subdirs)
        subdirs = sorted(d for d in sweep_root.iterdir() if d.is_dir())
        if subdirs:
            hits = sorted(subdirs[-1].glob("*/*/history.pkl"))

    groups: dict[str, list[Path]] = defaultdict(list)
    for p in hits:
        scenario = p.parts[-3]   # e.g. "scenario_0"
        groups[scenario].append(p)

    # Sort configs by canonical order
    def _order_key(p: Path) -> int:
        try:
            return config_order.index(p.parent.name)
        except ValueError:
            return 999   # unknown configs go last

    for k in groups:
        groups[k] = sorted(groups[k], key=_order_key)

    return dict(groups)
