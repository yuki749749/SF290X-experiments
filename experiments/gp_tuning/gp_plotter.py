import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex":          False,
    "mathtext.fontset":     "cm",       # Computer Modern via matplotlib's mathtext
    "font.family":          "serif",
    "font.serif":           ["DejaVu Serif", "Times New Roman", "serif"],
    "axes.labelsize":       9,
    "axes.titlesize":       10,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "legend.fontsize":      9,
    "figure.titlesize":     11,
    "axes.linewidth":       0.8,
    "xtick.major.width":    0.8,
    "ytick.major.width":    0.8,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "axes.grid":            True,
    "grid.linestyle":       "--",
    "grid.linewidth":       0.4,
    "grid.alpha":           0.5,
    "lines.linewidth":      1.5,
})

results_dir = "results/gp_tuning/"
hyperparameters_dir = "data/hyperparameters/"

results = joblib.load(os.path.join(results_dir, "gp_tuning_results.pkl"))['per_scenario']

def plot_hyperparameter_distributions(results):
    keys = results[0].keys()
    n_keys = len(keys)
    fig, axes = plt.subplots(1, n_keys, figsize=(4 * n_keys, 4))
    for i, key in enumerate(keys):
        values = [result[key] for result in results]
        axes[i].hist(values, bins=20, alpha=0.7)
        axes[i].set_title(key)
        axes[i].axvline(np.median(values), color='red', linestyle='dashed', linewidth=1, label='Median')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "gp_hyperparameter_distributions.png"))

def main():
    plot_hyperparameter_distributions(results)

if __name__ == "__main__":
    main()
