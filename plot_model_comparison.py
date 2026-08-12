"""
Model comparison plots for the soil heavy-metal regression pipeline.

Compares however many models you have result files for (e.g. cnn, res18,
res50, dense121), using each model's:
    {MODEL}_cv_fold_results.csv     (per-fold, per-target CV metrics)
    {MODEL}_final_test_results.csv  (per-target held-out test metrics)

NOTE ON ROC/PR CURVES: those require class labels and predicted class
probabilities. This is a regression task predicting continuous metal
concentrations, so ROC/PR curves don't apply here -- there's no "class"
for a Cd concentration to belong to. This script instead plots the
regression comparisons that are actually meaningful for these models:
MAE/RMSE/R^2 across CV folds and on the held-out test set, per target and
overall, fold-to-fold variability (how consistent each model is), and the
CV-vs-test gap (how much the held-out result differs from the CV estimate).

Usage:
    python plot_model_comparison.py
    python plot_model_comparison.py --dir path/to/results --out my_plots

By default, looks for *_cv_fold_results.csv / *_final_test_results.csv in
the current directory and writes PNGs to ./comparison_plots/.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGETS = ["Cd", "Cu", "Ni", "Mn", "Fe", "Zn"]
METRICS = ["MAE", "RMSE", "R2"]


def discover_models(results_dir):
    """
    Finds every {MODEL}_cv_fold_results.csv and confirms the matching
    {MODEL}_final_test_results.csv exists.

    Returns {model_name: {"cv": path, "test": path}}, e.g.
        {"cnn": {...}, "res18": {...}, "res50": {...}, "dense121": {...}}
    """
    cv_files = sorted(glob.glob(os.path.join(results_dir, "*_cv_fold_results.csv")))
    models = {}
    for cv_path in cv_files:
        fname = os.path.basename(cv_path)
        model_name = fname[: -len("_cv_fold_results.csv")]
        test_path = os.path.join(results_dir, f"{model_name}_final_test_results.csv")
        if not os.path.exists(test_path):
            print(
                f"WARNING: found {fname} but no matching "
                f"{model_name}_final_test_results.csv -- skipping '{model_name}'"
            )
            continue
        models[model_name] = {"cv": cv_path, "test": test_path}
    if not models:
        raise FileNotFoundError(
            f"No matching *_cv_fold_results.csv / *_final_test_results.csv pairs "
            f"found in '{results_dir}'. Expected files like 'cnn_cv_fold_results.csv' "
            f"and 'cnn_final_test_results.csv'."
        )
    return models


def load_all(models):
    """Loads and concatenates all models' CV and test results, tagging each row with model name."""
    cv_frames, test_frames = [], []
    for name, paths in models.items():
        cv = pd.read_csv(paths["cv"])
        cv["model"] = name
        cv_frames.append(cv)

        test = pd.read_csv(paths["test"])
        test["model"] = name
        test_frames.append(test)
    return pd.concat(cv_frames, ignore_index=True), pd.concat(test_frames, ignore_index=True)


def _model_colors(model_names):
    cmap = plt.get_cmap("tab10")
    return {m: cmap(i % 10) for i, m in enumerate(model_names)}


def _metric_label(metric):
    return "R$^2$" if metric == "R2" else metric


def plot_cv_overall_summary(cv_df, out_dir, colors):
    """Bar chart: mean +/- std of OVERALL R2/MAE/RMSE across folds, one bar per model."""
    overall = cv_df[cv_df["target"] == "OVERALL"]
    models = list(colors.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric in zip(axes, METRICS):
        means = [overall[overall["model"] == m][metric].mean() for m in models]
        stds = [overall[overall["model"] == m][metric].std() for m in models]
        ax.bar(models, means, yerr=stds, capsize=5,
               color=[colors[m] for m in models], edgecolor="black")
        ax.set_title(f"CV {_metric_label(metric)} (mean \u00b1 std across folds)")
        ax.set_ylabel(_metric_label(metric))
        ax.tick_params(axis="x", rotation=20)
        if metric == "R2":
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    fig.suptitle("Cross-Validation Overall Performance by Model", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_cv_overall_summary.png"), dpi=150)
    plt.close(fig)


def plot_cv_per_target(cv_df, out_dir, colors):
    """Grouped bar chart per target: mean +/- std across folds, bars grouped by model. One figure per metric."""
    per_target = cv_df[cv_df["target"] != "OVERALL"]
    models = list(colors.keys())

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(TARGETS))
        width = 0.8 / len(models)
        for i, m in enumerate(models):
            means, stds = [], []
            for t in TARGETS:
                sub = per_target[(per_target["model"] == m) & (per_target["target"] == t)][metric]
                means.append(sub.mean())
                stds.append(sub.std())
            ax.bar(x + i * width, means, width, yerr=stds, capsize=3,
                   label=m, color=colors[m], edgecolor="black", linewidth=0.5)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(TARGETS)
        ax.set_ylabel(_metric_label(metric))
        ax.set_title(f"CV {_metric_label(metric)} by Target and Model (mean \u00b1 std across folds)")
        if metric == "R2":
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"02_cv_per_target_{metric}.png"), dpi=150)
        plt.close(fig)


def plot_cv_fold_variability(cv_df, out_dir, colors):
    """
    Box plot (+ individual fold points overlaid): distribution of OVERALL
    metric across folds, per model -- shows how consistent/unstable each
    model is fold-to-fold, not just its average.
    """
    overall = cv_df[cv_df["target"] == "OVERALL"]
    models = list(colors.keys())

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8, 5))
        data = [overall[overall["model"] == m][metric].values for m in models]
        bp = ax.boxplot(data, tick_labels=models, patch_artist=True, showmeans=True)
        for patch, m in zip(bp["boxes"], models):
            patch.set_facecolor(colors[m])
            patch.set_alpha(0.6)
        rng = np.random.default_rng(0)
        for i, m in enumerate(models):
            vals = overall[overall["model"] == m][metric].values
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.full(len(vals), i + 1) + jitter, vals, color="black", zorder=3, s=25, alpha=0.7)
        ax.set_ylabel(_metric_label(metric))
        ax.set_title(f"Fold-to-Fold Variability: OVERALL {_metric_label(metric)}")
        if metric == "R2":
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"03_cv_fold_variability_{metric}.png"), dpi=150)
        plt.close(fig)


def plot_test_comparison(test_df, out_dir, colors):
    """Bar chart: held-out test set metric by target, grouped by model. One figure per metric."""
    per_target = test_df[test_df["target"] != "OVERALL"]
    models = list(colors.keys())

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(TARGETS))
        width = 0.8 / len(models)
        for i, m in enumerate(models):
            vals = [
                per_target[(per_target["model"] == m) & (per_target["target"] == t)][metric].values[0]
                for t in TARGETS
            ]
            ax.bar(x + i * width, vals, width, label=m, color=colors[m], edgecolor="black", linewidth=0.5)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(TARGETS)
        ax.set_ylabel(_metric_label(metric))
        ax.set_title(f"Held-Out Test {_metric_label(metric)} by Target and Model")
        if metric == "R2":
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"04_test_per_target_{metric}.png"), dpi=150)
        plt.close(fig)


def plot_cv_vs_test_gap(cv_df, test_df, out_dir, colors):
    """
    Dumbbell plot: CV mean (circle) vs single held-out test value (X) for
    OVERALL metric, per model -- visualizes the generalization gap, i.e.
    how much the one-shot test result differs from the CV estimate.
    """
    cv_overall = cv_df[cv_df["target"] == "OVERALL"]
    test_overall = test_df[test_df["target"] == "OVERALL"]
    models = list(colors.keys())

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, m in enumerate(models):
            cv_mean = cv_overall[cv_overall["model"] == m][metric].mean()
            test_val = test_overall[test_overall["model"] == m][metric].values[0]
            ax.plot([cv_mean, test_val], [i, i], color=colors[m], linewidth=2, zorder=1)
            ax.scatter([cv_mean], [i], color=colors[m], marker="o", s=100, zorder=2)
            ax.scatter([test_val], [i], color=colors[m], marker="X", s=130, zorder=2, edgecolor="black")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.set_xlabel(_metric_label(metric))
        ax.set_title(f"CV Mean (\u25cf) vs Held-Out Test (\u2716): OVERALL {_metric_label(metric)}")
        if metric == "R2":
            ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"05_cv_vs_test_gap_{metric}.png"), dpi=150)
        plt.close(fig)


def plot_r2_heatmap(cv_df, out_dir, colors):
    """Heatmap: rows=models, columns=targets, color=mean CV R2 -- at-a-glance comparison."""
    per_target = cv_df[cv_df["target"] != "OVERALL"]
    models = list(colors.keys())

    matrix = np.zeros((len(models), len(TARGETS)))
    for i, m in enumerate(models):
        for j, t in enumerate(TARGETS):
            matrix[i, j] = per_target[(per_target["model"] == m) & (per_target["target"] == t)]["R2"].mean()

    fig, ax = plt.subplots(figsize=(9, 0.7 * len(models) + 2))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(TARGETS)))
    ax.set_xticklabels(TARGETS)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(TARGETS)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Mean CV R$^2$")
    ax.set_title("CV Mean R$^2$ Heatmap: Model x Target")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_r2_heatmap.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=".", help="Directory containing the *_cv_fold_results.csv / *_final_test_results.csv files")
    parser.add_argument("--out", default="comparison_plots", help="Directory to save PNG plots into")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    models = discover_models(args.dir)
    print(f"Found {len(models)} model(s): {list(models.keys())}")

    cv_df, test_df = load_all(models)
    colors = _model_colors(list(models.keys()))

    plot_cv_overall_summary(cv_df, args.out, colors)
    plot_cv_per_target(cv_df, args.out, colors)
    plot_cv_fold_variability(cv_df, args.out, colors)
    plot_test_comparison(test_df, args.out, colors)
    plot_cv_vs_test_gap(cv_df, test_df, args.out, colors)
    plot_r2_heatmap(cv_df, args.out, colors)

    n_saved = len(glob.glob(os.path.join(args.out, "*.png")))
    print(f"\nSaved {n_saved} plots to {args.out}/")


if __name__ == "__main__":
    main()
