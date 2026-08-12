"""
Repeated grouped holdout evaluation.

Why this exists: with only ~36 independent soil samples, a single fixed
80/20 dev/test split is a genuinely noisy estimate of how well the model
generalizes -- whichever ~7 samples happen to land in "test" can swing
performance a lot, purely by chance (e.g. if that particular draw happens
to be unusually metal-rich or metal-poor compared to the rest).

This script repeats the WHOLE dev/test split + final-model training +
test evaluation across N_REPEATED_HOLDOUTS different random seeds, then
reports the mean and spread of test performance across all of them. A
tight spread means your single-split result from train.py was close to
representative; a wide spread confirms it wasn't, and the mean here is a
more trustworthy number to actually report.

This does NOT re-run the 5-fold CV inside every repeat (that would
multiply runtime by ~5x for no real benefit here, since we're not tuning
hyperparameters) -- it trains exactly one final model per split.

Run:  python repeated_holdout.py
"""
import numpy as np
import pandas as pd
import torch

import config
from dataset import build_file_table
from model import build_model
from utils import (
    set_seed, split_by_group, compute_hsv_stats, TargetScaler,
    make_loader, train_one_model, predict, regression_report,
)


def run_one_split(file_table, seed, device):
    dev_table, test_table = split_by_group(file_table, config.TEST_FRACTION_OF_GROUPS, seed=seed)
    final_train_table, final_es_table = split_by_group(
        dev_table, config.FINAL_ES_FRACTION_OF_GROUPS, seed=seed + 1
    )

    hsv_mean, hsv_std = compute_hsv_stats(dev_table)
    scaler = TargetScaler(dev_table)

    train_loader = make_loader(final_train_table, hsv_mean, hsv_std, augment=True, shuffle=True)
    es_loader = make_loader(final_es_table, hsv_mean, hsv_std, augment=False, shuffle=False)
    test_loader = make_loader(test_table, hsv_mean, hsv_std, augment=False, shuffle=False)

    model = build_model()
    model, _ = train_one_model(model, train_loader, es_loader, scaler, device, verbose=False)

    y_true, y_pred = predict(model, test_loader, scaler, device)
    report = regression_report(y_true, y_pred)
    report["seed"] = seed
    report["test_groups"] = ", ".join(sorted(test_table["group_id"].unique()))
    return report


def main():
    set_seed()
    device = config.DEVICE
    n_repeats = config.N_REPEATED_HOLDOUTS
    base_seed = config.REPEATED_HOLDOUT_BASE_SEED

    print(f"Using device: {device}")
    print(f"Running {n_repeats} repeated grouped holdout splits (this trains "
          f"{n_repeats} separate models -- will take a while)...\n")

    file_table = build_file_table()

    all_reports = []
    for i in range(n_repeats):
        seed = base_seed + i
        print(f"===== Repeat {i + 1}/{n_repeats} (seed={seed}) =====")
        report = run_one_split(file_table, seed, device)
        overall = report[report["target"] == "OVERALL"].iloc[0]
        print(f"  test groups: {report['test_groups'].iloc[0]}")
        print(f"  OVERALL -> MAE={overall['MAE']:.3f}  RMSE={overall['RMSE']:.3f}  R2={overall['R2']:.3f}\n")
        all_reports.append(report)

    all_reports = pd.concat(all_reports, ignore_index=True)
    all_reports.to_csv("repeated_holdout_results.csv", index=False)

    print(f"\n===== Repeated holdout summary: OVERALL (mean/std/min/max across {n_repeats} splits) =====")
    overall = all_reports[all_reports["target"] == "OVERALL"]
    print(overall[["MAE", "RMSE", "R2"]].agg(["mean", "std", "min", "max"]))

    print("\n===== Repeated holdout summary: per target (mean +/- std across splits) =====")
    per_target = all_reports[all_reports["target"] != "OVERALL"]
    print(per_target.groupby("target")[["MAE", "RMSE", "R2"]].agg(["mean", "std"]))

    print(
        "\nInterpretation: if R2's std above is large relative to its mean, your original "
        "single-split test result was likely just an unlucky (or lucky) draw rather than a "
        "reliable estimate of generalization -- report the mean +/- std from this run instead, "
        "or fall back to the 5-fold CV summary from train.py as your headline number."
    )
    print("\nSaved: repeated_holdout_results.csv")


if __name__ == "__main__":
    main()
