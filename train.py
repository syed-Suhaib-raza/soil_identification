"""
5-fold GROUPED cross-validation for soil heavy-metal regression from HSV images.

Why "grouped"? The dataset is 36 original soil-sample photos augmented into
~650 images. A plain random 5-fold split would scatter augmented copies of
the SAME original photo across train/val/test, so the model could partly
"memorize" a sample in training and then get evaluated on a near-duplicate
of it in validation -- inflated, contaminated scores. Instead, every split
below groups by the original sample id, so all augmented copies of a given
sample always land on the same side of every split.
Pipeline:
  1. Split the 36 original samples into dev (~29) and test (~7) groups.
     The test images are set aside and not touched again until the very end.
  2. Run 5-fold GroupKFold CV *within the dev set only* -> this is your
     cross-validated performance estimate, and the honest way to compare
     hyperparameters / architectures without ever peeking at the test set.
  3. Train one final model on the full dev set, evaluate it ONCE on the
     held-out test set for an unbiased final number.

Run:  python train.py
"""
import pandas as pd
import torch

import config
from dataset import build_file_table
from model import build_model
from utils import (
    set_seed, split_off_test_groups, split_by_group, make_group_kfold_splits,
    compute_hsv_stats, TargetScaler, make_loader,
    train_one_model, predict, regression_report,
)


def main():
    set_seed()
    device = config.DEVICE
    print(f"Using device: {device}")

    file_table = build_file_table()

    # 1) Hold out a test set BY GROUP - untouched until the very end.
    dev_table, test_table = split_off_test_groups(file_table)
    print(f"Dev set:  {len(dev_table)} images / {dev_table['group_id'].nunique()} samples")
    print(f"Test set: {len(test_table)} images / {test_table['group_id'].nunique()} samples")

    fold_reports = []

    # 2) 5-fold group CV on the dev set only.
    for fold, (train_table, val_table) in enumerate(make_group_kfold_splits(dev_table)):
        print(f"\n===== Fold {fold + 1}/{config.N_FOLDS} =====")
        print(f"train: {len(train_table)} imgs / {train_table['group_id'].nunique()} samples | "
              f"val: {len(val_table)} imgs / {val_table['group_id'].nunique()} samples")

        # HSV stats + target scaler fit on TRAIN split of this fold only.
        hsv_mean, hsv_std = compute_hsv_stats(train_table)
        scaler = TargetScaler(train_table)

        train_loader = make_loader(train_table, hsv_mean, hsv_std, augment=True, shuffle=True)
        val_loader = make_loader(val_table, hsv_mean, hsv_std, augment=False, shuffle=False)

        model = build_model()
        model, best_val_loss = train_one_model(model, train_loader, val_loader, scaler, device)

        y_true, y_pred = predict(model, val_loader, scaler, device)
        report = regression_report(y_true, y_pred)
        report["fold"] = fold + 1
        fold_reports.append(report)
        print(report.to_string(index=False))

    all_folds = pd.concat(fold_reports, ignore_index=True)

    print("\n===== 5-fold CV summary: OVERALL (mean +/- std across folds) =====")
    overall = all_folds[all_folds["target"] == "OVERALL"]
    print(overall[["MAE", "RMSE", "R2"]].agg(["mean", "std"]))

    print("\n===== 5-fold CV summary: per target (mean +/- std across folds) =====")
    per_target = all_folds[all_folds["target"] != "OVERALL"]
    print(per_target.groupby("target")[["MAE", "RMSE", "R2"]].agg(["mean", "std"]))

    all_folds.to_csv("cv_fold_results.csv", index=False)

    # 3) Final model: train on ALL dev data, evaluate ONCE on the held-out test set.
    print("\n===== Training final model on the full dev set =====")
    hsv_mean, hsv_std = compute_hsv_stats(dev_table)
    scaler = TargetScaler(dev_table)

    # small grouped slice purely for early-stopping the final model
    # (different seed so it isn't identical to the fold-1 split; still by group,
    # still nothing to do with the held-out test set)
    final_train_table, final_es_table = split_by_group(
        dev_table, config.FINAL_ES_FRACTION_OF_GROUPS, seed=config.RANDOM_SEED + 1
    )

    train_loader = make_loader(final_train_table, hsv_mean, hsv_std, augment=True, shuffle=True)
    es_loader = make_loader(final_es_table, hsv_mean, hsv_std, augment=False, shuffle=False)
    test_loader = make_loader(test_table, hsv_mean, hsv_std, augment=False, shuffle=False)

    final_model = build_model()
    final_model, _ = train_one_model(final_model, train_loader, es_loader, scaler, device)

    y_true, y_pred = predict(final_model, test_loader, scaler, device)
    test_report = regression_report(y_true, y_pred)
    print("\n===== FINAL held-out test performance (never used in CV or training) =====")
    print(test_report.to_string(index=False))
    test_report.to_csv("final_test_results.csv", index=False)

    torch.save(final_model.state_dict(), "final_model.pt")
    print("\nSaved: final_model.pt, cv_fold_results.csv, final_test_results.csv")


if __name__ == "__main__":
    main()
