"""
Group-aware splitting (the anti-contamination logic), per-fold scaling,
training loop, and regression metrics.
"""

import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

import config
from dataset import SoilHSVDataset, rgb_to_hsv_tensor


def set_seed(seed=None):
    seed = config.RANDOM_SEED if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------- #
# Group-aware splitting -- this is what prevents contamination.
# Every augmented copy of a given original sample carries the same
# group_id, and every split below keeps whole groups together.
# --------------------------------------------------------------------- #

def split_by_group(table: pd.DataFrame, fraction: float, seed=None):
    seed = config.RANDOM_SEED if seed is None else seed
    gss = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    groups = table["group_id"].values
    idx_a, idx_b = next(gss.split(table, groups=groups))
    a = table.iloc[idx_a].reset_index(drop=True)
    b = table.iloc[idx_b].reset_index(drop=True)
    assert set(a["group_id"]).isdisjoint(set(b["group_id"])), \
        "Leakage: a sample group ended up on both sides of the split!"
    return a, b


def split_off_test_groups(file_table: pd.DataFrame):
    """Held-out test set, split by group. Never touched during CV."""
    return split_by_group(file_table, config.TEST_FRACTION_OF_GROUPS)


def make_group_kfold_splits(dev_table: pd.DataFrame):
    """Yields (train_table, val_table) for each of N_FOLDS, grouped by sample."""
    gkf = GroupKFold(n_splits=config.N_FOLDS)
    groups = dev_table["group_id"].values
    for train_idx, val_idx in gkf.split(dev_table, groups=groups):
        train_table = dev_table.iloc[train_idx].reset_index(drop=True)
        val_table = dev_table.iloc[val_idx].reset_index(drop=True)
        assert set(train_table["group_id"]).isdisjoint(set(val_table["group_id"])), \
            "Leakage: a sample group appears in both train and val!"
        yield train_table, val_table


# --------------------------------------------------------------------- #
# Stats / scaling -- always fit on the TRAIN split of whatever split
# you're in, then applied (not refit) to val/test.
# --------------------------------------------------------------------- #

def compute_hsv_stats(file_table: pd.DataFrame):
    """Per-channel HSV mean/std over a table of images. Call on TRAIN rows only."""
    sums = torch.zeros(3)
    sums_sq = torch.zeros(3)
    n_pixels = 0
    for fp in file_table["filepath"]:
        img = Image.open(fp)
        t = rgb_to_hsv_tensor(img, config.IMAGE_WIDTH, config.IMAGE_HEIGHT)  # (3,H,W), already in [0,1]
        sums += t.sum(dim=(1, 2))
        sums_sq += (t ** 2).sum(dim=(1, 2))
        n_pixels += t.shape[1] * t.shape[2]
    mean = sums / n_pixels
    var = (sums_sq / n_pixels) - mean ** 2
    std = torch.sqrt(var.clamp(min=1e-8))
    return mean.tolist(), std.tolist()


class TargetScaler:
    """Standardizes each of the 6 target columns. Fit on TRAIN targets only."""

    def __init__(self, train_table: pd.DataFrame):
        vals = train_table[config.TARGET_COLUMNS].values.astype(np.float32)
        self.mean = torch.tensor(vals.mean(axis=0), dtype=torch.float32)
        self.std = torch.tensor(vals.std(axis=0) + 1e-8, dtype=torch.float32)

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean) / self.std

    def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std + self.mean


def make_loader(table, mean, std, augment, batch_size=None, shuffle=False):
    batch_size = batch_size or config.BATCH_SIZE
    ds = SoilHSVDataset(table, mean, std, augment=augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# --------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------- #

def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Per-target and overall MAE / RMSE / R^2, in original (unscaled) units."""
    rows = []
    for i, name in enumerate(config.TARGET_COLUMNS):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        rows.append({"target": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    df = pd.DataFrame(rows)
    overall = {
        "target": "OVERALL",
        "MAE": df["MAE"].mean(),
        "RMSE": df["RMSE"].mean(),
        "R2": df["R2"].mean(),
    }
    return pd.concat([df, pd.DataFrame([overall])], ignore_index=True)


# --------------------------------------------------------------------- #
# Train / eval loops
# --------------------------------------------------------------------- #

def train_one_model(model, train_loader, val_loader, scaler, device,
                     epochs=None, patience=None, lr=None, weight_decay=None,
                     verbose=True):
    epochs = config.EPOCHS if epochs is None else epochs
    patience = config.PATIENCE if patience is None else patience
    lr = config.LR if lr is None else lr
    weight_decay = config.WEIGHT_DECAY if weight_decay is None else weight_decay

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(2, patience // 2))
    loss_fn = torch.nn.SmoothL1Loss()

    best_val = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            y_scaled = scaler.transform(y.cpu()).to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y_scaled)
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_scaled = scaler.transform(y.cpu()).to(device)
                pred = model(x)
                loss = loss_fn(pred, y_scaled)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        sched.step(val_loss)

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, best_val


@torch.no_grad()
def predict(model, loader, scaler, device):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x = x.to(device)
        pred_scaled = model(x).cpu()
        pred = scaler.inverse_transform(pred_scaled)
        preds.append(pred.numpy())
        trues.append(y.numpy())
    return np.concatenate(trues), np.concatenate(preds)