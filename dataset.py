"""
Data loading for the soil heavy-metal regression pipeline.

The single most important function in this file is parse_original_id().
It maps every augmented image filename back to the original soil sample
it came from. Everything downstream (5-fold CV, the held-out test set)
relies on grouping by this id so that augmented copies of the same
original photo never end up split across train/val/test.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

import config


def _normalize_id(x) -> str:
    """
    Excel silently turns "01" into the number 1, "S03" stays a string, etc.
    This normalizes both the excel ID column and the filename-parsed ID so
    that "01" (from a filename) and 1 (read back from excel) are treated
    as the same sample instead of silently failing to match.
    """
    s = str(x).strip()
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


_ID_PATTERN = re.compile(r"^(\d+)([A-Z]?)([hv])")


def parse_original_id(filename: str) -> str:
    """
    Matches your naming convention, e.g.:
        "1Ah_m_x_a"     -> "1Ah"
        "1Ah_m_x1_b"    -> "1Ah"
        "1Bv_x2_c"      -> "1Bv"
        "10Ch_m_x1_a"   -> "10Ch"
        "13h_m_x1_a"    -> "13h"   (no horizon letter -- composite/bulk sample)

    i.e. leading site number + OPTIONAL horizon letter (A/B/C/...) +
    orientation letter (h/v) is treated as the ORIGINAL sample id.
    Everything after that (an optional "_m" moisture flag, the x/x1/x2
    exposure level, and the a/b/c replicate letter) is treated as a
    variant of that same physical sample and stays in the same group.

    The horizon letter is optional because some sites apparently have a
    composite/bulk sample with no horizon subdivision (e.g. "13h" rather
    than "13Ah"/"13Bh"/etc.) -- these are still treated as their own
    distinct group, just without a horizon letter in the id.

    Convention assumed: horizon letter (if present) is uppercase,
    orientation letter is lowercase h/v -- if your files don't follow
    that case convention, adjust the regex.

    IMPORTANT: your ground-truth excel's ID_COLUMN must contain these same
    values ("1Ah", "1Bv", "10Ch", "13h", ...) -- one row per horizon (or
    per composite sample), since different horizons at the same site will
    generally have different metal concentrations.

    If this assumption is wrong for your data, edit the regex/logic below.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    match = _ID_PATTERN.match(base)
    if not match:
        raise ValueError(
            f"Could not extract a site+horizon id from '{filename}' using the "
            "expected pattern (e.g. '1Ah_m_x1_b'). Edit parse_original_id() "
            "in dataset.py to match your naming scheme."
        )
    site, horizon, orientation = match.groups()
    return f"{site}{horizon.upper()}{orientation.lower()}"


def load_ground_truth() -> pd.DataFrame:
    """
    Loads the excel file and returns a DataFrame indexed by ID_COLUMN.

    In "per_image" mode (your data): ID_COLUMN holds the exact filename
    stem (no extension) of each image -- one row per image, e.g. row
    "1Ah_m_x1_b" provides the target values for "1Ah_m_x1_b.jpg".

    In "per_original" mode: ID_COLUMN instead holds one row per group
    (e.g. "1Ah"), and every augmented image belonging to that group gets
    the same target values from that one row.
    """
    df = pd.read_excel(config.GROUND_TRUTH_XLSX, sheet_name=config.SHEET_NAME)
    if config.ID_COLUMN not in df.columns:
        raise ValueError(
            f"ID_COLUMN='{config.ID_COLUMN}' not found in excel columns: {list(df.columns)}. "
            "Update config.ID_COLUMN to match your file."
        )
    missing = [c for c in config.TARGET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Ground-truth excel is missing expected columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    if config.GROUND_TRUTH_MODE == "per_original":
        df[config.ID_COLUMN] = df[config.ID_COLUMN].map(_normalize_id)
    else:
        df[config.ID_COLUMN] = df[config.ID_COLUMN].astype(str).str.strip()

    dupe_mask = df[config.ID_COLUMN].duplicated(keep=False)
    if dupe_mask.any():
        dupes = df.loc[dupe_mask, [config.ID_COLUMN] + config.TARGET_COLUMNS]
        n_dupe_ids = dupes[config.ID_COLUMN].nunique()
        print(
            f"WARNING: {n_dupe_ids} duplicate '{config.ID_COLUMN}' value(s) in the ground-truth "
            f"excel -- keeping the first occurrence of each, dropping the rest. This usually "
            f"means a data-entry typo (e.g. a row meant for a different sample got labeled with "
            f"an existing id). Affected rows:\n{dupes.to_string(index=False)}"
        )

    return df.drop_duplicates(subset=config.ID_COLUMN, keep="first").set_index(config.ID_COLUMN)


def build_file_table() -> pd.DataFrame:
    """
    Walks IMAGE_DIR and, for every image:
      - parse_original_id() gets the CV-fold group (which original
        physical sample it belongs to -- used only for anti-leakage
        grouping, never for the ground-truth lookup)
      - the ground-truth target values are looked up separately, either
        by the image's own filename stem (per_image mode) or by its
        group id (per_original mode)

    Returns a DataFrame with columns:
        filepath, group_id, Cd, Cu, Ni, Mn, Fe, Zn
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    filepaths = []
    for ext in exts:
        filepaths.extend(glob.glob(os.path.join(config.IMAGE_DIR, "**", ext), recursive=True))
    filepaths = sorted(filepaths)

    if not filepaths:
        raise FileNotFoundError(f"No images found under {config.IMAGE_DIR}")

    gt = load_ground_truth()
    # fallback maps, used only if an exact match fails (in this order)
    gt_lower_map = {str(k).lower(): k for k in gt.index}
    gt_loose_map = {re.sub(r"[_\s-]", "", str(k)).lower(): k for k in gt.index}

    rows = []
    unmatched = []
    case_mismatches = []
    loose_mismatches = []
    for fp in filepaths:
        fname = os.path.basename(fp)
        stem = os.path.splitext(fname)[0]
        group_id = parse_original_id(fname)

        lookup_key = stem if config.GROUND_TRUTH_MODE == "per_image" else _normalize_id(group_id)

        if lookup_key not in gt.index:
            loose_key = re.sub(r"[_\s-]", "", lookup_key).lower()
            if lookup_key.lower() in gt_lower_map:
                case_mismatches.append((fname, lookup_key))
                lookup_key = gt_lower_map[lookup_key.lower()]
            elif loose_key in gt_loose_map:
                loose_mismatches.append((fname, lookup_key, gt_loose_map[loose_key]))
                lookup_key = gt_loose_map[loose_key]
            else:
                unmatched.append((fname, lookup_key))
                continue

        row = {"filepath": fp, "group_id": group_id}
        for col in config.TARGET_COLUMNS:
            row[col] = gt.loc[lookup_key, col]
        rows.append(row)

    if case_mismatches:
        print(
            f"NOTE: {len(case_mismatches)} image(s) matched the ground-truth excel only after "
            f"ignoring case (e.g. filename vs excel ID differ in capitalization). First few: "
            f"{case_mismatches[:10]}"
        )
    if loose_mismatches:
        print(
            f"NOTE: {len(loose_mismatches)} image(s) matched the ground-truth excel only after "
            f"ignoring underscores/spaces/hyphens (e.g. filename 'x1a' vs excel id 'x1_a'). "
            f"VERIFY these are actually the same sample, not a coincidental collision:\n"
            f"{[(f, parsed, matched) for f, parsed, matched in loose_mismatches[:10]]}"
        )

    if unmatched:
        preview = unmatched[:10]
        raise ValueError(
            f"{len(unmatched)} image(s) could not be matched to a row in the "
            f"ground-truth excel (filename, lookup_key shown below). First few:\n{preview}\n"
            f"Check GROUND_TRUTH_MODE/ID_COLUMN in config.py and parse_original_id() in dataset.py."
        )

    table = pd.DataFrame(rows)
    n_groups = table["group_id"].nunique()
    print(f"Loaded {len(table)} images belonging to {n_groups} original samples.")
    if n_groups != config.N_ORIGINAL_IMAGES:
        print(
            f"WARNING: expected {config.N_ORIGINAL_IMAGES} original samples "
            f"but found {n_groups} distinct group ids. Double-check parse_original_id() "
            "-- if this number is wrong, your CV splits will be wrong too."
        )
    return table


def rgb_to_hsv_tensor(img: Image.Image, width: int, height: int) -> torch.Tensor:
    """
    PIL image -> resized HSV tensor scaled to [0, 1], shape (3, H, W).

    width/height should match the aspect ratio of your source photos
    (default config assumes 1280x960 / 4:3) so this is a plain downscale
    rather than a distorting squash.
    """
    img = img.convert("RGB").resize((width, height), Image.BILINEAR)
    hsv = img.convert("HSV")
    arr = np.asarray(hsv, dtype=np.float32) / 255.0  # PIL HSV channels are all 0-255
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3, H, W)
    return tensor


class SoilHSVDataset(Dataset):
    """
    file_table : subset of the DataFrame from build_file_table() (e.g. one fold's
                 train or val rows).
    mean/std   : per-channel HSV normalization stats. MUST be computed from the
                 TRAINING split only (see utils.compute_hsv_stats) and passed in
                 explicitly, so validation/test images never influence their own
                 normalization statistics.
    augment    : light extra augmentation (horizontal flip only). Off by default
                 since the 650 images are already augmented copies of the 36
                 originals -- more aggressive augmentation here would just be
                 augmenting augmentations.
    """

    def __init__(self, file_table: pd.DataFrame, mean, std, augment: bool = False):
        self.table = file_table.reset_index(drop=True)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.augment = augment

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        row = self.table.iloc[idx]
        img = Image.open(row["filepath"])
        tensor = rgb_to_hsv_tensor(img, config.IMAGE_WIDTH, config.IMAGE_HEIGHT)

        if self.augment and torch.rand(1).item() < 0.5:
            tensor = torch.flip(tensor, dims=[2])  # horizontal flip

        tensor = (tensor - self.mean) / (self.std + 1e-8)

        target = torch.tensor(
            row[config.TARGET_COLUMNS].values.astype(np.float32)
        )
        return tensor, target