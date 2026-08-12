"""
Central configuration for the soil heavy-metal regression pipeline.

>>> THE TWO THINGS YOU MUST EDIT FOR YOUR DATA <<<
1. IMAGE_DIR / GROUND_TRUTH_XLSX / ID_COLUMN below.
2. `parse_original_id()` in dataset.py -- tells the code which of the
   36 original soil-sample photos each of the ~650 augmented images came
   from. This is THE piece that prevents train/val/test contamination,
   so double check it matches your actual filenames before trusting results.
"""

import torch

# ---------------------------------------------------------------- paths ---
IMAGE_DIR = r"C:\Users\ssuhaib\Desktop\data\images"                      # folder containing all ~650 images
GROUND_TRUTH_XLSX = r"C:\Users\ssuhaib\Desktop\data\Data240226_augmented.xlsx"
SHEET_NAME = 0

# Confirmed from your actual ground-truth excel: it has 648 rows, ONE ROW
# PER IMAGE, and the ID column holds the exact filename stem of each image
# (e.g. "1Ah_m_x1_b" for "1Ah_m_x1_b.jpg"). Every augmented copy still gets
# its own explicit row (usually with the same values repeated across an
# original sample's copies, but looked up per-image, not assumed).
GROUND_TRUTH_MODE = "per_image"   # "per_image" (confirmed for your data) or "per_original"

# Column in the excel that identifies each row. For your file this is
# "Sample ID" and its values are full filename stems like "1Ah_m_x1_b".
ID_COLUMN = "Sample ID"
TARGET_COLUMNS = ["Cd", "Cu", "Ni", "Mn", "Fe", "Zn"]

# ---------------------------------------------------------- split sizes ---
N_ORIGINAL_IMAGES = 36
TEST_FRACTION_OF_GROUPS = 0.20   # ~7 of the 36 originals held out for the final test
FINAL_ES_FRACTION_OF_GROUPS = 0.15  # small internal val slice for early-stopping the final model
N_FOLDS = 5
RANDOM_SEED = 42

# For repeated_holdout.py: repeats the dev/test split across this many
# different seeds, training one final model per split, to see how much
# final test performance varies given only ~36 independent samples --
# a single fixed 80/20 split is a noisy estimate at this sample size.
N_REPEATED_HOLDOUTS = 10
REPEATED_HOLDOUT_BASE_SEED = 1000

# ------------------------------------------------------------ image/model -
# Native photos are 1280x960 (4:3). Resizing to a square would squash/distort
# them, so we resize to a target that keeps the same 4:3 ratio instead --
# since it's an exact ratio match, this is a pure downscale, no distortion
# and no cropping. Keep IMAGE_WIDTH:IMAGE_HEIGHT at 4:3 if you change these.
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 192
BACKBONE = "resnet18"     # "custom" (default, fewer params) or "resnet18" (transfer learning)
BATCH_SIZE = 16
EPOCHS = 60
PATIENCE = 10            # early-stopping patience, in epochs
LR = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")