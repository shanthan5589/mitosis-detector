from pathlib import Path

# Change this to your local dataset path
BASE_DIR = Path("D:/Mitosis WSI CCMCT")

# Directories
TRAINING_DATA_DIR = BASE_DIR / "training_data"
META_DATA_DIR = BASE_DIR / "meta_data"
MITOTIC_DIR = BASE_DIR / "mitotic"
NON_MITOTIC_DIR = BASE_DIR / "non_mitotic"

# Files
CLEAN_PATHS_FILE = BASE_DIR / "clean_paths.txt"

# SQLite database — included in the Kaggle download
DB_PATH = TRAINING_DATA_DIR / "MITOS_WSI_CCMCT_ODAEL_train_dcm.sqlite"

# Patch settings
PATCH_SIZE = 64
