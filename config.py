from pathlib import Path

# Change this to your local dataset path
BASE_DIR = Path("D:/Mitosis WSI CCMCT")

# Directories
TRAINING_DATA_DIR = BASE_DIR / "testing_data"
META_DATA_DIR = TRAINING_DATA_DIR / "meta_data"
MITOTIC_DIR = TRAINING_DATA_DIR/ "mitotic"
NON_MITOTIC_DIR = TRAINING_DATA_DIR / "non_mitotic"

# Files
CLEAN_PATHS_FILE = TRAINING_DATA_DIR / "clean_paths.txt"

# SQLite database — included in the Kaggle download
DB_PATH = TRAINING_DATA_DIR / "MITOS_WSI_CCMCT_ODAEL_test.dcm.sqlite"

# Patch settings
PATCH_SIZE = 64
