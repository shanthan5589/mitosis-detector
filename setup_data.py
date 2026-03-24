"""
Setup: Creates directory structure and extracts metadata tables from the
SQLite database as CSVs.

Usage:
    python setup_data.py --split train
    python setup_data.py --split test
"""

import argparse
import sqlite3
import pandas as pd
import os
from config import (TRAINING_DATA_DIR, META_DATA_DIR, MITOTIC_DIR, NON_MITOTIC_DIR, TRAIN_DB_PATH,
                    TEST_DATA_DIR, TEST_META_DATA_DIR, TEST_DB_PATH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True,
                        help="Which split to set up: train or test")
    args = parser.parse_args()

    if args.split == "train":
        data_dir = TRAINING_DATA_DIR
        meta_dir = META_DATA_DIR
        mitotic_dir = MITOTIC_DIR
        non_mitotic_dir = NON_MITOTIC_DIR
        db_path = TRAIN_DB_PATH
    else:
        data_dir = TEST_DATA_DIR
        meta_dir = TEST_META_DATA_DIR
        mitotic_dir = TEST_DATA_DIR / "mitotic"
        non_mitotic_dir = TEST_DATA_DIR / "non_mitotic"
        db_path = TEST_DB_PATH

    # Create directory structure
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(mitotic_dir, exist_ok=True)
    os.makedirs(non_mitotic_dir, exist_ok=True)

    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found at {db_path}. Place the .sqlite file inside {data_dir} first.")

    # Extract all tables from the database and save as CSVs
    conn = sqlite3.connect(str(db_path))

    tables = pd.read_sql_query('SELECT name FROM sqlite_master WHERE type="table";', conn)

    for table_name in tables["name"]:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        df.to_csv(meta_dir / f"{table_name}.csv", index=False)
        print(f"Saved {table_name}: {len(df)} rows")

    conn.close()


if __name__ == "__main__":
    main()
