"""
Setup: Creates directory structure and extracts metadata tables from the
SQLite database as CSVs.

Usage:
    python setup_data.py
"""

import sqlite3
import pandas as pd
import os
from config import BASE_DIR, TRAINING_DATA_DIR, META_DATA_DIR, MITOTIC_DIR, NON_MITOTIC_DIR, DB_PATH


def main():
    # Create directory structure
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    os.makedirs(META_DATA_DIR, exist_ok=True)
    os.makedirs(MITOTIC_DIR, exist_ok=True)
    os.makedirs(NON_MITOTIC_DIR, exist_ok=True)

    if not DB_PATH.exists():
        raise FileNotFoundError(f"SQLite database not found at {DB_PATH}")

    # Extract all tables from the database and save as CSVs
    conn = sqlite3.connect(str(DB_PATH))

    tables = pd.read_sql_query('SELECT name FROM sqlite_master WHERE type="table";', conn)

    for table_name in tables["name"]:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        df.to_csv(META_DATA_DIR / f"{table_name}.csv", index=False)
        print(f"Saved {table_name}: {len(df)} rows")

    conn.close()


if __name__ == "__main__":
    main()
