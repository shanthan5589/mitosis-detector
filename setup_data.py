"""
Create metadata CSV files from dataset SQLite databases.

Usage:
    python setup_data.py --split train
    python setup_data.py --split test
    python setup_data.py --split train --db-path "D:/path/to/file.sqlite"
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from config import TRAIN_DATA_DIR, TEST_DATA_DIR


REQUIRED_TABLES = ["Slides", "Annotations", "Annotations_coordinates"]


def resolve_db_path(data_dir: Path, user_db_path: str | None) -> Path:
    if user_db_path:
        db_path = Path(user_db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {db_path}")
        return db_path

    sqlite_files = sorted(data_dir.glob("*.sqlite"))
    if len(sqlite_files) == 1:
        return sqlite_files[0]
    if len(sqlite_files) == 0:
        raise FileNotFoundError(
            f"No .sqlite file found in {data_dir}. "
            "Place the database there or pass --db-path."
        )
    raise RuntimeError(
        f"Multiple .sqlite files found in {data_dir}. "
        "Pass --db-path to choose one explicitly."
    )


def export_tables(db_path: Path, meta_dir: Path) -> None:
    meta_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table';", conn
        )["name"].tolist()
        table_set = set(tables)

        missing = [t for t in REQUIRED_TABLES if t not in table_set]
        if missing:
            raise RuntimeError(
                "Required table(s) missing in database: "
                + ", ".join(missing)
            )

        for table_name in REQUIRED_TABLES:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            out_path = meta_dir / f"{table_name}.csv"
            df.to_csv(out_path, index=False)
            print(f"Saved {out_path} ({len(df)} rows)")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument(
        "--db-path",
        default=None,
        help="Optional absolute path to SQLite database. If omitted, auto-detects a single *.sqlite file in the selected data directory.",
    )
    args = parser.parse_args()

    data_dir = TRAIN_DATA_DIR if args.split == "train" else TEST_DATA_DIR
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_dir}. "
            "Set paths in config.py first."
        )

    db_path = resolve_db_path(data_dir, args.db_path)
    meta_dir = data_dir / "meta_data"

    print(f"Split: {args.split}")
    print(f"Database: {db_path}")
    print(f"Output: {meta_dir}\n")
    export_tables(db_path, meta_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
