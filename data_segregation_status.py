"""
Monitor extraction progress: Prints patch counts every minute.

Usage:
    python data_segregation_status.py --split train
    python data_segregation_status.py --split test
"""

import argparse
import os
import time
from datetime import datetime
from config import TRAINING_DATA_DIR, TEST_DATA_DIR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True,
                        help="Which split to monitor: train or test")
    args = parser.parse_args()

    data_dir = TRAINING_DATA_DIR if args.split == "train" else TEST_DATA_DIR

    prev_non = 0
    prev_mit = 0
    history = []

    while True:
        non_count = len(os.listdir(data_dir / "non_mitotic"))
        mit_count = len(os.listdir(data_dir / "mitotic"))

        total = non_count + mit_count
        speed = (non_count - prev_non) + (mit_count - prev_mit)
        history.append(speed)

        average = int(sum(history) / len(history))
        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        print(f"[{current_time}] Non: {non_count}, Mit: {mit_count} | Total: {total} | Speed: {speed}/min | Average: {average}/min")

        if len(history) > 1000:
            history = history[-500:]

        prev_non = non_count
        prev_mit = mit_count
        time.sleep(60)


if __name__ == "__main__":
    main()
