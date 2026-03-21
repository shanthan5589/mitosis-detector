import os
from pathlib import Path

import time
from datetime import datetime

base_dir = Path("D:\Mitosis WSI CCMCT")

prev_non = 0
prev_mit = 0

history = []

while True:

    non_count = len(os.listdir(base_dir / 'non_mitotic'))
    mit_count = len(os.listdir(base_dir / 'mitotic'))

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

    time.sleep(60)  # Check every minute