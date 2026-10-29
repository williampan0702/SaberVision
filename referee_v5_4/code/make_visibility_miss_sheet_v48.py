from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Contact sheet for loose-visible but strict-missing frames.")
    parser.add_argument("input", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--maximum-time", type=float, default=5791.0)
    args = parser.parse_args()
    with args.audit.open("r", encoding="utf-8-sig", newline="") as handle:
        misses = [
            row for row in csv.DictReader(handle)
            if row["loose_observable"] == "1"
            and row["strict_three_region"] == "0"
            and float(row["time_sec"]) <= args.maximum_time
        ]
    count = min(args.samples, len(misses))
    indices = sorted({round(index * (len(misses) - 1) / max(1, count - 1)) for index in range(count)})
    cap = cv2.VideoCapture(str(args.input))
    tiles = []
    for index in indices:
        timestamp = float(misses[index]["time_sec"])
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
        cv2.rectangle(frame, (0, 0), (480, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"MISS #{index + 1} t={timestamp:.1f}s", (7, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(frame)
    cap.release()
    columns = 5
    rows = math.ceil(len(tiles) / columns)
    blank = np.zeros((270, 480, 3), dtype=np.uint8)
    while len(tiles) < rows * columns:
        tiles.append(blank.copy())
    sheet = np.vstack([np.hstack(tiles[row * columns:(row + 1) * columns]) for row in range(rows)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), sheet)
    print(f"misses={len(misses)} sampled={len(indices)} output={args.output}")


if __name__ == "__main__":
    main()
