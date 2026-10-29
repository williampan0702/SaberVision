from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an evenly sampled event contact sheet.")
    parser.add_argument("input", type=Path)
    parser.add_argument("events", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--columns", type=int, default=5)
    args = parser.parse_args()
    with args.events.open("r", encoding="utf-8-sig", newline="") as handle:
        events = list(csv.DictReader(handle))
    if not events:
        raise RuntimeError("No events")
    count = min(args.samples, len(events))
    indices = sorted({round(index * (len(events) - 1) / max(1, count - 1)) for index in range(count)})
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    tiles = []
    for index in indices:
        event = events[index]
        start = float(event["start_sec"])
        end = float(event["end_sec"])
        timestamp = (start + end) * 0.5
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
        cv2.rectangle(frame, (0, 0), (480, 42), (0, 0, 0), -1)
        label = f"#{event.get('event_id', index + 1)} {event['state'].upper()} t={timestamp:.1f}s"
        cv2.putText(frame, label, (7, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(frame)
    cap.release()
    blank = np.zeros((270, 480, 3), dtype=np.uint8)
    rows = math.ceil(len(tiles) / args.columns)
    while len(tiles) < rows * args.columns:
        tiles.append(blank.copy())
    sheet = np.vstack([
        np.hstack(tiles[row * args.columns:(row + 1) * args.columns])
        for row in range(rows)
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet):
        raise RuntimeError(f"Cannot write {args.output}")
    print(f"events={len(events)} sampled={len(indices)} output={args.output}")


if __name__ == "__main__":
    main()
