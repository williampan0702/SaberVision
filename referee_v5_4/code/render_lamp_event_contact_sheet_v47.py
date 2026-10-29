from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from validate_dual_scoreboard_lamp_events_v47 import read_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one visual midpoint for each detected lamp event.")
    parser.add_argument("input", type=Path)
    parser.add_argument("events", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--columns", type=int, default=5)
    args = parser.parse_args()

    events = read_csv(args.events)
    tile_width, tile_height = 640, 360
    rows = (len(events) + args.columns - 1) // args.columns
    canvas = np.zeros((rows * tile_height, args.columns * tile_width, 3), dtype=np.uint8)
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    for index, event in enumerate(events):
        start = float(event["start_sec"])
        end = float(event["end_sec"])
        timestamp = (start + end) * 0.5
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        height, width = frame.shape[:2]
        crop = frame[round(height * 0.06) : round(height * 0.84), round(width * 0.05) : round(width * 0.95)].copy()
        state = event["state"]
        colour = (0, 255, 255) if state == "double" else (0, 0, 255) if state == "left" else (0, 255, 0)
        cv2.rectangle(crop, (0, 0), (crop.shape[1], 46), (0, 0, 0), -1)
        cv2.putText(
            crop,
            f"#{index + 1:02d} {state.upper()} {start:.1f}-{end:.1f}s peak={event.get('peak_score', '')}",
            (8, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            colour,
            2,
            cv2.LINE_AA,
        )
        tile = cv2.resize(crop, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, args.columns)
        canvas[row * tile_height : (row + 1) * tile_height, column * tile_width : (column + 1) * tile_width] = tile
    cap.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), canvas)
    print(f"rendered {len(events)} events", flush=True)


if __name__ == "__main__":
    main()
