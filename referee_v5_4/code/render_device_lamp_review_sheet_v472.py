from __future__ import annotations

import argparse
import bisect
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render zoomed physical-device lamp candidates for review.")
    parser.add_argument("input", type=Path)
    parser.add_argument("track", type=Path)
    parser.add_argument("events", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()

    events = load_csv(args.events)
    track = load_csv(args.track)
    track_times = [float(row["time_sec"]) for row in track]
    tile_width, tile_height = 600, 400
    sheet_rows = max(1, math.ceil(len(events) / args.columns))
    canvas = np.zeros((sheet_rows * tile_height, args.columns * tile_width, 3), dtype=np.uint8)
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for index, event in enumerate(events):
        timestamp = float(event.get("peak_time") or event["start_sec"])
        track_index = bisect.bisect_left(track_times, timestamp)
        candidates = []
        if track_index < len(track):
            candidates.append(track[track_index])
        if track_index:
            candidates.append(track[track_index - 1])
        if not candidates:
            continue
        device = min(candidates, key=lambda row: abs(float(row["time_sec"]) - timestamp))
        x1, y1, x2, y2 = (float(device[key]) for key in ("x1", "y1", "x2", "y2"))
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        crop_width = max(320.0, (x2 - x1) * 2.7)
        crop_height = max(230.0, (y2 - y1) * 2.7)
        left = max(0, min(frame_width - round(crop_width), round(cx - crop_width * 0.5)))
        top = max(0, min(frame_height - round(crop_height), round(cy - crop_height * 0.58)))
        right = min(frame_width, left + round(crop_width))
        bottom = min(frame_height, top + round(crop_height))
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        crop = frame[top:bottom, left:right].copy()
        colour = (0, 255, 255) if event["state"] == "double" else (0, 0, 255) if event["state"] == "left" else (0, 255, 0)
        cv2.rectangle(crop, (0, 0), (crop.shape[1], 38), (0, 0, 0), -1)
        cv2.putText(
            crop,
            f"#{index + 1:02d} {event['state'].upper()} t={timestamp:.2f} peak={event['peak_score']}",
            (8, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            colour,
            2,
            cv2.LINE_AA,
        )
        tile = cv2.resize(crop, (tile_width, tile_height), interpolation=cv2.INTER_CUBIC)
        row, column = divmod(index, args.columns)
        canvas[row * tile_height : (row + 1) * tile_height, column * tile_width : (column + 1) * tile_width] = tile

    cap.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), canvas)
    print(f"rendered {len(events)} review events", flush=True)


if __name__ == "__main__":
    main()
