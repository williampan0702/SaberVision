from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from validate_dual_scoreboard_lamp_events_v47 import read_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Render chronological frame sheets around detected score changes.")
    parser.add_argument("input", type=Path)
    parser.add_argument("score_changes", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--before", type=float, default=5.0)
    parser.add_argument("--after", type=float, default=0.5)
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument("--events-per-sheet", type=int, default=3)
    parser.add_argument(
        "--indices",
        type=str,
        default="",
        help="Optional comma-separated 1-based event indices to render (for example: 3,9,17,18).",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    all_changes = read_csv(args.score_changes)
    indexed_changes = list(enumerate(all_changes, start=1))
    if args.indices:
        requested = {int(value.strip()) for value in args.indices.split(",") if value.strip()}
        indexed_changes = [(index, change) for index, change in indexed_changes if index in requested]
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    offsets = []
    offset = -args.before
    while offset <= args.after + 1e-6:
        offsets.append(offset)
        offset += args.step
    columns = 6
    tile_width, tile_height = 640, 360
    event_rows = math.ceil(len(offsets) / columns)

    for sheet_start in range(0, len(indexed_changes), args.events_per_sheet):
        subset = indexed_changes[sheet_start : sheet_start + args.events_per_sheet]
        canvas = np.zeros((len(subset) * event_rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
        for event_offset, (change_index, change) in enumerate(subset):
            change_time = float(change["time_sec"])
            side = change["side"]
            for frame_index, relative in enumerate(offsets):
                timestamp = max(0.0, change_time + relative)
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                ok, frame = cap.read()
                if not ok:
                    continue
                height, width = frame.shape[:2]
                crop_x1 = round(width * 0.125)
                crop_x2 = round(width * 0.875)
                crop_y1 = round(height * 0.08)
                crop_y2 = round(height * 0.83)
                crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                colour = (0, 0, 255) if side == "left" else (0, 255, 0)
                cv2.rectangle(crop, (0, 0), (crop.shape[1], 44), (0, 0, 0), -1)
                cv2.putText(
                    crop,
                    f"#{change_index:02d} {side.upper()} change={change_time:.1f}  t={timestamp:.1f} ({relative:+.1f})",
                    (8, 29),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    colour,
                    2,
                    cv2.LINE_AA,
                )
                tile = cv2.resize(crop, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
                local_row, column = divmod(frame_index, columns)
                row = event_offset * event_rows + local_row
                canvas[
                    row * tile_height : (row + 1) * tile_height,
                    column * tile_width : (column + 1) * tile_width,
                ] = tile
        first_index = subset[0][0]
        last_index = subset[-1][0]
        cv2.imwrite(str(args.output / f"score_windows_{first_index:02d}_{last_index:02d}.jpg"), canvas)
    cap.release()
    print(f"rendered {len(indexed_changes)} score-change windows", flush=True)


if __name__ == "__main__":
    main()
