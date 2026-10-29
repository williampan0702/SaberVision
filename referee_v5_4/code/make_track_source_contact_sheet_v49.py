from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def box(row: dict[str, str], prefix: str) -> tuple[int, int, int, int]:
    return tuple(round(float(row[f"{prefix}_{key}"])) for key in ("x1", "y1", "x2", "y2"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render evenly sampled frames from one track source.")
    parser.add_argument("input", type=Path)
    parser.add_argument("track", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", default="three_region_relaxed_anchor_v49")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--columns", type=int, default=5)
    args = parser.parse_args()
    rows = [row for row in read_csv(args.track) if row.get("source") == args.source]
    if not rows:
        raise ValueError(f"No rows for source {args.source}")
    indices = np.linspace(0, len(rows) - 1, min(args.samples, len(rows))).round().astype(int)
    selected = [rows[index] for index in indices]
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    tile_width, tile_height = 480, 300
    tiles = []
    for row in selected:
        timestamp = float(row["time_sec"])
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        for prefix, color in (
            ("left_panel", (255, 0, 255)),
            ("center_box", (255, 255, 0)),
            ("right_panel", (255, 0, 255)),
        ):
            x1, y1, x2, y2 = box(row, prefix)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 52), (0, 0, 0), -1)
        cv2.putText(frame, f"RECOVERED t={timestamp:.1f}s", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA))
    cap.release()
    rows_count = math.ceil(len(tiles) / args.columns)
    sheet = np.zeros((rows_count * tile_height, args.columns * tile_width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        y = (index // args.columns) * tile_height
        x = (index % args.columns) * tile_width
        sheet[y:y + tile_height, x:x + tile_width] = tile
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), sheet)
    print(f"source_rows={len(rows)} sampled={len(tiles)} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
