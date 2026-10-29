#!/usr/bin/env python3
"""Build before/after score-overlay strips for reviewed lamp events."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def frame_at(cap: cv2.VideoCapture, time_sec: float) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_sec) * 1000)
    ok, frame = cap.read()
    if not ok:
        return np.zeros((360, 640, 3), dtype=np.uint8)
    return frame


def score_tile(frame: np.ndarray, label: str) -> np.ndarray:
    height, width = frame.shape[:2]
    crop = frame[int(height * 0.60):int(height * 0.98), int(width * 0.02):int(width * 0.98)]
    image = cv2.resize(crop, (620, 230), interpolation=cv2.INTER_AREA)
    cv2.rectangle(image, (0, 0), (620, 32), (0, 0, 0), -1)
    cv2.putText(image, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-page", type=int, default=4)
    parser.add_argument("--venue", help="Optional exact venue/test_id filter for mixed event files")
    args = parser.parse_args()
    with args.events.open("r", encoding="utf-8-sig", newline="") as handle:
        events = [
            row
            for row in csv.DictReader(handle)
            if row.get("state") == "double"
            and (
                not args.venue
                or row.get("venue") == args.venue
                or row.get("test_id") == args.venue
            )
        ]
    if not events:
        raise SystemExit("No matching double-light events")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(args.video)
    strips: list[np.ndarray] = []
    for event_index, event in enumerate(events, start=1):
        start, end = float(event["start_sec"]), float(event["end_sec"])
        times = [start - 1.0, end + 0.4, end + 2.4, end + 4.4]
        event_id = event.get("event_id") or str(event_index)
        tiles = [score_tile(frame_at(cap, value), f"event {event_id} | t={value:.1f}s") for value in times]
        strips.append(np.hstack(tiles))
    cap.release()
    for page, begin in enumerate(range(0, len(strips), args.per_page), 1):
        selected = strips[begin:begin + args.per_page]
        while len(selected) < args.per_page:
            selected.append(np.zeros_like(strips[0]))
        image = np.vstack(selected)
        path = args.output_dir / f"score_review_{page:02d}.jpg"
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(path)
    print(f"events={len(events)} pages={math.ceil(len(events) / args.per_page)}")


if __name__ == "__main__":
    main()
