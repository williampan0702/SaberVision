from __future__ import annotations

import argparse
import bisect
import csv
from pathlib import Path

import cv2


def load_track(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("valid", "1") in {"0", "false", "False"} or not row.get("x1"):
                continue
            rows.append(
                (
                    float(row["time_sec"]),
                    tuple(int(float(row[key])) for key in ("x1", "y1", "x2", "y2")),
                    row.get("source", "device_track"),
                )
            )
    rows.sort()
    return rows, [row[0] for row in rows]


def load_events(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {
                "state": row["state"].upper(),
                "start": float(row["start_sec"]),
                "end": float(row["end_sec"]),
                "sources": row.get("sources", ""),
            }
            for row in csv.DictReader(handle)
        ]


def nearest_track(track, times, timestamp: float, max_gap: float = 0.32):
    if not track:
        return None
    index = bisect.bisect_left(times, timestamp)
    candidates = []
    if index < len(track):
        candidates.append(track[index])
    if index:
        candidates.append(track[index - 1])
    nearest = min(candidates, key=lambda row: abs(row[0] - timestamp))
    return nearest if abs(nearest[0] - timestamp) <= max_gap else None


def expand(box, width: int, height: int, scale: float = 1.25):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    half_w, half_h = (x2 - x1) * scale * 0.5, (y2 - y1) * scale * 0.5
    return (
        max(0, round(cx - half_w)),
        max(0, round(cy - half_h)),
        min(width, round(cx + half_w)),
        min(height, round(cy + half_h)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a fused v4.7 locator/lamp demonstration.")
    parser.add_argument("input", type=Path)
    parser.add_argument("track", type=Path)
    parser.add_argument("events", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()

    track, track_times = load_track(args.track)
    events = load_events(args.events)
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, round(source_fps / min(source_fps, args.fps)))
    output_fps = source_fps / stride
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height)
    )
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)
    read_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = args.start + read_index / source_fps
        read_index += 1
        if timestamp > args.end:
            break
        if (read_index - 1) % stride:
            continue
        device = nearest_track(track, track_times, timestamp)
        event_time = round(timestamp, 4)
        event = next((item for item in events if item["start"] <= event_time <= item["end"]), None)
        state = event["state"] if event else "NONE"
        if device is not None:
            _track_time, box, source = device
            search = expand(box, width, height)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
            cv2.rectangle(frame, (search[0], search[1]), (search[2], search[3]), (0, 255, 255), 2)
            locator = source
        else:
            locator = "missing; edge fallback active" if event else "missing"
        sources = event["sources"] if event else "none"
        state_color = (0, 255, 255) if state == "NONE" else (0, 255, 0)
        cv2.rectangle(frame, (0, 0), (1060, 78), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"V47 FINAL: {state}  t={timestamp:.2f}s  sources={sources}",
            (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.67, state_color, 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"LOCATOR: {locator}",
            (8, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA,
        )
        writer.write(frame)
    cap.release()
    writer.release()


if __name__ == "__main__":
    main()
