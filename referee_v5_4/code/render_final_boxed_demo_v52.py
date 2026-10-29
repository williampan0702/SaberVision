from __future__ import annotations

import argparse
import bisect
import csv
from pathlib import Path

import cv2


REGION_COLOURS = {
    "left_panel": (255, 0, 255),
    "center_box": (255, 255, 0),
    "right_panel": (255, 0, 255),
    "device": (0, 255, 255),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_track(path: Path):
    output = []
    for row in read_csv(path):
        boxes = {}
        if "left_panel_valid" in row:
            for region in ("left_panel", "center_box", "right_panel"):
                if row.get(f"{region}_valid") != "1":
                    continue
                keys = (f"{region}_x1", f"{region}_y1", f"{region}_x2", f"{region}_y2")
                if all(row.get(key) not in (None, "") for key in keys):
                    boxes[region] = tuple(round(float(row[key])) for key in keys)
        elif all(row.get(key) not in (None, "") for key in ("x1", "y1", "x2", "y2")):
            boxes["device"] = tuple(round(float(row[key])) for key in ("x1", "y1", "x2", "y2"))
        if boxes:
            output.append((float(row["time_sec"]), boxes))
    output.sort(key=lambda item: item[0])
    return output, [item[0] for item in output]


def nearest_boxes(track, times, timestamp: float, maximum_gap: float = 0.35):
    index = bisect.bisect_left(times, timestamp)
    candidates = []
    if index < len(track):
        candidates.append(track[index])
    if index:
        candidates.append(track[index - 1])
    if not candidates:
        return {}
    nearest = min(candidates, key=lambda item: abs(item[0] - timestamp))
    return nearest[1] if abs(nearest[0] - timestamp) <= maximum_gap else {}


def active_event(rows, timestamp: float):
    for row in rows:
        if float(row["start_sec"]) - 0.08 <= timestamp <= float(row["end_sec"]) + 0.08:
            return row
    return None


def latest_event(finals, reviews, timestamp: float):
    candidates = []
    for kind, rows in (("FINAL", finals), ("REVIEW", reviews)):
        for row in rows:
            start = float(row["start_sec"])
            if start <= timestamp:
                candidates.append((start, kind, row))
    return max(candidates, default=None, key=lambda item: item[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Render final V5.4 boxes plus current and latched touch state.")
    parser.add_argument("input", type=Path)
    parser.add_argument("track", type=Path)
    parser.add_argument("final_events", type=Path)
    parser.add_argument("review_events", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--route", default="unknown")
    args = parser.parse_args()

    track, times = load_track(args.track)
    finals = read_csv(args.final_events)
    reviews = read_csv(args.review_events)
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, round(source_fps / min(source_fps, args.fps)))
    output_fps = source_fps / stride
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
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
        for name, box in nearest_boxes(track, times, timestamp).items():
            colour = REGION_COLOURS[name]
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, name, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1, cv2.LINE_AA)

        current_final = active_event(finals, timestamp)
        current_review = active_event(reviews, timestamp)
        latest = latest_event(finals, reviews, timestamp)
        if current_final:
            state_text = f"CURRENT FINAL: {current_final['state'].upper()}"
            state_colour = (0, 255, 0)
        elif current_review:
            state_text = f"CURRENT REVIEW: {current_review['state'].upper()}"
            state_colour = (0, 215, 255)
        elif latest:
            started, kind, row = latest
            state_text = f"CURRENT LIGHT: NONE   LAST {kind}: {row['state'].upper()} @ {started:.2f}s"
            state_colour = (210, 210, 210)
        else:
            state_text = "CURRENT LIGHT: NONE   LAST TOUCH: NONE"
            state_colour = (210, 210, 210)
        cv2.rectangle(frame, (0, 0), (width, 84), (0, 0, 0), -1)
        cv2.putText(frame, f"V5.4 ROUTE: {args.route}  t={timestamp:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, state_text, (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.68, state_colour, 2, cv2.LINE_AA)
        writer.write(frame)
    cap.release()
    writer.release()
    print(args.output)


if __name__ == "__main__":
    main()
