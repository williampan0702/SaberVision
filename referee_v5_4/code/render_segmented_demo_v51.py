from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def at_time(rows: list[dict[str, str]], timestamp: float) -> dict[str, str] | None:
    for row in rows:
        if float(row["start_sec"]) <= timestamp <= float(row["end_sec"]):
            return row
    return None


def latest_started(finals: list[dict[str, str]], reviews: list[dict[str, str]], timestamp: float):
    candidates = []
    for kind, rows in (("FINAL", finals), ("REVIEW", reviews)):
        for row in rows:
            start = float(row["start_sec"])
            if start <= timestamp:
                candidates.append((start, kind, row))
    return max(candidates, default=None, key=lambda item: item[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a V5.4 segmented route/event demo.")
    parser.add_argument("input", type=Path)
    parser.add_argument("route_timeline", type=Path)
    parser.add_argument("final_events", type=Path)
    parser.add_argument("review_events", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=-1.0)
    args = parser.parse_args()

    routes = read_csv(args.route_timeline)
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
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = args.start + index / source_fps
        index += 1
        if args.end >= 0 and timestamp > args.end:
            break
        if (index - 1) % stride:
            continue
        route = at_time(routes, timestamp)
        final = at_time(finals, timestamp)
        review = at_time(reviews, timestamp)
        cv2.rectangle(frame, (0, 0), (width, 86), (0, 0, 0), -1)
        route_name = route.get("route", "unknown") if route else "unknown"
        cv2.putText(frame, f"V5.4 SEGMENT ROUTE: {route_name}  t={timestamp:.2f}s", (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 230, 255), 2, cv2.LINE_AA)
        if final:
            message = f"FINAL: {final['state'].upper()}"
            colour = (0, 255, 0)
        elif review:
            message = f"REVIEW ONLY: {review['state'].upper()}"
            colour = (0, 215, 255)
        else:
            previous = latest_started(finals, reviews, timestamp)
            if previous:
                started, kind, row = previous
                message = f"CURRENT LIGHT: NONE   LAST {kind}: {row['state'].upper()} @ {started:.2f}s"
            else:
                message = "CURRENT LIGHT: NONE   LAST TOUCH: NONE"
            colour = (190, 190, 190)
        cv2.putText(frame, message, (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.72, colour, 2, cv2.LINE_AA)
        writer.write(frame)
    cap.release()
    writer.release()
    print(args.output)


if __name__ == "__main__":
    main()
