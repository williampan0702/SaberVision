from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def line(frame, record: dict | None, color: tuple[int, int, int]) -> None:
    if not record or not record.get("guard") or not record.get("tip"):
        return
    if len(record["guard"]) < 2 or len(record["tip"]) < 2:
        return
    guard_values = [finite_float(value) for value in record["guard"][:2]]
    tip_values = [finite_float(value) for value in record["tip"][:2]]
    if any(value is None for value in guard_values + tip_values):
        return
    guard = tuple(int(round(float(value))) for value in guard_values)
    tip = tuple(int(round(float(value))) for value in tip_values)
    thickness = 4 if record.get("source") == "geometry" else 2
    cv2.line(frame, guard, tip, color, thickness, cv2.LINE_AA)
    cv2.circle(frame, guard, 7, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.drawMarker(frame, tip, color, cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA)


def put(frame, text: str, x: int, y: int, color=(255, 255, 255), scale=0.62, thickness=2):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def event_at(events: list[dict], time_sec: float) -> dict | None:
    valid = []
    for event in events:
        if not isinstance(event, dict):
            continue
        start = finite_float(event.get("start_sec"))
        end = finite_float(event.get("end_sec"))
        if start is not None and end is not None and end >= start:
            valid.append((start, end, event))
    active = [event for start, end, event in valid if start <= time_sec <= end]
    if active:
        return active[0]
    previous = [(start, event) for start, _end, event in valid if start <= time_sec]
    return max(previous, key=lambda item: item[0])[1] if previous else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Render unified SaberVision MVP evidence overlay.")
    parser.add_argument("--pose-video", type=Path, required=True)
    parser.add_argument("--blade-json", type=Path)
    parser.add_argument("--verdict-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    blade = read_json(args.blade_json) if args.blade_json else {}
    verdict = read_json(args.verdict_json)
    blade_frames = {}
    for item in blade.get("frames", []):
        if not isinstance(item, dict):
            continue
        frame_number = finite_float(item.get("frame"))
        if frame_number is not None and frame_number >= 0 and frame_number.is_integer():
            blade_frames[int(frame_number)] = item
    events = verdict.get("events", [])
    cap = cv2.VideoCapture(str(args.pose_video))
    if not cap.isOpened():
        raise RuntimeError(args.pose_video)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(args.output)

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        time_sec = frame_index / fps
        blades = blade_frames.get(frame_index) or {}
        line(frame, blades.get("left"), (255, 170, 30))
        line(frame, blades.get("right"), (30, 140, 255))

        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (min(width - 8, 720), 150), (8, 8, 8), -1)
        frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
        put(frame, "SaberVision MVP | athlete + pose + blade + scoring device", 22, 36, (70, 230, 255), 0.66)
        put(frame, f"time {time_sec:05.2f}s", 22, 65, (220, 220, 220), 0.56, 1)

        event = event_at(events, time_sec)
        if event is None:
            put(frame, "DEVICE: waiting for lamp event", 22, 96, (180, 180, 180))
            put(frame, "VERDICT: pending", 22, 128, (180, 180, 180), 0.72)
        else:
            touch = str(event.get("touch_state") or "REVIEW").upper()
            decision = event.get("verdict") or {}
            awarded = str(decision.get("awarded_side") or "REVIEW").upper()
            status = str(decision.get("status") or "unknown")
            lamp_color = (80, 255, 80) if touch == "LEFT" else (80, 80, 255) if touch == "RIGHT" else (0, 215, 255)
            put(frame, f"DEVICE: {touch} | {event['start_sec']:.2f}-{event['end_sec']:.2f}s", 22, 96, lamp_color)
            put(frame, f"VERDICT: {awarded} | {status}", 22, 128, lamp_color, 0.72)
        writer.write(frame)
        frame_index += 1
    cap.release()
    writer.release()
    print(json.dumps({"output": str(args.output), "frames": frame_index, "events": len(events)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
