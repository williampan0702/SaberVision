from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from scan_physical_lamps_v43 import center, scene_cut, stable_tracks, update_tracks, write_events


DEFAULT_MODEL = Path(
    r"D:\SaberVision_data\SaberVision_referee_recognition_v4_6_experimental"
    r"\models\physical_lamp_locator_v45b.pt"
)


def color_support(frame, box, color: str) -> float:
    """Return the fraction of a detection that visibly carries its lamp colour."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    visible = (saturation >= 65) & (value >= 105)
    if color == "red_lamp":
        coloured = visible & ((hue <= 16) | (hue >= 164))
    else:
        coloured = visible & (hue >= 34) & (hue <= 100)
    return float(coloured.mean())


def supported_state(frame, stable, width: int, height: int, strong_threshold: float):
    red = [track for track in stable if track.color == "red_lamp"]
    green = [track for track in stable if track.color == "green_lamp"]
    pairs = []
    for red_track in red:
        red_x, red_y = center(red_track.box)
        for green_track in green:
            green_x, green_y = center(green_track.box)
            dx = (green_x - red_x) / width
            dy = abs(green_y - red_y) / height
            red_supported = red_track.score >= strong_threshold or color_support(
                frame, red_track.box, red_track.color
            ) >= 0.08
            green_supported = green_track.score >= strong_threshold or color_support(
                frame, green_track.box, green_track.color
            ) >= 0.08
            if (
                0.012 <= dx <= 0.14
                and dy <= 0.055
                and max(red_track.score, green_track.score) >= strong_threshold
                and red_supported
                and green_supported
            ):
                pairs.append((red_track.score + green_track.score, red_track, green_track))
    if pairs:
        _score, red_track, green_track = max(pairs, key=lambda item: item[0])
        return [red_track], [green_track]
    strong_red = [track for track in red if track.score >= strong_threshold]
    strong_green = [track for track in green if track.score >= strong_threshold]
    if strong_red and strong_green:
        strongest = max(strong_red + strong_green, key=lambda track: track.score)
        return ([strongest], []) if strongest.color == "red_lamp" else ([], [strongest])
    return strong_red, strong_green


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-frame fallback: one strong lamp may support a nearby weak opposite lamp.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--analysis-fps", type=float, default=15.0)
    parser.add_argument("--weak-threshold", type=float, default=0.005)
    parser.add_argument("--strong-threshold", type=float, default=0.05)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / source_fps
    start = max(0.0, args.start)
    end = min(duration, args.end if args.end is not None else duration)
    stride = max(1, round(source_fps / min(source_fps, args.analysis_fps)))
    output_fps = source_fps / stride
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    writer = None if args.no_preview else cv2.VideoWriter(
        str(args.output / "edge_lamp_fallback_preview.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )
    rows = []
    tracks = []
    previous_hist = None
    read_index = 0
    analyzed_index = 0
    started = time.perf_counter()
    names = model.names
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = start + read_index / source_fps
        read_index += 1
        if timestamp > end:
            break
        if (read_index - 1) % stride:
            continue
        cut, previous_hist = scene_cut(frame, previous_hist)
        if cut:
            tracks = []
        result = model.predict(
            frame,
            imgsz=960,
            conf=args.weak_threshold,
            iou=0.55,
            max_det=30,
            device=0,
            verbose=False,
        )[0]
        detections = []
        for xyxy, confidence, class_id in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
        ):
            x1, y1, x2, y2 = tuple(map(int, xyxy))
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            if not (0 <= cx <= width and height * 0.10 <= cy <= height * 0.72):
                continue
            label = names[int(class_id)]
            detections.append(
                {
                    "color": "red_lamp" if label == "red_lamp" else "green_lamp",
                    "box": (x1, y1, x2, y2),
                    "score": float(confidence),
                }
            )
        tracks = update_tracks(tracks, detections, analyzed_index, width, height)
        stable = stable_tracks(tracks)
        red, green = supported_state(frame, stable, width, height, args.strong_threshold)
        peak = max((track.score for track in red + green), default=0.0)
        rows.append(
            {
                "frame": analyzed_index,
                "time_sec": f"{timestamp:.4f}",
                "scene_cut": int(cut),
                "raw_candidates": len(detections),
                "red_tracks": len(red),
                "green_tracks": len(green),
                "red_on": int(bool(red)),
                "green_on": int(bool(green)),
                "peak_score": f"{peak:.5f}",
            }
        )
        for track in red + green:
            x1, y1, x2, y2 = track.box
            color = (0, 0, 255) if track.color == "red_lamp" else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{track.color}:{track.score:.3f}",
                (x1, max(18, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        state = "DOUBLE" if red and green else "LEFT" if red else "RIGHT" if green else "NONE"
        cv2.rectangle(frame, (0, 0), (760, 42), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"V47 EDGE FALLBACK: {state} t={timestamp:.2f}s",
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if writer is not None:
            writer.write(frame)
        analyzed_index += 1
    cap.release()
    if writer is not None:
        writer.release()

    with (args.output / "edge_lamp_fallback_frames.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    events = write_events(rows, args.output / "edge_lamp_fallback_events.csv")
    summary = {
        "input": str(args.input),
        "start_sec": start,
        "end_sec": end,
        "analyzed_frames": analyzed_index,
        "elapsed_sec": time.perf_counter() - started,
        "weak_threshold": args.weak_threshold,
        "strong_threshold": args.strong_threshold,
        "events": events,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
