from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from scan_physical_lamps_v43 import Track, center, scene_cut, stable_tracks, update_tracks, write_events


DEFAULT_MODEL = Path(r"D:\SaberVision_data\SaberVision_lamp_state_training_v1\stage52_physical_lamp_yolo_v43\weights\best.pt")


def resolve_physical_state(stable, width: int, height: int):
    """Keep a double only when red is physically left of a nearby green lamp."""
    red = [track for track in stable if track.color == "red_lamp"]
    green = [track for track in stable if track.color == "green_lamp"]
    pairs = []
    for red_track in red:
        red_x, red_y = center(red_track.box)
        for green_track in green:
            green_x, green_y = center(green_track.box)
            dx = (green_x - red_x) / width
            dy = abs(green_y - red_y) / height
            if 0.012 <= dx <= 0.14 and dy <= 0.055:
                pairs.append((min(red_track.score, green_track.score), red_track.score + green_track.score,
                              red_track, green_track))
    if pairs:
        _minimum_score, _total_score, red_track, green_track = max(pairs, key=lambda item: (item[0], item[1]))
        return [red_track], [green_track]
    if red and green:
        strongest = max(red + green, key=lambda track: track.score)
        return ([strongest], []) if strongest.color == "red_lamp" else ([], [strongest])
    return red, green


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-frame YOLO physical lamp scan experiment.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--analysis-fps", type=float, default=15.0)
    parser.add_argument("--threshold", type=float, default=0.025)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / source_fps
    start, end = max(0.0, args.start), min(duration, args.end if args.end is not None else duration)
    stride = max(1, round(source_fps / min(source_fps, args.analysis_fps)))
    output_fps = source_fps / stride
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    writer = cv2.VideoWriter(str(args.output / "physical_lamp_yolo_preview.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    rows, tracks, previous_hist = [], [], None
    read_index, analyzed_index = 0, 0
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
        result = model.predict(frame, imgsz=960, conf=args.threshold, iou=0.55, max_det=30,
                               device=0, verbose=False)[0]
        detections = []
        for xyxy, confidence, class_id in zip(result.boxes.xyxy.cpu().numpy(),
                                              result.boxes.conf.cpu().numpy(),
                                              result.boxes.cls.cpu().numpy()):
            label = names[int(class_id)]
            color = "red_lamp" if label == "red_lamp" else "green_lamp"
            detections.append({"color": color, "box": tuple(map(int, xyxy)), "score": float(confidence)})
        tracks = update_tracks(tracks, detections, analyzed_index, width, height)
        stable = stable_tracks(tracks)
        red, green = resolve_physical_state(stable, width, height)
        peak = max((track.score for track in stable), default=0.0)
        rows.append({"frame": analyzed_index, "time_sec": f"{timestamp:.4f}", "scene_cut": int(cut),
                     "raw_candidates": len(detections), "red_tracks": len(red), "green_tracks": len(green),
                     "red_on": int(bool(red)), "green_on": int(bool(green)), "peak_score": f"{peak:.5f}"})
        for track in red + green:
            x1, y1, x2, y2 = track.box
            color = (0, 0, 255) if track.color == "red_lamp" else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{track.color}:{track.score:.2f}", (x1, max(18, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        state = "DOUBLE" if red and green else "LEFT" if red else "RIGHT" if green else "NONE"
        cv2.rectangle(frame, (0, 0), (560, 42), (0, 0, 0), -1)
        cv2.putText(frame, f"YOLO PHYSICAL LAMPS: {state}  t={timestamp:.2f}s", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        analyzed_index += 1
    cap.release()
    writer.release()
    with (args.output / "physical_lamp_yolo_frames.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    events = write_events(rows, args.output / "physical_lamp_yolo_events.csv")
    summary = {"input": str(args.input), "source_fps": source_fps, "analysis_fps": output_fps,
               "stride": stride, "start_sec": start, "end_sec": end, "analyzed_frames": analyzed_index,
               "elapsed_sec": time.perf_counter() - started, "threshold": args.threshold, "events": events}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
