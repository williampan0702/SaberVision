from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import time
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO

from scan_physical_lamps_v43 import scene_cut, stable_tracks, update_tracks, write_events
from scan_physical_lamps_yolo_v43 import resolve_physical_state
from diagnose_temporal_onset_v46 import activation


DEFAULT_LAMP_MODEL = Path(
    r"D:\SaberVision_data\SaberVision_lamp_state_training_v1"
    r"\stage60_physical_lamp_yolo_v45b_checkpoint_selection\weights\last.pt"
)


def load_device_track(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    track = []
    for row in rows:
        if row.get("valid", "1") in {"0", "false", "False"} or not row.get("x1", ""):
            continue
        track.append(
            (
                float(row["time_sec"]),
                tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2")),
                float(row.get("confidence") or (1.0 if row.get("source") == "anchor_structure" else 0.75)),
            )
        )
    track.sort()
    return track, [item[0] for item in track]


def nearest_box(track, times, timestamp: float, max_gap: float):
    if not track:
        return None
    index = bisect.bisect_left(times, timestamp)
    candidates = []
    if index < len(track):
        candidates.append(track[index])
    if index > 0:
        candidates.append(track[index - 1])
    nearest = min(candidates, key=lambda item: abs(item[0] - timestamp))
    return nearest if abs(nearest[0] - timestamp) <= max_gap else None


def expand_box(box, width: int, height: int, expansion: float):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = (x2 - x1) * expansion / 2
    half_h = (y2 - y1) * expansion / 2
    return (
        max(0, round(cx - half_w)),
        max(0, round(cy - half_h)),
        min(width, round(cx + half_w)),
        min(height, round(cy + half_h)),
    )


def peak_candidate_y(event: dict, candidate_rows: list[dict]) -> float | None:
    label = "red_lamp" if event["state"] == "left" else "green_lamp"
    start = float(event["start_sec"]) - 0.08
    end = float(event["end_sec"]) + 0.08
    candidates = [
        row
        for row in candidate_rows
        if row["decision"] == "accepted"
        and row["label"] == label
        and start <= float(row["time_sec"]) <= end
        and row["candidate_score"]
    ]
    if not candidates:
        return None
    strongest = max(candidates, key=lambda row: float(row["candidate_score"]))
    return float(strongest["relative_center_y"])


def merge_opposite_onsets(
    events: list[dict],
    max_gap: float,
    candidate_rows: list[dict],
    max_vertical_gap: float,
) -> list[dict]:
    """Fuse red/green onsets from one scoring action when video timing is staggered."""
    merged: list[dict] = []
    index = 0
    while index < len(events):
        current = events[index]
        if index + 1 < len(events):
            following = events[index + 1]
            states = {current["state"], following["state"]}
            gap = float(following["start_sec"]) - float(current["end_sec"])
            if states == {"left", "right"} and 0.0 <= gap <= max_gap:
                start_sec = min(float(current["start_sec"]), float(following["start_sec"]))
                end_sec = max(float(current["end_sec"]), float(following["end_sec"]))
                current_y = peak_candidate_y(current, candidate_rows)
                following_y = peak_candidate_y(following, candidate_rows)
                if current_y is not None and following_y is not None:
                    state = (
                        "double"
                        if abs(current_y - following_y) <= max_vertical_gap
                        else current["state"] if current_y < following_y else following["state"]
                    )
                else:
                    state = "double"
                merged.append(
                    {
                        "state": state,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "duration_sec": end_sec - start_sec,
                        "peak_score": f"{max(float(current['peak_score']), float(following['peak_score'])):.5f}",
                    }
                )
                index += 2
                continue
        merged.append(current)
        index += 1
    return merged


def merge_nearby_event_fragments(events: list[dict], max_gap: float, max_span: float) -> list[dict]:
    """Collapse short state flicker belonging to one scoring-machine illumination."""
    merged: list[dict] = []
    for event in events:
        if not merged:
            merged.append(dict(event))
            continue
        previous = merged[-1]
        gap = float(event["start_sec"]) - float(previous["end_sec"])
        combined_end = max(float(previous["end_sec"]), float(event["end_sec"]))
        combined_span = combined_end - float(previous["start_sec"])
        compatible_states = (
            previous["state"] == event["state"]
            or "double" in {previous["state"], event["state"]}
        )
        if compatible_states and 0.0 <= gap <= max_gap and combined_span <= max_span:
            states = {previous["state"], event["state"]}
            if "double" in states or states == {"left", "right"}:
                previous["state"] = "double"
            previous["end_sec"] = combined_end
            previous["duration_sec"] = combined_span
            previous["peak_score"] = f"{max(float(previous['peak_score']), float(event['peak_score'])):.5f}"
        else:
            merged.append(dict(event))
    return merged


def write_merged_events(path: Path, events: list[dict]) -> None:
    fields = ["state", "start_sec", "end_sec", "duration_sec", "peak_score"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lamp YOLO only inside stable v4.6 device tracks.")
    parser.add_argument("input", type=Path)
    parser.add_argument("device_track", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lamp-model", type=Path, default=DEFAULT_LAMP_MODEL)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--analysis-fps", type=float, default=15.0)
    parser.add_argument("--track-max-gap", type=float, default=1.1)
    parser.add_argument("--device-expansion", type=float, default=1.25)
    parser.add_argument("--lamp-threshold", type=float, default=0.005)
    parser.add_argument("--event-threshold", type=float, default=0.075)
    parser.add_argument("--onset-offset", type=float, default=0.67)
    parser.add_argument("--onset-threshold", type=float, default=0.035)
    parser.add_argument("--onset-ratio", type=float, default=1.20)
    parser.add_argument("--max-relative-y", type=float, default=0.48)
    parser.add_argument("--max-relative-width", type=float, default=0.20)
    parser.add_argument("--max-relative-height", type=float, default=0.28)
    parser.add_argument("--opposite-onset-merge-gap", type=float, default=0.60)
    parser.add_argument("--max-opposite-vertical-gap", type=float, default=0.10)
    parser.add_argument("--fragment-merge-gap", type=float, default=0.60)
    parser.add_argument("--fragment-merge-max-span", type=float, default=3.0)
    parser.add_argument("--required-track-hits", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    device_track, device_times = load_device_track(args.device_track)
    model = YOLO(str(args.lamp_model))
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, round(source_fps / min(source_fps, args.analysis_fps)))
    output_fps = source_fps / stride
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    writer = cv2.VideoWriter(
        str(args.output / "device_gated_lamp_preview.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )
    rows, candidate_rows, tracks, previous_hist = [], [], [], None
    frame_history: deque = deque(maxlen=max(12, round(output_fps * 1.5)))
    read_index, analyzed_index = 0, 0
    started = time.perf_counter()
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
        cut, previous_hist = scene_cut(frame, previous_hist)
        if cut:
            tracks = []
        device_row = nearest_box(device_track, device_times, timestamp, args.track_max_gap)
        detections = []
        raw_candidates = 0
        roi = None
        if device_row is not None:
            _device_time, device_box, _device_confidence = device_row
            dx1, dy1, dx2, dy2 = device_box
            device_width = max(1.0, dx2 - dx1)
            device_height = max(1.0, dy2 - dy1)
            roi = expand_box(device_box, width, height, args.device_expansion)
            rx1, ry1, rx2, ry2 = roi
            if rx2 > rx1 and ry2 > ry1:
                # Keep the full-frame scale used during lamp-detector training,
                # then gate predicted centers by the stable device ROI.
                result = model.predict(
                    frame,
                    imgsz=960,
                    conf=args.lamp_threshold,
                    iou=0.50,
                    max_det=30,
                    device=0,
                    verbose=False,
                )[0]
                if result.boxes is not None:
                    for xyxy, confidence, class_id in zip(
                        result.boxes.xyxy.cpu().numpy(),
                        result.boxes.conf.cpu().numpy(),
                        result.boxes.cls.cpu().numpy(),
                    ):
                        x1, y1, x2, y2 = map(int, xyxy)
                        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                        if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                            continue
                        raw_candidates += 1
                        label = model.names[int(class_id)]
                        audit = {
                            "time_sec": f"{timestamp:.4f}",
                            "label": label,
                            "confidence": f"{float(confidence):.6f}",
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "relative_center_x": f"{(cx - dx1) / device_width:.6f}",
                            "relative_center_y": f"{(cy - dy1) / device_height:.6f}",
                            "relative_width": f"{(x2 - x1) / device_width:.6f}",
                            "relative_height": f"{(y2 - y1) / device_height:.6f}",
                            "current_score": "",
                            "baseline_score": "",
                            "onset": "",
                            "ratio": "",
                            "candidate_score": "",
                            "decision": "no_past_frame",
                        }
                        relative_center_y = (cy - dy1) / device_height
                        relative_width = (x2 - x1) / device_width
                        relative_height = (y2 - y1) / device_height
                        if (
                            relative_center_y > args.max_relative_y
                            or relative_width > args.max_relative_width
                            or relative_height > args.max_relative_height
                        ):
                            audit["decision"] = "rejected_geometry"
                            candidate_rows.append(audit)
                            continue
                        past_candidates = [
                            item for item in frame_history
                            if item[2] is not None and abs((timestamp - item[0]) - args.onset_offset) <= 0.16
                        ]
                        if not past_candidates:
                            candidate_rows.append(audit)
                            continue
                        past_time, past_frame, past_roi = min(
                            past_candidates,
                            key=lambda item: abs((timestamp - item[0]) - args.onset_offset),
                        )
                        prx1, pry1, prx2, pry2 = past_roi
                        rel_box = (
                            (x1 - rx1) / max(1, rx2 - rx1),
                            (y1 - ry1) / max(1, ry2 - ry1),
                            (x2 - rx1) / max(1, rx2 - rx1),
                            (y2 - ry1) / max(1, ry2 - ry1),
                        )
                        past_box = (
                            prx1 + rel_box[0] * (prx2 - prx1),
                            pry1 + rel_box[1] * (pry2 - pry1),
                            prx1 + rel_box[2] * (prx2 - prx1),
                            pry1 + rel_box[3] * (pry2 - pry1),
                        )
                        current_score = activation(frame, (x1, y1, x2, y2), label)["score"]
                        baseline = activation(past_frame, past_box, label)["score"]
                        onset = current_score - baseline
                        ratio = current_score / max(0.01, baseline)
                        audit.update(
                            {
                                "current_score": f"{current_score:.6f}",
                                "baseline_score": f"{baseline:.6f}",
                                "onset": f"{onset:.6f}",
                                "ratio": f"{ratio:.6f}",
                                "candidate_score": f"{math.sqrt(float(confidence) * max(0.0, onset)):.6f}",
                                "decision": "rejected_onset",
                            }
                        )
                        if onset < args.onset_threshold or ratio < args.onset_ratio:
                            candidate_rows.append(audit)
                            continue
                        audit["decision"] = "accepted"
                        candidate_rows.append(audit)
                        detections.append(
                            {
                                "color": "red_lamp" if label == "red_lamp" else "green_lamp",
                                "box": (x1, y1, x2, y2),
                                "score": math.sqrt(float(confidence) * max(0.0, onset)),
                            }
                        )
        tracks = update_tracks(tracks, detections, analyzed_index, width, height)
        stable = stable_tracks(tracks, required_hits=args.required_track_hits)
        red, green = resolve_physical_state(stable, width, height)
        peak = max((track.score for track in stable), default=0.0)
        rows.append(
            {
                "frame": analyzed_index,
                "time_sec": f"{timestamp:.4f}",
                "scene_cut": int(cut),
                "device_valid": int(device_row is not None),
                "raw_candidates": raw_candidates,
                "onset_candidates": len(detections),
                "red_tracks": len(red),
                "green_tracks": len(green),
                "red_on": int(bool(red)),
                "green_on": int(bool(green)),
                "peak_score": f"{peak:.5f}",
            }
        )
        # Preserve the unannotated frame for device-coordinate temporal comparison.
        frame_history.append((timestamp, frame.copy(), roi))
        if roi is not None:
            cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 255), 2)
        for track in red + green:
            x1, y1, x2, y2 = track.box
            color = (0, 0, 255) if track.color == "red_lamp" else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{track.color}:{track.score:.3f}", (x1, max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        state = "DOUBLE" if red and green else "LEFT" if red else "RIGHT" if green else "NONE"
        cv2.rectangle(frame, (0, 0), (650, 42), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"DEVICE-GATED: {state} device={int(device_row is not None)} t={timestamp:.2f}s",
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        analyzed_index += 1
    cap.release()
    writer.release()
    with (args.output / "device_gated_lamp_frames.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    if candidate_rows:
        with (args.output / "candidate_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            csv_writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
            csv_writer.writeheader()
            csv_writer.writerows(candidate_rows)
    event_path = args.output / "device_gated_lamp_events.csv"
    events = write_events(
        rows,
        event_path,
        minimum_sec=0.0,
        minimum_peak=0.0,
        maximum_sec=4.0,
    )
    events = merge_opposite_onsets(
        events,
        args.opposite_onset_merge_gap,
        candidate_rows,
        args.max_opposite_vertical_gap,
    )
    events = merge_nearby_event_fragments(
        events,
        args.fragment_merge_gap,
        args.fragment_merge_max_span,
    )
    events = [
        event
        for event in events
        if float(event["duration_sec"]) >= 0.12
        and float(event["peak_score"]) >= args.event_threshold
        and float(event["duration_sec"]) <= 4.0
    ]
    write_merged_events(event_path, events)
    summary = {
        "input": str(args.input),
        "device_track": str(args.device_track),
        "start": args.start,
        "end": args.end,
        "analyzed_frames": analyzed_index,
        "device_valid_frames": sum(int(row["device_valid"]) for row in rows),
        "elapsed_sec": time.perf_counter() - started,
        "events": events,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
