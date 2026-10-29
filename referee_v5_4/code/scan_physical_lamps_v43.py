from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models

from build_lit_lamp_candidate_dataset_v43 import candidate_crop, components


DEFAULT_MODEL = Path(r"D:\Codex输出\SaberVision_lamp_state_training_v1\stage45_lit_lamp_classifier_v43\lit_lamp_resnet18_v43.pt")


@dataclass
class Track:
    color: str
    box: tuple[int, int, int, int]
    score: float
    history: deque
    last_frame: int


def center(box):
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def scene_cut(frame: np.ndarray, previous: np.ndarray | None) -> tuple[bool, np.ndarray]:
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    if previous is None:
        return False, hist
    distance = float(cv2.compareHist(previous, hist, cv2.HISTCMP_BHATTACHARYYA))
    return distance >= 0.46, hist


def preprocess(crops: list[np.ndarray], device: torch.device) -> torch.Tensor:
    array = np.stack([cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in crops]).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).to(device, non_blocking=True)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint["classes"]


def bright_core_candidates(frame: np.ndarray) -> list[tuple[str, tuple[int, int, int, int, int]]]:
    """Split white-hot lamp centers whose colored halos merge into one component."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    bright = ((value >= 228) & (saturation <= 105)).astype(np.uint8) * 255
    count, _, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    h, w = frame.shape[:2]
    result = []
    for x, y, bw, bh, area in stats[1:]:
        if area < max(8, round(w * h * 0.000009)) or bw < 3 or bh < 3:
            continue
        if bw > w * 0.12 or bh > h * 0.10:
            continue
        pad = max(8, round(max(bw, bh) * 1.3))
        x1, y1, x2, y2 = max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad)
        local_h, local_s, local_v = hue[y1:y2, x1:x2], saturation[y1:y2, x1:x2], value[y1:y2, x1:x2]
        red = float(np.mean((local_s >= 70) & (local_v >= 120) & ((local_h <= 14) | (local_h >= 166))))
        green = float(np.mean((local_s >= 55) & (local_v >= 115) & (local_h >= 34) & (local_h <= 100)))
        if max(red, green) < 0.035 or max(red, green) < min(red, green) * 1.18:
            continue
        color = "red_lamp" if red > green else "green_lamp"
        result.append((color, (int(x), int(y), int(x + bw), int(y + bh), int(area))))
    return result


def near_duplicate(box, existing) -> bool:
    cx, cy = center(box)
    for other in existing:
        ox, oy = center(other)
        scale = max(box[2] - box[0], box[3] - box[1], other[2] - other[0], other[3] - other[1], 6)
        if math.hypot(cx - ox, cy - oy) <= scale * 0.65:
            return True
    return False


def classify_candidates(frame, model, classes, device, threshold, max_per_color=28):
    proposals, crops = [], []
    boxes_by_color = {"red_lamp": [], "green_lamp": []}
    for color in ("red_lamp", "green_lamp"):
        found = sorted(components(frame, color), key=lambda item: item[4], reverse=True)[:max_per_color]
        for box in found:
            proposals.append((color, box))
            crops.append(candidate_crop(frame, box))
            boxes_by_color[color].append(box[:4])
    for color, box in bright_core_candidates(frame):
        if near_duplicate(box[:4], boxes_by_color[color]):
            continue
        proposals.append((color, box))
        crops.append(candidate_crop(frame, box))
        boxes_by_color[color].append(box[:4])
    if not crops:
        return []
    background_index = classes.index("background")
    detections = []
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        probabilities = model(preprocess(crops, device)).softmax(1).float().cpu().numpy()
    for (color, box), probs in zip(proposals, probabilities):
        lamp_score = float(1.0 - probs[background_index])
        if lamp_score >= threshold:
            detections.append({"color": color, "box": box[:4], "score": lamp_score})
    return detections


def update_tracks(tracks: list[Track], detections: list[dict], frame_index: int,
                  width: int, height: int, history_size: int = 5) -> list[Track]:
    matched = set()
    diagonal = math.hypot(width, height)
    for track in tracks:
        best_index, best_distance = None, float("inf")
        tx, ty = center(track.box)
        for index, detection in enumerate(detections):
            if index in matched or detection["color"] != track.color:
                continue
            dx, dy = center(detection["box"])
            distance = math.hypot(dx - tx, dy - ty) / diagonal
            if distance < best_distance:
                best_index, best_distance = index, distance
        track.history.append(False)
        if best_index is not None and best_distance <= 0.055:
            detection = detections[best_index]
            matched.add(best_index)
            track.box = detection["box"]
            track.score = detection["score"]
            track.history[-1] = True
            track.last_frame = frame_index
    tracks = [track for track in tracks if frame_index - track.last_frame <= history_size]
    for index, detection in enumerate(detections):
        if index not in matched:
            history = deque([True], maxlen=history_size)
            tracks.append(Track(detection["color"], detection["box"], detection["score"], history, frame_index))
    return tracks


def stable_tracks(tracks: list[Track], required_hits: int = 3):
    return [track for track in tracks if sum(track.history) >= required_hits and track.history[-1]]


def write_events(
    rows: list[dict], path: Path, gap_sec: float = 0.38,
    minimum_sec: float = 0.20, minimum_peak: float = 0.0,
    maximum_sec: float | None = None,
):
    events, current = [], None
    for row in rows:
        state = "double" if row["red_on"] and row["green_on"] else (
            "left" if row["red_on"] else "right" if row["green_on"] else "none")
        t = float(row["time_sec"])
        if state == "none":
            if current and t - current["last_on"] > gap_sec:
                current["end_sec"] = current["last_on"]
                events.append(current)
                current = None
            continue
        if current is None or state != current["state"] or t - current["last_on"] > gap_sec:
            if current:
                current["end_sec"] = current["last_on"]
                events.append(current)
            current = {"state": state, "start_sec": t, "last_on": t, "peak_score": row["peak_score"]}
        else:
            current["last_on"] = t
            current["peak_score"] = max(current["peak_score"], row["peak_score"])
    if current:
        current["end_sec"] = current["last_on"]
        events.append(current)
    events = [
        event for event in events
        if event["end_sec"] - event["start_sec"] >= minimum_sec
        and float(event["peak_score"]) >= minimum_peak
        and (maximum_sec is None or event["end_sec"] - event["start_sec"] <= maximum_sec)
    ]
    for event in events:
        event.pop("last_on", None)
        event["duration_sec"] = event["end_sec"] - event["start_sec"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["state", "start_sec", "end_sec", "duration_sec", "peak_score"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent full-frame physical-lamp scanner for v4.3.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--analysis-fps", type=float, default=30.0)
    parser.add_argument("--threshold", type=float, default=0.225)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(args.model, device)
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
    writer = cv2.VideoWriter(str(args.output / "physical_lamp_preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             output_fps, (width, height))
    rows, tracks, previous_hist = [], [], None
    read_index, analyzed_index = 0, 0
    started = time.perf_counter()
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
        detections = classify_candidates(frame, model, classes, device, args.threshold)
        tracks = update_tracks(tracks, detections, analyzed_index, width, height)
        stable = stable_tracks(tracks)
        red = [track for track in stable if track.color == "red_lamp"]
        green = [track for track in stable if track.color == "green_lamp"]
        peak = max((track.score for track in stable), default=0.0)
        rows.append({"frame": analyzed_index, "time_sec": f"{timestamp:.4f}", "scene_cut": int(cut),
                     "raw_candidates": len(detections), "red_tracks": len(red), "green_tracks": len(green),
                     "red_on": int(bool(red)), "green_on": int(bool(green)), "peak_score": f"{peak:.5f}"})
        for track in stable:
            x1, y1, x2, y2 = track.box
            color = (0, 0, 255) if track.color == "red_lamp" else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{track.color}:{track.score:.2f}", (x1, max(18, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        state = "DOUBLE" if red and green else "LEFT" if red else "RIGHT" if green else "NONE"
        cv2.rectangle(frame, (0, 0), (520, 42), (0, 0, 0), -1)
        cv2.putText(frame, f"PHYSICAL LAMPS: {state}  t={timestamp:.2f}s", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        analyzed_index += 1
        if analyzed_index % 300 == 0:
            print(f"frames={analyzed_index} t={timestamp:.1f}s elapsed={time.perf_counter()-started:.1f}s", flush=True)
    cap.release()
    writer.release()
    with (args.output / "physical_lamp_frames.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    events = write_events(rows, args.output / "physical_lamp_events.csv")
    summary = {"input": str(args.input), "source_fps": source_fps, "analysis_fps": output_fps,
               "stride": stride, "start_sec": start, "end_sec": end, "analyzed_frames": analyzed_index,
               "elapsed_sec": time.perf_counter() - started, "events": events}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
