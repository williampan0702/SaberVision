from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np


ROIS = {
    "left": (505, 608, 565, 653),
    "right": (715, 608, 775, 653),
}


def digit_mask(frame: np.ndarray, roi: tuple[int, int, int, int]):
    x1, y1, x2, y2 = roi
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, 0.0, 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean_gray = float(gray.mean())
    raw_mask = (gray <= 105).astype(np.uint8)
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    black_fraction = float(raw_mask.mean())
    if mean_gray < 135 or not 0.015 <= black_fraction <= 0.40:
        return None, mean_gray, black_fraction

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(raw_mask, 8)
    digit_components = []
    crop_h, crop_w = raw_mask.shape
    for index in range(1, count):
        x, y, component_w, component_h, area = stats[index]
        # Digits are tall interior components.  Curved panel borders touch an
        # edge; compression specks and punctuation are too small or too short.
        if area < 14 or component_h < crop_h * 0.30:
            continue
        if x <= 0 or y <= 0 or x + component_w >= crop_w - 1 or y + component_h >= crop_h - 1:
            continue
        digit_components.append((x, y, component_w, component_h, area))
    if not digit_components:
        return None, mean_gray, black_fraction

    x1 = min(item[0] for item in digit_components)
    y1 = min(item[1] for item in digit_components)
    x2 = max(item[0] + item[2] for item in digit_components)
    y2 = max(item[1] + item[3] for item in digit_components)
    digit = raw_mask[y1:y2, x1:x2]
    if digit.size == 0:
        return None, mean_gray, black_fraction
    canvas = np.zeros((48, 64), dtype=np.uint8)
    scale = min(56.0 / digit.shape[1], 40.0 / digit.shape[0])
    target_w = max(1, round(digit.shape[1] * scale))
    target_h = max(1, round(digit.shape[0] * scale))
    normalized = cv2.resize(digit, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    offset_x = (canvas.shape[1] - target_w) // 2
    offset_y = (canvas.shape[0] - target_h) // 2
    canvas[offset_y : offset_y + target_h, offset_x : offset_x + target_w] = normalized
    return canvas, mean_gray, black_fraction


def mask_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.logical_xor(first, second).mean())


class PersistentChange:
    def __init__(self, required_frames: int, change_threshold: float, consistency_threshold: float):
        self.required_frames = required_frames
        self.change_threshold = change_threshold
        self.consistency_threshold = consistency_threshold
        self.current: np.ndarray | None = None
        self.initial = deque(maxlen=required_frames)
        self.candidate: list[tuple[float, np.ndarray]] = []
        self.state_index = 0

    def update(self, timestamp: float, mask: np.ndarray | None):
        if mask is None:
            self.candidate.clear()
            return None, "not_visible", ""
        if self.current is None:
            self.initial.append(mask)
            if len(self.initial) == self.required_frames:
                self.current = (np.median(np.stack(self.initial), axis=0) >= 0.5).astype(np.uint8)
            return None, "initializing", ""

        distance = mask_distance(self.current, mask)
        if distance < self.change_threshold:
            self.candidate.clear()
            return None, "stable", distance

        if self.candidate and mask_distance(self.candidate[-1][1], mask) > self.consistency_threshold:
            self.candidate = [(timestamp, mask)]
        else:
            self.candidate.append((timestamp, mask))
        if len(self.candidate) < self.required_frames:
            return None, "candidate", distance

        candidate_stack = np.stack([item[1] for item in self.candidate[-self.required_frames :]])
        candidate_template = (np.median(candidate_stack, axis=0) >= 0.5).astype(np.uint8)
        if mask_distance(self.current, candidate_template) < self.change_threshold:
            self.candidate.clear()
            return None, "returned", distance
        change_time = self.candidate[-self.required_frames][0]
        transition_distance = mask_distance(self.current, candidate_template)
        previous_state = self.state_index
        self.state_index += 1
        self.current = candidate_template
        self.candidate.clear()
        return {
            "time_sec": round(change_time, 4),
            "from_state": previous_state,
            "to_state": self.state_index,
            "distance": round(transition_distance, 6),
        }, "confirmed_change", distance


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect persistent left/right score changes from the fixed Riyadh overlay.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--required-frames", type=int, default=3)
    parser.add_argument("--change-threshold", type=float, default=0.060)
    parser.add_argument("--consistency-threshold", type=float, default=0.035)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(source_fps / min(source_fps, args.fps)))
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)
    detectors = {
        side: PersistentChange(args.required_frames, args.change_threshold, args.consistency_threshold)
        for side in ROIS
    }
    rows = []
    changes = []
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
        row = {"time_sec": f"{timestamp:.4f}"}
        for side, roi in ROIS.items():
            mask, mean_gray, black_fraction = digit_mask(frame, roi)
            change, status, distance = detectors[side].update(timestamp, mask)
            row.update(
                {
                    f"{side}_visible": int(mask is not None),
                    f"{side}_mean_gray": f"{mean_gray:.3f}",
                    f"{side}_black_fraction": f"{black_fraction:.6f}",
                    f"{side}_status": status,
                    f"{side}_distance": f"{float(distance):.6f}" if distance != "" else "",
                    f"{side}_state_index": detectors[side].state_index,
                }
            )
            if change is not None:
                changes.append({"side": side, **change})
        rows.append(row)
    cap.release()

    with (args.output / "score_overlay_frames.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    change_fields = ["side", "time_sec", "from_state", "to_state", "distance"]
    with (args.output / "score_changes.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=change_fields)
        writer.writeheader()
        writer.writerows(changes)
    summary = {
        "input": str(args.input),
        "sampled_frames": len(rows),
        "changes": changes,
        "counts": {side: sum(change["side"] == side for change in changes) for side in ROIS},
        "settings": {
            "fps": args.fps,
            "required_frames": args.required_frames,
            "change_threshold": args.change_threshold,
            "consistency_threshold": args.consistency_threshold,
            "rois": ROIS,
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
