from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from track_device_anchors_v46 import center, scene_hist, valid_shape


def valid_shape_v47(row: dict[str, str], width: int, height: int) -> bool:
    """Accept both the older wide panels and the newer tall central scoring boxes."""
    bw = (float(row["x2"]) - float(row["x1"])) / width
    bh = (float(row["y2"]) - float(row["y1"])) / height
    aspect = bw / max(bh, 1e-6)
    return 0.04 <= bw <= 0.38 and 0.035 <= bh <= 0.34 and 0.20 <= aspect <= 4.5


def expand_box(box: np.ndarray, width: int, height: int, factor: float = 1.25) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    half_w = (x2 - x1) * factor * 0.5
    half_h = (y2 - y1) * factor * 0.5
    return (
        max(0, round(cx - half_w)),
        max(0, round(cy - half_h)),
        min(width, round(cx + half_w)),
        min(height, round(cy + half_h)),
    )


def feature_points(gray: np.ndarray, box: np.ndarray, maximum: int = 120) -> np.ndarray | None:
    height, width = gray.shape
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    margin_x = max(2, round((x2 - x1) * 0.06))
    margin_y = max(2, round((y2 - y1) * 0.06))
    x1, x2 = max(0, x1 + margin_x), min(width, x2 - margin_x)
    y1, y2 = max(0, y1 + margin_y), min(height, y2 - margin_y)
    if x2 <= x1 or y2 <= y1:
        return None
    mask = np.zeros_like(gray)
    mask[y1:y2, x1:x2] = 255
    return cv2.goodFeaturesToTrack(
        gray,
        mask=mask,
        maxCorners=maximum,
        qualityLevel=0.012,
        minDistance=5,
        blockSize=5,
    )


def optical_flow_box(
    previous_gray: np.ndarray,
    gray: np.ndarray,
    points: np.ndarray | None,
    box: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    if points is None or len(points) < 6:
        return None, None, 0
    next_points, status, _error = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        gray,
        points,
        None,
        winSize=(25, 25),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_points is None or status is None:
        return None, None, 0
    back_points, back_status, _back_error = cv2.calcOpticalFlowPyrLK(
        gray,
        previous_gray,
        next_points,
        None,
        winSize=(25, 25),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if back_points is None or back_status is None:
        return None, None, 0
    forward = next_points.reshape(-1, 2)
    backward = back_points.reshape(-1, 2)
    original = points.reshape(-1, 2)
    valid = (
        status.reshape(-1).astype(bool)
        & back_status.reshape(-1).astype(bool)
        & (np.linalg.norm(original - backward, axis=1) <= 1.8)
    )
    old = original[valid]
    new = forward[valid]
    if len(old) < 6:
        return None, None, len(old)
    matrix, inliers = cv2.estimateAffinePartial2D(
        old,
        new,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.2,
        maxIters=1000,
        confidence=0.995,
    )
    if matrix is None or inliers is None or int(inliers.sum()) < 5:
        return None, None, len(old)
    scale = math.hypot(float(matrix[0, 0]), float(matrix[0, 1]))
    diagonal = math.hypot(width, height)
    motion = math.hypot(float(matrix[0, 2]), float(matrix[1, 2])) / diagonal
    if not 0.92 <= scale <= 1.08 or motion > 0.045:
        return None, None, int(inliers.sum())
    corners = np.array(
        [[[box[0], box[1]]], [[box[2], box[1]]], [[box[2], box[3]]], [[box[0], box[3]]]],
        dtype=np.float32,
    )
    transformed = cv2.transform(corners, matrix).reshape(-1, 2)
    candidate = np.array(
        [transformed[:, 0].min(), transformed[:, 1].min(), transformed[:, 0].max(), transformed[:, 1].max()],
        dtype=np.float32,
    )
    candidate[[0, 2]] = np.clip(candidate[[0, 2]], 0, width - 1)
    candidate[[1, 3]] = np.clip(candidate[[1, 3]], 0, height - 1)
    row = {key: str(value) for key, value in zip(("x1", "y1", "x2", "y2"), candidate)}
    if not valid_shape_v47(row, width, height):
        return None, None, int(inliers.sum())
    kept = new[inliers.reshape(-1).astype(bool)].reshape(-1, 1, 2).astype(np.float32)
    return candidate, kept, len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid detector/optical-flow score-device tracker for v4.7.")
    parser.add_argument("input", type=Path)
    parser.add_argument("anchors", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--required-hits", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--flow-max-gap", type=float, default=7.0)
    parser.add_argument("--hold-max-gap", type=float, default=0.8)
    parser.add_argument("--acquisition-max-gap", type=float, default=1.2)
    parser.add_argument("--cut-threshold", type=float, default=0.60)
    parser.add_argument("--feature-factor", type=float, default=1.60)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with args.anchors.open("r", encoding="utf-8-sig", newline="") as handle:
        anchor_rows = list(csv.DictReader(handle))
    grouped: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in anchor_rows:
        grouped[round(float(row["time_sec"]), 3)].append(row)

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    diagonal = math.hypot(width, height)
    writer = cv2.VideoWriter(
        str(args.output / "device_track_preview_v47.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )

    previous_hist = None
    previous_gray = None
    points = None
    smooth_box = None
    locked = False
    last_anchor_time = None
    hit_history: deque[bool] = deque(maxlen=max(12, round(args.fps * 2.5)))
    rows: list[dict[str, object]] = []
    timestamp = args.start
    started = time.perf_counter()
    cut_count = 0
    acquisition_resets = 0

    while timestamp <= args.end + 1e-6:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = cap.read()
        if not ok:
            timestamp += 1.0 / args.fps
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_hist = scene_hist(frame)
        cut = (
            previous_hist is not None
            and cv2.compareHist(previous_hist, current_hist, cv2.HISTCMP_BHATTACHARYYA) >= args.cut_threshold
        )
        previous_hist = current_hist
        if cut:
            cut_count += 1
            smooth_box = None
            points = None
            locked = False
            last_anchor_time = None
            hit_history.clear()

        # Before a track is locked, one weak or badly shaped first candidate
        # must not poison acquisition indefinitely.  Forget that seed after a
        # short unsupported gap, then let a later pair of learned detections
        # establish a fresh track.  Locked tracks still use the stricter
        # continuity and optical-flow rules below.
        if (
            not locked
            and smooth_box is not None
            and last_anchor_time is not None
            and timestamp - last_anchor_time > args.acquisition_max_gap
        ):
            smooth_box = None
            points = None
            last_anchor_time = None
            hit_history.clear()
            acquisition_resets += 1

        candidates = [
            row
            for row in grouped.get(round(timestamp, 3), [])
            if row.get("confidence") and valid_shape_v47(row, width, height)
        ]
        candidates.sort(key=lambda row: float(row["confidence"]), reverse=True)
        chosen = None
        if candidates and smooth_box is None:
            chosen = candidates[0]
        elif candidates and smooth_box is not None:
            sx, sy = center(smooth_box)
            scored = []
            for row in candidates:
                box = np.array([float(row[key]) for key in ("x1", "y1", "x2", "y2")], dtype=np.float32)
                cx, cy = center(box)
                distance = math.hypot(cx - sx, cy - sy) / diagonal
                size_change = abs(math.log(max(1.0, (box[2] - box[0]) * (box[3] - box[1])) / max(1.0, (smooth_box[2] - smooth_box[0]) * (smooth_box[3] - smooth_box[1]))))
                if distance <= 0.10 and size_change <= 0.55:
                    scored.append((distance + 0.025 * size_change - 0.03 * float(row["confidence"]), row))
            if scored:
                chosen = min(scored, key=lambda item: item[0])[1]

        source = "missing"
        flow_points = 0
        if chosen is not None:
            raw_box = np.array([float(chosen[key]) for key in ("x1", "y1", "x2", "y2")], dtype=np.float32)
            smooth_box = raw_box if smooth_box is None else (1.0 - args.alpha) * smooth_box + args.alpha * raw_box
            hit_history.append(True)
            last_anchor_time = timestamp
            if sum(hit_history) >= args.required_hits:
                locked = True
            feature_box = np.array(expand_box(smooth_box, width, height, args.feature_factor), dtype=np.float32)
            points = feature_points(gray, feature_box)
            flow_points = 0 if points is None else len(points)
            source = "anchor"
        elif locked and smooth_box is not None and last_anchor_time is not None:
            hit_history.append(False)
            gap = timestamp - last_anchor_time
            flowed_box = None
            flowed_points = None
            if gap <= args.flow_max_gap and previous_gray is not None:
                flowed_box, flowed_points, flow_points = optical_flow_box(
                    previous_gray, gray, points, smooth_box, width, height
                )
            if flowed_box is not None:
                smooth_box = flowed_box
                points = flowed_points
                if points is None or len(points) < 18:
                    feature_box = np.array(
                        expand_box(smooth_box, width, height, args.feature_factor), dtype=np.float32
                    )
                    refreshed = feature_points(gray, feature_box)
                    if refreshed is not None and len(refreshed) >= 6:
                        points = refreshed
                source = "optical_flow"
            elif gap <= args.hold_max_gap:
                source = "hold"
            else:
                smooth_box = None
                points = None
                locked = False
                hit_history.clear()
        else:
            hit_history.append(False)

        if locked and smooth_box is not None:
            x1, y1, x2, y2 = [round(value) for value in smooth_box]
            confidence = float(chosen["confidence"]) if chosen is not None else 0.0
            rows.append(
                {
                    "time_sec": round(timestamp, 3),
                    "scene_cut": int(cut),
                    "confidence": round(confidence, 6),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "source": source,
                    "flow_points": flow_points,
                    "seconds_since_anchor": round(timestamp - (last_anchor_time or timestamp), 3),
                }
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            rx1, ry1, rx2, ry2 = expand_box(smooth_box, width, height)
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
            status = f"TRACK source={source} gap={timestamp - (last_anchor_time or timestamp):.1f}s points={flow_points}"
            color = (0, 255, 0) if source == "anchor" else (0, 200, 255)
        else:
            status = "TRACK MISSING"
            color = (0, 0, 255)
        cv2.rectangle(frame, (0, 0), (760, 42), (0, 0, 0), -1)
        cv2.putText(frame, f"{status} t={timestamp:.1f}s", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        writer.write(frame)
        previous_gray = gray
        timestamp += 1.0 / args.fps

    cap.release()
    writer.release()
    csv_path = args.output / "stable_device_track.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer_csv.writeheader()
            writer_csv.writerows(rows)
    else:
        csv_path.write_text("time_sec\n", encoding="utf-8-sig")
    summary = {
        "input": str(args.input),
        "start": args.start,
        "end": args.end,
        "fps": args.fps,
        "sampled_frames": round((args.end - args.start) * args.fps) + 1,
        "stable_rows": len(rows),
        "coverage": len(rows) / max(1, round((args.end - args.start) * args.fps) + 1),
        "source_counts": {
            source_name: sum(1 for row in rows if row["source"] == source_name)
            for source_name in ("anchor", "optical_flow", "hold")
        },
        "scene_cuts": cut_count,
        "acquisition_resets": acquisition_resets,
        "elapsed_sec": time.perf_counter() - started,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
