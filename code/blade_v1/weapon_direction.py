from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Box:
    confidence: float
    xyxy: tuple[float, float, float, float]

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy
        return np.array(((x1 + x2) / 2.0, (y1 + y2) / 2.0), dtype=np.float32)

    @property
    def width(self) -> float:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self) -> float:
        return self.xyxy[3] - self.xyxy[1]


@dataclass
class LineCandidate:
    guard: np.ndarray
    tip: np.ndarray
    confidence: float
    score: float


@dataclass
class BladeTrack:
    guard: np.ndarray | None = None
    vector: np.ndarray | None = None
    missing: int = 999
    pending_guard: np.ndarray | None = None
    pending_vector: np.ndarray | None = None
    pending_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate blade directions from fencer boxes and thin-line geometry. "
            "The discarded foil guard/tip model is not used."
        )
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--fencer-model",
        type=Path,
        default=REPO_ROOT / "models" / "fencer_v4.pt",
    )
    parser.add_argument("--fencer-conf", type=float, default=0.22)
    parser.add_argument("--max-estimate-frames", type=int, default=5)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--draw-candidates", action="store_true")
    return parser.parse_args()


def model_boxes(result) -> list[Box]:
    if result.boxes is None:
        return []
    boxes = []
    for xyxy, confidence, class_id in zip(
        result.boxes.xyxy.detach().cpu().numpy(),
        result.boxes.conf.detach().cpu().numpy(),
        result.boxes.cls.detach().cpu().numpy(),
    ):
        if int(class_id) == 0:
            boxes.append(Box(float(confidence), tuple(float(value) for value in xyxy)))
    return boxes


def select_fencers(detections: list[Box], width: int, height: int) -> list[Box]:
    plausible: list[tuple[float, Box]] = []
    for det in detections:
        if det.xyxy[3] / height < 0.50:
            continue
        if det.height / height < 0.12 or det.width / width < 0.015:
            continue
        plausible.append((det.confidence + 1.45 * det.xyxy[3] / height, det))
    plausible.sort(key=lambda item: item[0], reverse=True)
    top = [item[1] for item in plausible[:8]]

    best: tuple[Box, Box] | None = None
    best_score = -1e9
    for index, first in enumerate(top):
        for second in top[index + 1 :]:
            left, right = sorted((first, second), key=lambda item: item.center[0])
            separation = (right.center[0] - left.center[0]) / width
            midpoint = (right.center[0] + left.center[0]) / (2.0 * width)
            bottom_delta = abs(right.xyxy[3] - left.xyxy[3]) / height
            if not 0.14 <= separation <= 0.78:
                continue
            if not 0.22 <= midpoint <= 0.78 or bottom_delta > 0.22:
                continue
            score = first.confidence + second.confidence - 0.7 * bottom_delta
            if score > best_score:
                best_score = score
                best = (left, right)
    return list(best) if best else []


def expected_guard(fencer: Box, side: str) -> np.ndarray:
    x1, y1, x2, _ = fencer.xyxy
    x = x2 - 0.06 * fencer.width if side == "left" else x1 + 0.06 * fencer.width
    return np.array((x, y1 + 0.34 * fencer.height), dtype=np.float32)


def point_line_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    vector = end - start
    denominator = float(np.dot(vector, vector))
    if denominator < 1e-6:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, vector) / denominator)
    return float(np.linalg.norm(point - (start + t * vector)))


def project_to_line(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    vector = end - start
    denominator = float(np.dot(vector, vector))
    if denominator < 1e-6:
        return point.copy()
    t = float(np.dot(point - start, vector) / denominator)
    return start + t * vector


def line_segments(
    gray_roi: np.ndarray, foreground_roi: np.ndarray, offset: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray_roi)
    # A white top-hat keeps thin bright ridges (metallic blade pixels) while
    # suppressing broad objects and ordinary one-sided boundaries.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    enhanced = cv2.morphologyEx(clahe, cv2.MORPH_TOPHAT, kernel)
    foreground_mask = np.where(foreground_roi >= 8, 255, 0).astype(np.uint8)
    foreground_mask = cv2.dilate(foreground_mask, np.ones((5, 5), np.uint8), iterations=1)
    enhanced = cv2.bitwise_and(enhanced, foreground_mask)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    edges = cv2.Canny(enhanced, 24, 82, apertureSize=3, L2gradient=True)
    segments: list[tuple[np.ndarray, np.ndarray]] = []

    hough = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 360, threshold=26, minLineLength=34, maxLineGap=20
    )
    if hough is not None:
        for line in np.asarray(hough).reshape(-1, 4):
            segments.append(
                (
                    np.array((line[0], line[1]), dtype=np.float32) + offset,
                    np.array((line[2], line[3]), dtype=np.float32) + offset,
                )
            )

    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(enhanced)[0]
    if detected is not None:
        for line in np.asarray(detected).reshape(-1, 4):
            start = np.array((line[0], line[1]), dtype=np.float32) + offset
            end = np.array((line[2], line[3]), dtype=np.float32) + offset
            if np.linalg.norm(end - start) >= 34:
                segments.append((start, end))
    return segments


def foreground_support(diff: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length < 2:
        return 0.0
    count = max(12, min(90, int(length / 4)))
    height, width = diff.shape
    values = []
    for t in np.linspace(0.06, 0.94, count):
        point = start + t * vector
        x = int(np.clip(round(float(point[0])), 0, width - 1))
        y = int(np.clip(round(float(point[1])), 0, height - 1))
        patch = diff[max(0, y - 2) : min(height, y + 3), max(0, x - 2) : min(width, x + 3)]
        values.append(float(np.max(patch)))
    return float(np.percentile(values, 55))


def ridge_contrast(gray: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length < 2:
        return -999.0
    unit = vector / length
    normal = np.array((-unit[1], unit[0]), dtype=np.float32)
    count = max(12, min(80, int(length / 4)))
    values = []
    height, width = gray.shape
    for t in np.linspace(0.08, 0.92, count):
        point = start + t * vector
        inner = []
        for offset in (-2.0, 0.0, 2.0):
            sample = point + normal * offset
            x = int(np.clip(round(float(sample[0])), 0, width - 1))
            y = int(np.clip(round(float(sample[1])), 0, height - 1))
            inner.append(float(gray[y, x]))
        sides = []
        for offset in (-7.0, 7.0):
            sample = point + normal * offset
            x = int(np.clip(round(float(sample[0])), 0, width - 1))
            y = int(np.clip(round(float(sample[1])), 0, height - 1))
            sides.append(float(gray[y, x]))
        # The center must be brighter than both sides, unlike a normal edge.
        values.append(max(inner) - max(sides))
    return float(np.percentile(values, 60))


def angle_similarity(first: np.ndarray, second: np.ndarray | None) -> float:
    if second is None:
        return 0.5
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < 1 or second_norm < 1:
        return 0.0
    return abs(float(np.dot(first, second) / (first_norm * second_norm)))


def detect_blade_line(
    frame: np.ndarray,
    background_gray: np.ndarray,
    fencer: Box,
    opponent: Box,
    side: str,
    previous_vector: np.ndarray | None,
    anchor_override: np.ndarray | None = None,
) -> tuple[LineCandidate | None, list[tuple[np.ndarray, np.ndarray]]]:
    height, width = frame.shape[:2]
    anchor = (
        expected_guard(fencer, side)
        if anchor_override is None
        else np.asarray(anchor_override, dtype=np.float32)
    )
    midpoint = float((fencer.center[0] + opponent.center[0]) / 2.0)
    vertical_radius = max(90.0, 0.48 * fencer.height)
    if side == "left":
        x1, x2, sign = anchor[0] - 0.08 * width, midpoint + 0.14 * width, 1.0
    else:
        x1, x2, sign = midpoint - 0.14 * width, anchor[0] + 0.08 * width, -1.0
    y1, y2 = anchor[1] - vertical_radius, anchor[1] + vertical_radius
    left = max(0, min(width - 2, int(round(x1))))
    right = max(left + 2, min(width, int(round(x2))))
    top = max(0, min(height - 2, int(round(y1))))
    bottom = max(top + 2, min(height, int(round(y2))))
    full_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    full_diff = cv2.absdiff(full_gray, background_gray)
    gray = full_gray[top:bottom, left:right]
    diff_roi = full_diff[top:bottom, left:right]
    segments = line_segments(
        gray, diff_roi, np.array((left, top), dtype=np.float32)
    )

    ranked: list[tuple[float, np.ndarray, np.ndarray]] = []
    for start, end in segments:
        vector = end - start
        length = float(np.linalg.norm(vector))
        # A broadcast-scale sabre should be a local segment emerging from the
        # hand, not a long background edge whose infinite extension happens to
        # pass through the estimated guard position.
        if length < max(38.0, 0.42 * fencer.height):
            continue
        if length > 1.08 * fencer.height:
            continue
        if abs(float(vector[0])) / max(length, 1.0) < 0.16:
            continue
        forward_start = sign * float(start[0] - anchor[0])
        forward_end = sign * float(end[0] - anchor[0])
        if forward_end < forward_start:
            start, end = end, start
            vector = end - start
            forward_start, forward_end = forward_end, forward_start
        if forward_end < 0.12 * fencer.height or forward_end > 1.62 * fencer.height:
            continue
        if forward_start < -0.24 * fencer.height or forward_start > 0.34 * fencer.height:
            continue
        endpoint_distance = min(
            float(np.linalg.norm(start - anchor)), float(np.linalg.norm(end - anchor))
        )
        if endpoint_distance > max(40.0, 0.28 * fencer.height):
            continue
        distance = point_line_distance(anchor, start, end)
        if distance > max(34.0, 0.16 * fencer.height):
            continue
        ridge = ridge_contrast(full_gray, start, end)
        if ridge < 4.0:
            continue
        support = foreground_support(full_diff, start, end)
        if support < 9.0:
            continue
        temporal = angle_similarity(vector, previous_vector)
        horizontal = abs(float(vector[0])) / length
        score = (
            0.0060 * length
            + 1.65 * forward_end / width
            + 0.55 * horizontal
            + 0.80 * temporal
            + 0.035 * ridge
            + 0.018 * support
            - 1.25 * distance / max(50.0, 0.22 * fencer.height)
            - 1.15 * endpoint_distance / max(50.0, 0.28 * fencer.height)
        )
        ranked.append((score, start, end))

    if not ranked:
        return None, segments
    score, start, end = max(ranked, key=lambda item: item[0])
    # Deliberately do not join other collinear segments here. The earlier
    # prototype could stitch a real short glint to a door/floor edge and invent
    # a blade several metres long.
    tip = end
    blade_length = float(np.linalg.norm(tip - anchor))
    if blade_length < 0.42 * fencer.height or blade_length > 1.12 * fencer.height:
        return None, segments
    confidence = float(np.clip(0.10 + 0.17 * score, 0.10, 0.76))
    return LineCandidate(anchor, tip, confidence, score), segments


def smooth(old: np.ndarray | None, new: np.ndarray, alpha: float) -> np.ndarray:
    return new.copy() if old is None else alpha * new + (1.0 - alpha) * old


def update_track(
    track: BladeTrack, detection: LineCandidate | None, max_estimate_frames: int
) -> tuple[np.ndarray | None, np.ndarray | None, str | None, float | None]:
    if detection is not None:
        candidate_vector = detection.tip - detection.guard
        candidate_length = float(np.linalg.norm(candidate_vector))
        pending_length = (
            float(np.linalg.norm(track.pending_vector))
            if track.pending_vector is not None
            else 0.0
        )
        consistent = (
            track.pending_vector is not None
            and angle_similarity(candidate_vector, track.pending_vector)
            >= math.cos(math.radians(18))
            and 0.68 <= candidate_length / max(pending_length, 1.0) <= 1.47
        )
        if consistent:
            track.pending_guard = smooth(track.pending_guard, detection.guard, 0.55)
            track.pending_vector = smooth(track.pending_vector, candidate_vector, 0.55)
            track.pending_count += 1
        else:
            track.pending_guard = detection.guard.copy()
            track.pending_vector = candidate_vector.copy()
            track.pending_count = 1
        # A single thin edge is too ambiguous in this footage. Require three
        # consecutive compatible observations before exposing a result.
        if track.pending_count >= 3:
            track.guard = smooth(track.guard, track.pending_guard, 0.70)
            track.vector = smooth(track.vector, track.pending_vector, 0.68)
            track.missing = 0
            return track.guard, track.guard + track.vector, "geometry", detection.confidence
    else:
        track.pending_guard = None
        track.pending_vector = None
        track.pending_count = 0
    track.missing += 1
    if track.guard is None or track.vector is None or track.missing > max_estimate_frames:
        return None, None, None, None
    return track.guard, track.guard + track.vector, "temporal", None


def dashed_line(frame, start, end, color, thickness=3, dash=14, gap=9) -> None:
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length < 1:
        return
    unit = vector / length
    position = 0.0
    while position < length:
        a = start + unit * position
        b = start + unit * min(length, position + dash)
        cv2.line(
            frame,
            tuple(np.rint(a).astype(int)),
            tuple(np.rint(b).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )
        position += dash + gap


def draw_blade(frame, side, guard, tip, source, confidence) -> None:
    if guard is None or tip is None:
        return
    color = (255, 170, 30) if side == "left" else (30, 140, 255)
    guard_point = tuple(np.rint(guard).astype(int))
    tip_point = tuple(np.rint(tip).astype(int))
    if source == "geometry":
        cv2.line(frame, guard_point, tip_point, color, 4, cv2.LINE_AA)
    else:
        dashed_line(frame, guard, tip, color)
    cv2.circle(frame, guard_point, 8, color, -1, cv2.LINE_AA)
    cv2.circle(frame, guard_point, 13, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.drawMarker(frame, tip_point, color, cv2.MARKER_TILTED_CROSS, 20, 3, cv2.LINE_AA)
    vector = tip - guard
    angle = math.degrees(math.atan2(-float(vector[1]), float(vector[0])))
    label = f"{side} {angle:+.0f}deg {source}"
    cv2.putText(
        frame,
        label,
        (tip_point[0] + 8, max(24, tip_point[1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def point_json(point: np.ndarray | None) -> list[float] | None:
    return None if point is None else [round(float(point[0]), 2), round(float(point[1]), 2)]


def build_background(video_path: Path, sample_count: int = 17) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        raise RuntimeError(f"No frames: {video_path}")
    samples = []
    for position in np.linspace(0.10, 0.96, sample_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(position * (frame_count - 1))))
        ok, frame = cap.read()
        if ok:
            samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    if len(samples) < 5:
        raise RuntimeError(f"Too few background samples: {len(samples)}")
    return np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)


def main() -> None:
    args = parse_args()
    for path in (args.video, args.fencer_model):
        if not path.exists():
            raise FileNotFoundError(path)
    print("building_median_background", flush=True)
    background_gray = build_background(args.video)
    model = YOLO(str(args.fencer_model))
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(args.video)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(args.output_video)

    tracks = {"left": BladeTrack(), "right": BladeTrack()}
    stats = {
        "frames": 0,
        "frames_with_two_fencers": 0,
        "left_geometry": 0,
        "right_geometry": 0,
        "left_temporal": 0,
        "right_temporal": 0,
    }
    records = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames is not None and frame_index >= args.max_frames):
            break
        result = model.predict(
            frame, imgsz=768, conf=args.fencer_conf, iou=0.55, device=0, verbose=False
        )[0]
        fencers = select_fencers(model_boxes(result), width, height)
        raw_results: dict[str, LineCandidate | None] = {"left": None, "right": None}
        frame_record: dict[str, object] = {
            "frame": frame_index,
            "time_sec": round(frame_index / fps, 4),
        }

        if len(fencers) == 2:
            stats["frames_with_two_fencers"] += 1
            all_segments: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
            for side, fencer, opponent in (
                ("left", fencers[0], fencers[1]),
                ("right", fencers[1], fencers[0]),
            ):
                candidate, segments = detect_blade_line(
                    frame,
                    background_gray,
                    fencer,
                    opponent,
                    side,
                    tracks[side].vector,
                )
                raw_results[side] = candidate
                all_segments[side] = segments
            if args.draw_candidates:
                for side, segments in all_segments.items():
                    color = (100, 160, 100) if side == "left" else (100, 100, 160)
                    for start, end in segments:
                        cv2.line(
                            frame,
                            tuple(np.rint(start).astype(int)),
                            tuple(np.rint(end).astype(int)),
                            color,
                            1,
                            cv2.LINE_AA,
                        )
            for side, fencer in zip(("left", "right"), fencers):
                x1, y1, x2, y2 = map(round, fencer.xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)
                cv2.circle(
                    frame,
                    tuple(np.rint(expected_guard(fencer, side)).astype(int)),
                    5,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )

        for side in ("left", "right"):
            guard, tip, source, confidence = update_track(
                tracks[side], raw_results[side], args.max_estimate_frames
            )
            if source is not None:
                stats[f"{side}_{source}"] += 1
            draw_blade(frame, side, guard, tip, source, confidence)
            vector = None if guard is None or tip is None else tip - guard
            frame_record[side] = {
                "guard": point_json(guard),
                "guard_source": "estimated_from_fencer_box" if guard is not None else None,
                "tip": point_json(tip),
                "source": source,
                "confidence": round(confidence, 4) if confidence is not None else None,
                "angle_deg": round(
                    math.degrees(math.atan2(-float(vector[1]), float(vector[0]))), 2
                )
                if vector is not None
                else None,
                "length_px": round(float(np.linalg.norm(vector)), 2)
                if vector is not None
                else None,
            }

        cv2.rectangle(frame, (0, 0), (840, 38), (0, 0, 0), -1)
        cv2.putText(
            frame,
            "conservative prototype: solid=accepted geometry | dashed=temporal hold",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        records.append(frame_record)
        stats["frames"] += 1
        frame_index += 1
        if frame_index % 100 == 0:
            print(f"processed={frame_index}/{total}", flush=True)

    cap.release()
    writer.release()
    payload = {
        "video": str(args.video),
        "fencer_model": str(args.fencer_model),
        "method": "fencer boxes + median-background subtraction + guard anchor + bright-ridge Hough/LSD geometry + temporal hold",
        "old_foil_guard_tip_model_used": False,
        "fps": fps,
        "width": width,
        "height": height,
        "stats": stats,
        "frames": records,
    }
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"output_video={args.output_video}")
    print(f"output_json={args.output_json}")


if __name__ == "__main__":
    main()
