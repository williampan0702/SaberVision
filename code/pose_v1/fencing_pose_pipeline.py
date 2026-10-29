from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[2]


KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

SKELETON = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


@dataclass
class Box:
    confidence: float
    xyxy: np.ndarray

    @property
    def width(self) -> float:
        return float(self.xyxy[2] - self.xyxy[0])

    @property
    def height(self) -> float:
        return float(self.xyxy[3] - self.xyxy[1])

    @property
    def center(self) -> np.ndarray:
        return np.array(
            ((self.xyxy[0] + self.xyxy[2]) / 2, (self.xyxy[1] + self.xyxy[3]) / 2),
            dtype=np.float32,
        )


@dataclass
class PointState:
    xy: np.ndarray | None = None
    confidence: float = 0.0
    missing: int = 999


@dataclass
class FencerTrack:
    box: Box | None = None
    box_missing: int = 999
    points: list[PointState] = field(
        default_factory=lambda: [PointState() for _ in range(17)]
    )
    weapon_wrist_index: int | None = None
    pelvis_history: list[np.ndarray] = field(default_factory=list)
    inward_speed: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fencing-oriented body pose baseline with temporal smoothing and motion features."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--fencer-model",
        type=Path,
        default=REPO_ROOT / "models" / "fencer_v4.pt",
    )
    parser.add_argument(
        "--pose-model",
        type=Path,
        default=REPO_ROOT / "models" / "fencing_pose_v1.pt",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--fencer-imgsz", type=int, default=768)
    parser.add_argument(
        "--fencer-pair-profile",
        choices=("legacy", "multivenue"),
        default="legacy",
        help=(
            "legacy preserves the v3 geometry filter; multivenue accepts wider "
            "full-body lunge boxes produced by the v4 detector"
        ),
    )
    parser.add_argument("--pose-imgsz", type=int, default=512)
    parser.add_argument(
        "--fallback-person-model", type=Path,
        help=(
            "Optional generic person/Pose model used only when the custom "
            "fencer model cannot form a pair (useful for high-angle views)."
        ),
    )
    parser.add_argument("--fallback-imgsz", type=int, default=1280)
    parser.add_argument(
        "--lunge-mode", choices=("conservative", "balanced"),
        default="conservative",
        help=(
            "conservative keeps the high-precision v1 geometry; balanced "
            "adds a high-speed fallback calibrated across three review sets"
        ),
    )
    return parser.parse_args()


def model_boxes(result) -> list[Box]:
    if result.boxes is None:
        return []
    boxes: list[Box] = []
    for xyxy, confidence, class_id in zip(
        result.boxes.xyxy.detach().cpu().numpy(),
        result.boxes.conf.detach().cpu().numpy(),
        result.boxes.cls.detach().cpu().numpy(),
    ):
        if int(class_id) == 0:
            boxes.append(
                Box(float(confidence), np.asarray(xyxy, dtype=np.float32).copy())
            )
    return boxes


def select_fencers(
    detections: list[Box], width: int, height: int,
    min_bottom_ratio: float = 0.50,
    max_bottom_ratio: float = 1.0,
    min_height_ratio: float = 0.12,
    min_aspect_ratio: float = 1.05,
    min_separation_ratio: float = 0.14,
    max_separation_ratio: float = 0.78,
    min_midpoint_ratio: float = 0.22,
    max_midpoint_ratio: float = 0.78,
    max_bottom_delta_ratio: float = 0.22,
    min_height_balance: float = 0.43,
) -> list[Box]:
    plausible: list[tuple[float, Box]] = []
    for det in detections:
        if det.xyxy[3] / height < min_bottom_ratio:
            continue
        if det.xyxy[3] / height > max_bottom_ratio:
            continue
        if det.height / height < min_height_ratio or det.width / width < 0.015:
            continue
        if det.height / max(det.width, 1.0) < min_aspect_ratio:
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
            height_balance = min(first.height, second.height) / max(
                max(first.height, second.height), 1.0
            )
            if not min_separation_ratio <= separation <= max_separation_ratio:
                continue
            if (
                not min_midpoint_ratio <= midpoint <= max_midpoint_ratio
                or bottom_delta > max_bottom_delta_ratio
                or height_balance < min_height_balance
            ):
                continue
            # Printed/background fencers are often confident but much smaller than
            # the real opponent. Prefer a similarly scaled on-piste pair.
            score = (
                first.confidence
                + second.confidence
                + 0.35 * height_balance
                - 0.7 * bottom_delta
            )
            if score > best_score:
                best_score = score
                best = (left, right)
    return list(best) if best else []


def pair_temporally_plausible(
    selected: list[Box], tracks: dict[str, FencerTrack], width: int,
    max_missing_frames: int = 6,
) -> bool:
    if len(selected) != 2:
        return False
    for side, detection in zip(("left", "right"), selected):
        track = tracks[side]
        if track.box is None or track.box_missing > max_missing_frames:
            continue
        horizontal_jump = abs(
            float(detection.center[0] - track.box.center[0])
        ) / max(width, 1)
        scale_ratio = detection.height / max(track.box.height, 1.0)
        if horizontal_jump > 0.20 or not 0.35 <= scale_ratio <= 2.8:
            return False
    return True


def update_box(
    track: FencerTrack, detection: Box | None, max_missing_frames: int = 6
) -> Box | None:
    if detection is not None:
        if track.box is None or track.box_missing >= max_missing_frames:
            track.box = Box(detection.confidence, detection.xyxy.copy())
        else:
            track.box.xyxy = 0.72 * detection.xyxy + 0.28 * track.box.xyxy
            track.box.confidence = detection.confidence
        track.box_missing = 0
    else:
        track.box_missing += 1
        if track.box_missing > max_missing_frames:
            track.box = None
    return track.box


def pose_crop(frame: np.ndarray, box: Box) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = frame.shape[:2]
    pad_x = 0.20 * box.width
    pad_y = 0.10 * box.height
    x1 = max(0, int(round(box.xyxy[0] - pad_x)))
    y1 = max(0, int(round(box.xyxy[1] - pad_y)))
    x2 = min(width, int(round(box.xyxy[2] + pad_x)))
    y2 = min(height, int(round(box.xyxy[3] + pad_y)))
    return frame[y1:y2, x1:x2].copy(), (x1, y1)


def detect_pose(
    pose_model: YOLO,
    frame: np.ndarray,
    box: Box,
    pose_imgsz: int,
    device: int | str,
) -> tuple[np.ndarray, np.ndarray] | None:
    crop, (offset_x, offset_y) = pose_crop(frame, box)
    if crop.size == 0:
        return None
    result = pose_model.predict(
        crop,
        imgsz=pose_imgsz,
        conf=0.12,
        iou=0.50,
        device=device,
        verbose=False,
    )[0]
    if result.keypoints is None or result.keypoints.xy is None:
        return None
    if len(result.keypoints.xy) == 0:
        return None

    person_index = 0
    if result.boxes is not None and len(result.boxes) > 1:
        crop_center = np.array((crop.shape[1] / 2, crop.shape[0] / 2), dtype=np.float32)
        ranks = []
        for index, pose_box in enumerate(result.boxes.xyxy.detach().cpu().numpy()):
            center = np.array(
                ((pose_box[0] + pose_box[2]) / 2, (pose_box[1] + pose_box[3]) / 2),
                dtype=np.float32,
            )
            area = max(1.0, float(pose_box[2] - pose_box[0])) * max(
                1.0, float(pose_box[3] - pose_box[1])
            )
            ranks.append(area / (1.0 + 2.0 * float(np.linalg.norm(center - crop_center))))
        person_index = int(np.argmax(ranks))

    xy = result.keypoints.xy[person_index].detach().cpu().numpy().astype(np.float32)
    if result.keypoints.conf is None:
        confidence = np.ones((len(xy),), dtype=np.float32)
    else:
        confidence = (
            result.keypoints.conf[person_index]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    xy[:, 0] += offset_x
    xy[:, 1] += offset_y
    return xy, confidence


def update_points(
    track: FencerTrack, measurement: tuple[np.ndarray, np.ndarray] | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    measured_xy = measurement[0] if measurement is not None else np.zeros((17, 2), np.float32)
    measured_conf = measurement[1] if measurement is not None else np.zeros((17,), np.float32)
    output_xy = np.zeros((17, 2), dtype=np.float32)
    output_conf = np.zeros((17,), dtype=np.float32)
    held = np.zeros((17,), dtype=bool)

    for index, state in enumerate(track.points):
        valid = (
            index < len(measured_conf)
            and float(measured_conf[index]) >= 0.12
            and float(np.linalg.norm(measured_xy[index])) > 1.0
        )
        if valid:
            alpha = float(np.clip(0.38 + 0.42 * measured_conf[index], 0.42, 0.78))
            if state.xy is None or state.missing > 4:
                state.xy = measured_xy[index].copy()
            else:
                state.xy = alpha * measured_xy[index] + (1.0 - alpha) * state.xy
            state.confidence = float(measured_conf[index])
            state.missing = 0
        else:
            state.missing += 1
            if state.xy is not None and state.missing <= 4:
                state.confidence *= 0.72
                held[index] = True
            else:
                state.confidence = 0.0
        if state.xy is not None and state.confidence > 0.04:
            output_xy[index] = state.xy
            output_conf[index] = state.confidence
    return output_xy, output_conf, held


def midpoint(
    xy: np.ndarray, confidence: np.ndarray, first: int, second: int, minimum: float = 0.12
) -> np.ndarray | None:
    if confidence[first] >= minimum and confidence[second] >= minimum:
        return (xy[first] + xy[second]) / 2.0
    if confidence[first] >= minimum:
        return xy[first].copy()
    if confidence[second] >= minimum:
        return xy[second].copy()
    return None


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first = a - b
    second = c - b
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1e-6:
        return float("nan")
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def safe_angle(
    xy: np.ndarray, confidence: np.ndarray, indexes: tuple[int, int, int]
) -> float | None:
    if any(confidence[index] < 0.12 for index in indexes):
        return None
    value = joint_angle(xy[indexes[0]], xy[indexes[1]], xy[indexes[2]])
    return None if not math.isfinite(value) else float(value)


def choose_weapon_wrist(
    track: FencerTrack,
    xy: np.ndarray,
    confidence: np.ndarray,
    box: Box,
    side: str,
) -> int | None:
    sign = 1.0 if side == "left" else -1.0
    candidates = []
    for elbow_index, wrist_index in ((7, 9), (8, 10)):
        if confidence[wrist_index] < 0.10 or confidence[elbow_index] < 0.08:
            continue
        inward = sign * float(xy[wrist_index, 0] - box.center[0]) / max(box.width, 1.0)
        extension = float(np.linalg.norm(xy[wrist_index] - xy[elbow_index])) / max(
            box.height, 1.0
        )
        score = (
            1.15 * float(confidence[wrist_index])
            + 0.65 * float(confidence[elbow_index])
            + 0.75 * inward
            + 0.30 * extension
        )
        if track.weapon_wrist_index == wrist_index:
            score += 0.18
        candidates.append((score, wrist_index))
    if not candidates:
        return track.weapon_wrist_index
    score, wrist_index = max(candidates)
    if score < 0.30:
        return track.weapon_wrist_index
    track.weapon_wrist_index = wrist_index
    return wrist_index


def pose_features(
    track: FencerTrack,
    xy: np.ndarray,
    confidence: np.ndarray,
    box: Box,
    side: str,
    fps: float,
    lunge_mode: str = "conservative",
) -> dict[str, object]:
    pelvis = midpoint(xy, confidence, 11, 12)
    shoulders = midpoint(xy, confidence, 5, 6)
    if pelvis is None:
        pelvis = box.center.copy()
    weapon_wrist_index = choose_weapon_wrist(track, xy, confidence, box, side)
    weapon_elbow_index = (
        weapon_wrist_index - 2 if weapon_wrist_index in (9, 10) else None
    )
    weapon_shoulder_index = (
        weapon_wrist_index - 4 if weapon_wrist_index in (9, 10) else None
    )

    sign = 1.0 if side == "left" else -1.0
    track.pelvis_history.append(pelvis.copy())
    if len(track.pelvis_history) > 7:
        track.pelvis_history.pop(0)
    if len(track.pelvis_history) < 3:
        raw_speed = 0.0
    else:
        elapsed_frames = len(track.pelvis_history) - 1
        raw_speed = (
            sign
            * float(track.pelvis_history[-1][0] - track.pelvis_history[0][0])
            / max(box.height, 1.0)
            * fps
            / elapsed_frames
        )
    track.inward_speed = 0.42 * raw_speed + 0.58 * track.inward_speed

    if track.inward_speed > 0.18:
        movement = "advance"
    elif track.inward_speed < -0.18:
        movement = "retreat"
    else:
        movement = "steady"

    stance_ratio = None
    if confidence[15] >= 0.12 and confidence[16] >= 0.12:
        stance_ratio = abs(float(xy[15, 0] - xy[16, 0])) / max(box.height, 1.0)

    left_elbow = safe_angle(xy, confidence, (5, 7, 9))
    right_elbow = safe_angle(xy, confidence, (6, 8, 10))
    left_knee = safe_angle(xy, confidence, (11, 13, 15))
    right_knee = safe_angle(xy, confidence, (12, 14, 16))
    weapon_elbow_angle = None
    if weapon_wrist_index == 9:
        weapon_elbow_angle = left_elbow
    elif weapon_wrist_index == 10:
        weapon_elbow_angle = right_elbow

    torso_lean_deg = None
    if shoulders is not None:
        vector = shoulders - pelvis
        torso_lean_deg = math.degrees(math.atan2(float(vector[0]), -float(vector[1])))

    quality = float(np.mean(confidence >= 0.25))
    valid_knees = [
        float(angle) for angle in (left_knee, right_knee)
        if angle is not None
    ]
    # Provisional fencing-specific rule tuned on a small AI-reviewed
    # single-exchange set. Requiring one bent knee and one relatively
    # extended knee prevents ordinary advancing steps from becoming lunges.
    conservative_lunge = bool(
        stance_ratio is not None
        and stance_ratio >= 0.36
        and track.inward_speed >= 0.02
        and weapon_elbow_angle is not None
        and weapon_elbow_angle >= 110.0
        and len(valid_knees) == 2
        and min(valid_knees) <= 135.0
        and max(valid_knees) >= 145.0
    )
    # Balanced mode recovers fast, low-resolution lunges whose bent-knee
    # angle is biased upward by blur.  High inward speed is deliberately
    # required because ordinary advancing counters otherwise become false
    # lunges.  Conservative remains the default for precision-sensitive use.
    dynamic_lunge = bool(
        lunge_mode == "balanced"
        and stance_ratio is not None
        and stance_ratio >= 0.28
        and track.inward_speed >= 0.85
        and weapon_elbow_angle is not None
        and weapon_elbow_angle >= 130.0
        and len(valid_knees) == 2
        and min(valid_knees) <= 155.0
        and max(valid_knees) >= 140.0
    )
    lunge_candidate = conservative_lunge or dynamic_lunge
    lunge_rule_source = (
        "conservative" if conservative_lunge
        else "balanced_dynamic" if dynamic_lunge
        else None
    )

    return {
        "pelvis": pelvis,
        "shoulders": shoulders,
        "weapon_wrist_index": weapon_wrist_index,
        "weapon_elbow_index": weapon_elbow_index,
        "weapon_shoulder_index": weapon_shoulder_index,
        "inward_speed_body_heights_per_sec": track.inward_speed,
        "movement": movement,
        "stance_ratio": stance_ratio,
        "left_elbow_angle": left_elbow,
        "right_elbow_angle": right_elbow,
        "left_knee_angle": left_knee,
        "right_knee_angle": right_knee,
        "weapon_elbow_angle": weapon_elbow_angle,
        "torso_lean_deg": torso_lean_deg,
        "pose_quality": quality,
        "lunge_candidate": lunge_candidate,
        "lunge_rule_source": lunge_rule_source,
    }


def draw_pose(
    frame: np.ndarray,
    side: str,
    box: Box,
    xy: np.ndarray,
    confidence: np.ndarray,
    held: np.ndarray,
    features: dict[str, object],
) -> None:
    color = (255, 180, 40) if side == "left" else (40, 150, 255)
    muted = tuple(int(value * 0.60) for value in color)
    x1, y1, x2, y2 = np.rint(box.xyxy).astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (130, 130, 130), 1)

    for first, second in SKELETON:
        if confidence[first] >= 0.12 and confidence[second] >= 0.12:
            cv2.line(
                frame,
                tuple(np.rint(xy[first]).astype(int)),
                tuple(np.rint(xy[second]).astype(int)),
                muted if held[first] or held[second] else color,
                3,
                cv2.LINE_AA,
            )
    for index, point in enumerate(xy):
        if confidence[index] < 0.12:
            continue
        center = tuple(np.rint(point).astype(int))
        if held[index]:
            cv2.circle(frame, center, 5, muted, 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, center, 5, color, -1, cv2.LINE_AA)

    weapon_wrist_index = features["weapon_wrist_index"]
    if isinstance(weapon_wrist_index, int) and confidence[weapon_wrist_index] >= 0.10:
        wrist = tuple(np.rint(xy[weapon_wrist_index]).astype(int))
        cv2.circle(frame, wrist, 12, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(
            frame,
            "weapon wrist",
            (wrist[0] + 10, wrist[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    status = str(features["movement"])
    if features["lunge_candidate"]:
        status += " / lunge?"
    cv2.putText(
        frame,
        f"{side}: {status}  v={features['inward_speed_body_heights_per_sec']:+.2f}",
        (x1, max(28, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )


def point_payload(
    xy: np.ndarray, confidence: np.ndarray, held: np.ndarray
) -> dict[str, dict[str, object] | None]:
    payload: dict[str, dict[str, object] | None] = {}
    for index, name in enumerate(KEYPOINT_NAMES):
        if confidence[index] < 0.04:
            payload[name] = None
        else:
            payload[name] = {
                "x": round(float(xy[index, 0]), 2),
                "y": round(float(xy[index, 1]), 2),
                "confidence": round(float(confidence[index]), 4),
                "held": bool(held[index]),
            }
    return payload


def scalar(value: object, digits: int = 4) -> object:
    if value is None:
        return None
    if isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, np.ndarray):
        return [round(float(item), 2) for item in value]
    return round(float(value), digits)


def main() -> None:
    args = parse_args()
    for path in (args.video, args.fencer_model, args.pose_model):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.fallback_person_model and not args.fallback_person_model.exists():
        raise FileNotFoundError(args.fallback_person_model)
    for path in (args.output_video, args.output_json, args.output_csv):
        path.parent.mkdir(parents=True, exist_ok=True)

    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    fencer_model = YOLO(str(args.fencer_model))
    pose_model = YOLO(str(args.pose_model))
    fallback_person_model = (
        YOLO(str(args.fallback_person_model))
        if args.fallback_person_model else None
    )
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(args.video)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    box_hold_frames = max(6, min(20, int(round(0.8 * fps))))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    custom_selection_kwargs = (
        {
            "min_bottom_ratio": 0.45,
            "min_height_ratio": 0.12,
            "min_aspect_ratio": 0.68,
            "min_separation_ratio": 0.08,
            "max_separation_ratio": 0.88,
            # Keep the legacy center-band guard: it rejects a reviewed Tallinn
            # pause shot where a sideline official and one fencer form a pair.
            "min_midpoint_ratio": 0.22,
            "max_midpoint_ratio": 0.78,
            "max_bottom_delta_ratio": 0.26,
            "min_height_balance": 0.34,
        }
        if args.fencer_pair_profile == "multivenue"
        else {}
    )
    preflight_frames = min(total, max(1, int(round(fps))))
    preflight_custom_pairs = 0
    for _ in range(preflight_frames):
        preflight_ok, preflight_frame = cap.read()
        if not preflight_ok:
            break
        preflight_result = fencer_model.predict(
            preflight_frame,
            imgsz=args.fencer_imgsz,
            conf=0.22,
            iou=0.55,
            device=device,
            verbose=False,
        )[0]
        if len(
            select_fencers(
                model_boxes(preflight_result), width, height, **custom_selection_kwargs
            )
        ) == 2:
            preflight_custom_pairs += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fallback_allowed_for_clip = preflight_custom_pairs == 0
    writer = cv2.VideoWriter(
        str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(args.output_video)

    tracks = {"left": FencerTrack(), "right": FencerTrack()}
    fallback_mode = False
    records: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    stats = {
        "frames": 0,
        "frames_with_two_fencer_detections": 0,
        "frames_with_custom_fencer_pair": 0,
        "frames_with_fallback_person_pair": 0,
        "preflight_custom_pair_frames": preflight_custom_pairs,
        "fallback_allowed_for_clip": fallback_allowed_for_clip,
        "left_pose_frames": 0,
        "right_pose_frames": 0,
        "left_good_pose_frames": 0,
        "right_good_pose_frames": 0,
    }

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames is not None and frame_index >= args.max_frames):
            break
        result = fencer_model.predict(
            frame,
            imgsz=args.fencer_imgsz,
            conf=0.22,
            iou=0.55,
            device=device,
            verbose=False,
        )[0]
        custom_selected = select_fencers(
            model_boxes(result), width, height, **custom_selection_kwargs
        )
        detection_source = "held"
        if len(custom_selected) == 2 and pair_temporally_plausible(
            custom_selected, tracks, width, box_hold_frames
        ):
            selected = custom_selected
            detection_source = "custom"
            fallback_mode = False
            stats["frames_with_custom_fencer_pair"] += 1
        else:
            selected = []
        tracks_uninitialized = any(track.box is None for track in tracks.values())
        allow_fallback = (
            fallback_person_model is not None
            and fallback_allowed_for_clip
            and len(selected) != 2
            and (
                tracks_uninitialized
                or fallback_mode
            )
        )
        if allow_fallback:
            fallback_result = fallback_person_model.predict(
                frame,
                imgsz=args.fallback_imgsz,
                conf=0.10,
                iou=0.55,
                device=device,
                verbose=False,
            )[0]
            selected = select_fencers(
                model_boxes(fallback_result), width, height,
                # Generic COCO Pose readily detects printed fencers.  Requiring
                # both candidates to reach the lower piste band rejects the
                # large Tunis wall graphics while retaining the reviewed
                # high-angle Genova pair (box bottoms remain >= 0.60 H).
                min_bottom_ratio=0.59,
                # Foreground referees/spectators sit below the piste band and
                # can otherwise pair with one real fencer in wide broadcasts.
                max_bottom_ratio=0.88,
                min_height_ratio=0.07,
                min_aspect_ratio=0.55,
            )
            if len(selected) == 2 and pair_temporally_plausible(
                selected, tracks, width, box_hold_frames
            ):
                detection_source = "fallback"
                fallback_mode = True
                stats["frames_with_fallback_person_pair"] += 1
            else:
                selected = []
        if len(selected) == 2:
            stats["frames_with_two_fencer_detections"] += 1
        detected_by_side = {
            "left": selected[0] if len(selected) == 2 else None,
            "right": selected[1] if len(selected) == 2 else None,
        }
        frame_record: dict[str, object] = {
            "frame": frame_index,
            "time_sec": round(frame_index / fps, 4),
            "detection_source": detection_source,
        }
        frame_features: dict[str, dict[str, object]] = {}

        for side in ("left", "right"):
            track = tracks[side]
            box = update_box(
                track, detected_by_side[side], box_hold_frames
            )
            if box is None:
                xy, confidence, held = update_points(track, None)
                frame_record[side] = {"box": None, "keypoints": point_payload(xy, confidence, held)}
                continue
            measurement = detect_pose(
                pose_model, frame, box, args.pose_imgsz, device
            )
            xy, confidence, held = update_points(track, measurement)
            features = pose_features(
                track, xy, confidence, box, side, fps, args.lunge_mode
            )
            frame_features[side] = features
            if measurement is not None:
                stats[f"{side}_pose_frames"] += 1
            if features["pose_quality"] >= 0.60:
                stats[f"{side}_good_pose_frames"] += 1
            draw_pose(frame, side, box, xy, confidence, held, features)
            frame_record[side] = {
                "box": [round(float(value), 2) for value in box.xyxy],
                "box_confidence": round(float(box.confidence), 4),
                "keypoints": point_payload(xy, confidence, held),
                "features": {
                    key: scalar(value)
                    for key, value in features.items()
                    if key not in {"weapon_wrist_index", "weapon_elbow_index", "weapon_shoulder_index"}
                },
                "weapon_wrist": KEYPOINT_NAMES[features["weapon_wrist_index"]]
                if isinstance(features["weapon_wrist_index"], int)
                else None,
            }

        distance = None
        if "left" in frame_features and "right" in frame_features:
            left_pelvis = frame_features["left"]["pelvis"]
            right_pelvis = frame_features["right"]["pelvis"]
            left_box = tracks["left"].box
            right_box = tracks["right"].box
            if (
                isinstance(left_pelvis, np.ndarray)
                and isinstance(right_pelvis, np.ndarray)
                and left_box is not None
                and right_box is not None
            ):
                average_height = (left_box.height + right_box.height) / 2.0
                distance = abs(float(right_pelvis[0] - left_pelvis[0])) / max(
                    average_height, 1.0
                )
        frame_record["opponent_distance_body_heights"] = scalar(distance)

        feature_row: dict[str, object] = {
            "frame": frame_index,
            "time_sec": round(frame_index / fps, 4),
            "opponent_distance_body_heights": "" if distance is None else round(distance, 4),
        }
        for side in ("left", "right"):
            features = frame_features.get(side)
            for key in (
                "movement",
                "inward_speed_body_heights_per_sec",
                "stance_ratio",
                "weapon_elbow_angle",
                "left_knee_angle",
                "right_knee_angle",
                "torso_lean_deg",
                "pose_quality",
                "lunge_candidate",
                "lunge_rule_source",
            ):
                value = None if features is None else features.get(key)
                feature_row[f"{side}_{key}"] = "" if value is None else scalar(value)
        feature_rows.append(feature_row)

        cv2.rectangle(frame, (0, 0), (1050, 42), (0, 0, 0), -1)
        cv2.putText(
            frame,
            (
                "Fencing Pose v2: 17 keypoints + temporal smoothing | "
                f"lunge={args.lunge_mode} | yellow ring = weapon wrist"
            ),
            (12, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if distance is not None:
            cv2.putText(
                frame,
                f"distance={distance:.2f} body-heights",
                (width - 430, 29),
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
        if frame_index % 50 == 0:
            print(f"processed={frame_index}/{total}", flush=True)

    cap.release()
    writer.release()
    payload = {
        "video": str(args.video),
        "fencer_model": str(args.fencer_model),
        "pose_model": str(args.pose_model),
        "device": str(device),
        "fps": fps,
        "width": width,
        "height": height,
        "keypoint_names": KEYPOINT_NAMES,
        "movement_labels_are_experimental": True,
        "lunge_mode": args.lunge_mode,
        "fallback_person_model": (
            str(args.fallback_person_model)
            if args.fallback_person_model else None
        ),
        "stats": stats,
        "frames": records,
    }
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(feature_rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(feature_rows)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    print(f"output_video={args.output_video}", flush=True)
    print(f"output_json={args.output_json}", flush=True)
    print(f"output_csv={args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
