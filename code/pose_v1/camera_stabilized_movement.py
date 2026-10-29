from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraMotionConfig:
    max_corners: int = 350
    quality_level: float = 0.01
    min_distance: float = 8.0
    ransac_threshold: float = 2.5
    minimum_inliers: int = 10
    maximum_translation_ratio: float = 0.18
    minimum_scale: float = 0.86
    maximum_scale: float = 1.16
    # Broadcast score overlays are frequently locked to the top of the screen
    # while the actual camera pans.  Excluding that band prevents the overlay
    # from being mistaken for the stationary arena background.
    top_mask_ratio: float = 0.18
    bottom_mask_ratio: float = 0.82
    fencer_mask_padding_ratio: float = 0.10


DEFAULT_CONFIG = CameraMotionConfig()


def _number(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _features(frame: dict, side: str) -> dict | None:
    record = frame.get(side)
    if not isinstance(record, dict):
        return None
    features = record.get("features")
    return features if isinstance(features, dict) else None


def _pelvis(frame: dict, side: str) -> np.ndarray | None:
    features = _features(frame, side)
    value = features.get("pelvis") if features else None
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x, y = _number(value[0]), _number(value[1])
    if x is None or y is None:
        return None
    return np.asarray([x, y, 1.0], dtype=np.float64)


def _box(frame: dict, side: str) -> list[float] | None:
    record = frame.get(side)
    value = record.get("box") if isinstance(record, dict) else None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    numbers = [_number(item) for item in value]
    return None if any(item is None for item in numbers) else numbers


def _feature_mask(
    shape: tuple[int, int],
    previous_record: dict,
    current_record: dict,
    config: CameraMotionConfig,
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    y1 = max(0, int(round(config.top_mask_ratio * height)))
    y2 = min(height, int(round(config.bottom_mask_ratio * height)))
    mask[y1:y2, :] = 255
    for record in (previous_record, current_record):
        for side in ("left", "right"):
            box = _box(record, side)
            if box is None:
                continue
            x1, top, x2, bottom = box
            pad_x = config.fencer_mask_padding_ratio * max(1.0, x2 - x1)
            pad_y = config.fencer_mask_padding_ratio * max(1.0, bottom - top)
            left = max(0, int(round(x1 - pad_x)))
            right = min(width, int(round(x2 + pad_x)))
            upper = max(0, int(round(top - pad_y)))
            lower = min(height, int(round(bottom + pad_y)))
            mask[upper:lower, left:right] = 0
    return mask


def estimate_background_affine(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_record: dict,
    current_record: dict,
    config: CameraMotionConfig = DEFAULT_CONFIG,
) -> tuple[np.ndarray, int, int] | None:
    height, width = previous_gray.shape[:2]
    mask = _feature_mask(
        (height, width), previous_record, current_record, config
    )
    points = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=config.max_corners,
        qualityLevel=config.quality_level,
        minDistance=config.min_distance,
        mask=mask,
        blockSize=5,
    )
    if points is None or len(points) < config.minimum_inliers:
        return None
    followed, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None,
        winSize=(25, 25),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )
    if followed is None or status is None:
        return None
    valid = status.reshape(-1).astype(bool)
    source = points.reshape(-1, 2)[valid]
    target = followed.reshape(-1, 2)[valid]
    if len(source) < config.minimum_inliers:
        return None
    affine, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=config.ransac_threshold,
        maxIters=2000,
        confidence=0.995,
        refineIters=10,
    )
    if affine is None or inliers is None:
        return None
    inlier_count = int(inliers.sum())
    if inlier_count < config.minimum_inliers:
        return None
    scale = float(np.hypot(affine[0, 0], affine[0, 1]))
    translation = float(np.hypot(affine[0, 2], affine[1, 2]))
    if (
        not config.minimum_scale <= scale <= config.maximum_scale
        or translation > config.maximum_translation_ratio * width
    ):
        return None
    return affine.astype(np.float64), inlier_count, len(source)


def stabilized_movement_features(
    data: dict,
    pose_json_path: Path | None = None,
    start_frame: int | None = None,
    contact_frame: int | None = None,
    config: CameraMotionConfig = DEFAULT_CONFIG,
) -> dict:
    video_value = data.get("video")
    if not isinstance(video_value, str) or not video_value:
        return {"status": "missing_video_path", "config": asdict(config)}
    video_path = Path(video_value)
    if not video_path.is_absolute() and pose_json_path is not None:
        workspace_candidate = Path.cwd() / video_path
        pose_candidate = pose_json_path.parent / video_path
        video_path = (
            workspace_candidate
            if workspace_candidate.exists()
            else pose_candidate
        )
    if not video_path.exists():
        return {
            "status": "video_not_found",
            "video": str(video_path),
            "config": asdict(config),
        }

    records = {
        int(frame.get("frame", -1)): frame
        for frame in data.get("frames", [])
        if isinstance(frame, dict)
    }
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {
            "status": "video_open_failed",
            "video": str(video_path),
            "config": asdict(config),
        }

    per_side = {"left": [], "right": []}
    attempted_transforms = 0
    accepted_transforms = 0
    inlier_ratios: list[float] = []
    previous_gray = None
    previous_record = None
    frame_index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        record = records.get(frame_index)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if (
            previous_gray is not None
            and previous_record is not None
            and record is not None
            and (start_frame is None or frame_index > start_frame)
            and (contact_frame is None or frame_index <= contact_frame)
        ):
            attempted_transforms += 1
            estimated = estimate_background_affine(
                previous_gray,
                gray,
                previous_record,
                record,
                config,
            )
            if estimated is not None:
                affine, inliers, tracked = estimated
                accepted_transforms += 1
                inlier_ratios.append(inliers / max(tracked, 1))
                for side in ("left", "right"):
                    old_pelvis = _pelvis(previous_record, side)
                    new_pelvis = _pelvis(record, side)
                    old_box = _box(previous_record, side)
                    new_box = _box(record, side)
                    if (
                        old_pelvis is None
                        or new_pelvis is None
                        or old_box is None
                        or new_box is None
                    ):
                        continue
                    predicted = affine @ old_pelvis
                    body_height = np.median(
                        [old_box[3] - old_box[1], new_box[3] - new_box[1]]
                    )
                    if body_height <= 1.0:
                        continue
                    screen_residual = (
                        float(new_pelvis[0]) - float(predicted[0])
                    ) / float(body_height)
                    inward = screen_residual if side == "left" else -screen_residual
                    per_side[side].append(inward)
        previous_gray = gray
        previous_record = record
        frame_index += 1
    capture.release()

    result = {
        "status": "ok" if accepted_transforms else "insufficient_background_tracks",
        "video": str(video_path),
        "attempted_transforms": attempted_transforms,
        "accepted_transforms": accepted_transforms,
        "transform_coverage": round(
            accepted_transforms / max(attempted_transforms, 1), 4
        ),
        "median_inlier_ratio": round(
            float(np.median(inlier_ratios)) if inlier_ratios else 0.0, 4
        ),
        "config": asdict(config),
        "athletes": {},
    }
    fps = max(1.0, float(data.get("fps") or 1.0))
    for side, increments in per_side.items():
        array = np.asarray(increments, dtype=np.float64)
        if array.size:
            # One bad local registration must not decide a whole exchange.
            lower, upper = np.quantile(array, [0.05, 0.95])
            clipped = np.clip(array, lower, upper)
            net = float(clipped.sum())
            median_speed = float(np.median(clipped) * fps)
            peak_speed = float(np.max(clipped) * fps)
        else:
            net = median_speed = peak_speed = 0.0
        result["athletes"][side] = {
            "inward_net_body_heights": round(net, 4),
            "median_inward_speed_body_heights_per_sec": round(
                median_speed, 4
            ),
            "peak_inward_speed_body_heights_per_sec": round(peak_speed, 4),
            "sample_intervals": int(array.size),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate fencer movement after background camera compensation."
    )
    parser.add_argument("pose_json", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {}
    for path in args.pose_json:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Import lazily to keep this module usable by the Pose pipeline.
        from evaluate_action_rules import find_exchange_phase

        start_frame, contact_frame = find_exchange_phase(data)
        payload[path.stem] = stabilized_movement_features(
            data,
            pose_json_path=path,
            start_frame=start_frame,
            contact_frame=contact_frame,
        )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
