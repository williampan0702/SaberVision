from __future__ import annotations

from dataclasses import asdict, dataclass

from evaluate_action_rules import evaluate_clip, find_exchange_phase, side_features
from evaluate_phase_lunge_fallback import highspeed_event_flag
from search_phase_lunge_fallback import phase_features
from summarize_action_sequence_v2 import BALANCED


@dataclass(frozen=True)
class WindowThresholds:
    window_frames: int = 4
    stance: float = 0.40
    inward_speed: float = 0.65
    weapon_elbow: float = 135.0
    front_knee_max: float = 155.0
    rear_knee_min: float = 145.0
    leg_coherence_stance: float = 0.28
    leg_coherence_front_knee_max: float = 160.0
    leg_coherence_rear_knee_min: float = 140.0
    minimum_inward_net: float = 0.20
    high_local_speed_override: float = 0.85


DEFAULT_THRESHOLDS = WindowThresholds()


def _number(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _front_and_rear_knees(frame: dict, side: str, feat: dict):
    record = frame.get(side)
    if not isinstance(record, dict):
        return None, None
    keypoints = record.get("keypoints")
    if not isinstance(keypoints, dict):
        return None, None
    left_ankle = keypoints.get("left_ankle")
    right_ankle = keypoints.get("right_ankle")
    if not isinstance(left_ankle, dict) or not isinstance(right_ankle, dict):
        return None, None
    left_x = _number(left_ankle.get("x"))
    right_x = _number(right_ankle.get("x"))
    left_knee = _number(feat.get("left_knee_angle"))
    right_knee = _number(feat.get("right_knee_angle"))
    if None in (left_x, right_x, left_knee, right_knee):
        return None, None
    # The left-side athlete attacks towards increasing x; the right-side
    # athlete attacks towards decreasing x.  Anatomical left/right labels can
    # flip with the camera, so use ankle position to identify the front leg.
    front_is_left = left_x >= right_x if side == "left" else left_x <= right_x
    if front_is_left:
        return left_knee, right_knee
    return right_knee, left_knee


def phase_samples(data: dict, side: str) -> list[dict]:
    start_frame, contact_frame = find_exchange_phase(data)
    samples = []
    for frame in data.get("frames", []):
        frame_no = int(frame.get("frame", -1))
        if start_frame is not None and frame_no < start_frame:
            continue
        if contact_frame is not None and frame_no > contact_frame + 3:
            continue
        feat = side_features(frame, side)
        if not isinstance(feat, dict):
            continue
        front_knee, rear_knee = _front_and_rear_knees(frame, side, feat)
        samples.append({
            "frame": frame_no,
            "stance": _number(feat.get("stance_ratio")),
            "inward_speed": _number(
                feat.get("inward_speed_body_heights_per_sec")
            ),
            "weapon_elbow": _number(feat.get("weapon_elbow_angle")),
            "front_knee": front_knee,
            "rear_knee": rear_knee,
        })
    return samples


def _maximum(window: list[dict], key: str, default: float = 0.0) -> float:
    values = [sample[key] for sample in window if sample.get(key) is not None]
    return max(values) if values else default


def _minimum(window: list[dict], key: str, default: float = 180.0) -> float:
    values = [sample[key] for sample in window if sample.get(key) is not None]
    return min(values) if values else default


def temporal_window_evidence(
    data: dict,
    side: str,
    thresholds: WindowThresholds = DEFAULT_THRESHOLDS,
) -> dict:
    samples = phase_samples(data, side)
    event = phase_features(data, side)
    best = None
    width = max(2, int(thresholds.window_frames))
    for start in range(len(samples)):
        window = samples[start:start + width]
        if len(window) < 2:
            continue
        maxima = {
            "stance": _maximum(window, "stance"),
            "inward_speed": _maximum(window, "inward_speed"),
            "weapon_elbow": _maximum(window, "weapon_elbow"),
            "rear_knee": _maximum(window, "rear_knee"),
        }
        minima = {"front_knee": _minimum(window, "front_knee")}
        flags = {
            "stance": maxima["stance"] >= thresholds.stance,
            "inward_speed": (
                maxima["inward_speed"] >= thresholds.inward_speed
            ),
            "weapon_elbow": (
                maxima["weapon_elbow"] >= thresholds.weapon_elbow
            ),
            "front_knee": (
                minima["front_knee"] <= thresholds.front_knee_max
            ),
            "rear_knee": maxima["rear_knee"] >= thresholds.rear_knee_min,
        }
        leg_coherent = any(
            sample.get("stance") is not None
            and sample.get("front_knee") is not None
            and sample.get("rear_knee") is not None
            and sample["stance"] >= thresholds.leg_coherence_stance
            and sample["front_knee"] <= thresholds.leg_coherence_front_knee_max
            and sample["rear_knee"] >= thresholds.leg_coherence_rear_knee_min
            for sample in window
        )
        inward_net = float(event.get("inward_net") or 0.0)
        direction_supported = bool(
            inward_net >= thresholds.minimum_inward_net
            or maxima["inward_speed"] >= thresholds.high_local_speed_override
        )
        component_count = sum(bool(value) for value in flags.values())
        supported = bool(
            component_count == len(flags)
            and leg_coherent
            and direction_supported
        )
        candidate = {
            "start_frame": window[0]["frame"],
            "end_frame": window[-1]["frame"],
            "component_count": component_count,
            "component_flags": flags,
            "leg_coherent_in_one_frame": leg_coherent,
            "direction_supported": direction_supported,
            "supported": supported,
            "maxima": {key: round(value, 4) for key, value in maxima.items()},
            "minima": {key: round(value, 4) for key, value in minima.items()},
        }
        rank = (
            int(supported), component_count, int(leg_coherent),
            maxima["inward_speed"], maxima["stance"],
        )
        if best is None or rank > best[0]:
            best = (rank, candidate)
    return {
        "supported": bool(best and best[1]["supported"]),
        "best_window": best[1] if best else None,
        "samples": len(samples),
        "thresholds": asdict(thresholds),
    }


def classify_lunge_evidence(
    data: dict,
    side: str,
    confirmed_stance_threshold: float = 0.36,
) -> dict:
    base = evaluate_clip(
        data=data,
        side=side,
        movement_min_votes=3,
        lunge_min_run=1,
        method="phase_net",
        net_movement_threshold=0.10,
        lunge_source="leg_geometry",
        stance_threshold=confirmed_stance_threshold,
        speed_threshold=0.02,
        elbow_threshold=110.0,
        bent_knee_max=135.0,
        rear_knee_min=145.0,
        **BALANCED,
    )
    event = phase_features(data, side)
    window = temporal_window_evidence(data, side)
    confirmed = bool(base["predicted_lunge"])
    highspeed = bool(highspeed_event_flag(event))
    if confirmed:
        state = "confirmed_geometry"
        lunge_present = True
        requires_resolution = False
    elif window["supported"]:
        state = "supported_window_geometry"
        lunge_present = None
        requires_resolution = True
    elif highspeed:
        state = "possible_highspeed_event"
        lunge_present = None
        requires_resolution = True
    else:
        state = "none"
        lunge_present = False
        requires_resolution = False
    return {
        "state": state,
        "lunge_present": lunge_present,
        "confirmed_geometry": confirmed,
        "supported_window_geometry": bool(window["supported"]),
        "possible_highspeed_event": highspeed,
        "requires_downstream_resolution": requires_resolution,
        "base": base,
        "event": event,
        "window": window,
    }
