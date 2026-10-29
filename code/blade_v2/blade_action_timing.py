#!/usr/bin/env python3
"""Extract conservative fencing timing candidates from synchronized Pose/blade JSON.

This module deliberately separates three concepts:

* Pose can support an ``attack_extension_start`` candidate.
* A directly observed blade tip can support a ``tip_approach`` candidate.
* Two directly observed blade segments can support blade-proximity and possible
  parry candidates.

Temporal-held blade lines are useful for visualization, but are never accepted
as contact or parry evidence here.  The output is evidence for a later fencing
decision layer, not an automatic priority decision.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional


SIDES = ("left", "right")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def point_to_rect_distance(point: tuple[float, float], box: Iterable[float]) -> float:
    """Euclidean distance from a point to an axis-aligned rectangle (zero inside)."""
    x, y = point
    x1, y1, x2, y2 = [float(value) for value in box]
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return math.hypot(dx, dy)


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: tuple[float, float], b: tuple[float, float], p: tuple[float, float]) -> bool:
    eps = 1e-9
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
        and abs(_orientation(a, b, p)) <= eps
    )


def _segments_intersect(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]
) -> bool:
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    return (
        (abs(o1) <= 1e-9 and _on_segment(a, b, c))
        or (abs(o2) <= 1e-9 and _on_segment(a, b, d))
        or (abs(o3) <= 1e-9 and _on_segment(c, d, a))
        or (abs(o4) <= 1e-9 and _on_segment(c, d, b))
    )


def _point_to_segment_distance(
    p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> float:
    vx, vy = b[0] - a[0], b[1] - a[1]
    denom = vx * vx + vy * vy
    if denom <= 1e-12:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / denom
    t = max(0.0, min(1.0, t))
    qx, qy = a[0] + t * vx, a[1] + t * vy
    return math.hypot(p[0] - qx, p[1] - qy)


def segment_distance(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]
) -> float:
    """Minimum distance between two closed 2D line segments."""
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(
        _point_to_segment_distance(a, c, d),
        _point_to_segment_distance(b, c, d),
        _point_to_segment_distance(c, a, b),
        _point_to_segment_distance(d, a, b),
    )


def angle_delta_degrees(a: float, b: float) -> float:
    """Smallest undirected line-angle difference in degrees (0..90)."""
    delta = abs(float(a) - float(b)) % 180.0
    return min(delta, 180.0 - delta)


def nearest_pose_frame(pose_frames: list[dict[str, Any]], time_sec: float, max_gap: float = 0.16) -> Optional[dict[str, Any]]:
    if not pose_frames:
        return None
    result = min(pose_frames, key=lambda frame: abs(float(frame["time_sec"]) - time_sec))
    return result if abs(float(result["time_sec"]) - time_sec) <= max_gap else None


def _body_height(frame: dict[str, Any]) -> Optional[float]:
    heights = []
    for side in SIDES:
        record = frame.get(side)
        if record and record.get("box"):
            box = record["box"]
            heights.append(max(1.0, float(box[3]) - float(box[1])))
    return sum(heights) / len(heights) if heights else None


def _extension_condition(frame: dict[str, Any], side: str) -> bool:
    record = frame.get(side) or {}
    features = record.get("features") or {}
    speed = features.get("inward_speed_body_heights_per_sec")
    elbow = features.get("weapon_elbow_angle")
    movement = features.get("movement")
    return (
        speed is not None
        and elbow is not None
        and float(speed) >= 0.18
        and float(elbow) >= 140.0
        and movement == "advance"
    )


def detect_attack_extension(pose_frames: list[dict[str, Any]], side: str) -> dict[str, Any]:
    """Find first two-frame sustained extension/advance run.

    A short speed rise is required to avoid labeling a continuously extended,
    stationary guard position as attack initiation.
    """
    for index in range(len(pose_frames) - 1):
        if not (_extension_condition(pose_frames[index], side) and _extension_condition(pose_frames[index + 1], side)):
            continue
        current = pose_frames[index]
        lookback = pose_frames[max(0, index - 4)]
        current_features = current[side]["features"]
        old_features = (lookback.get(side) or {}).get("features") or {}
        speed_now = float(current_features["inward_speed_body_heights_per_sec"])
        speed_old = float(old_features.get("inward_speed_body_heights_per_sec", speed_now))
        if speed_now - speed_old < 0.10:
            continue
        end_index = index + 1
        while end_index + 1 < len(pose_frames) and _extension_condition(pose_frames[end_index + 1], side):
            end_index += 1
        evidence = []
        for frame in pose_frames[index : min(index + 3, end_index + 1)]:
            features = frame[side]["features"]
            evidence.append(
                {
                    "frame": frame["frame"],
                    "time_sec": frame["time_sec"],
                    "movement": features.get("movement"),
                    "inward_speed_body_heights_per_sec": features.get("inward_speed_body_heights_per_sec"),
                    "weapon_elbow_angle": features.get("weapon_elbow_angle"),
                }
            )
        return {
            "state": "candidate",
            "start_sec": current["time_sec"],
            "end_sec": pose_frames[end_index]["time_sec"],
            "basis": "two_or_more_pose_samples_show_sustained_advance_speed_and_extended_weapon_arm",
            "evidence": evidence,
        }
    return {
        "state": "not_observed",
        "start_sec": None,
        "end_sec": None,
        "basis": "pose_sequence_did_not_meet_sustained_extension_rule",
        "evidence": [],
    }


def group_times(times: list[float], max_gap: float) -> list[dict[str, Any]]:
    if not times:
        return []
    ordered = sorted(times)
    groups: list[list[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value - groups[-1][-1] <= max_gap:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [
        {"start_sec": round(group[0], 4), "end_sec": round(group[-1], 4), "observations": len(group)}
        for group in groups
    ]


def _direct_segment(frame: dict[str, Any], side: str) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
    record = frame.get(side) or {}
    if record.get("source") != "geometry" or not record.get("guard") or not record.get("tip"):
        return None
    guard = tuple(float(v) for v in record["guard"])
    tip = tuple(float(v) for v in record["tip"])
    return guard, tip


def collect_blade_visibility(blade_frames: list[dict[str, Any]], fps: float) -> dict[str, Any]:
    max_gap = max(0.12, 2.1 / max(fps, 1.0))
    result: dict[str, Any] = {}
    for side in SIDES:
        direct = [float(frame["time_sec"]) for frame in blade_frames if (frame.get(side) or {}).get("source") == "geometry"]
        held = [float(frame["time_sec"]) for frame in blade_frames if (frame.get(side) or {}).get("source") == "temporal"]
        result[side] = {
            "direct_geometry_intervals": group_times(direct, max_gap),
            "temporal_hold_intervals": group_times(held, max_gap),
            "direct_geometry_observations": len(direct),
            "temporal_hold_observations": len(held),
        }
    return result


def detect_tip_approach(
    pose_frames: list[dict[str, Any]], blade_frames: list[dict[str, Any]], threshold_body_heights: float
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    for blade_frame in blade_frames:
        time_sec = float(blade_frame["time_sec"])
        pose_frame = nearest_pose_frame(pose_frames, time_sec)
        if pose_frame is None:
            continue
        for side in SIDES:
            segment = _direct_segment(blade_frame, side)
            opponent = "right" if side == "left" else "left"
            opponent_record = pose_frame.get(opponent)
            if segment is None or not opponent_record or not opponent_record.get("box"):
                continue
            guard, tip = segment
            toward_opponent = tip[0] > guard[0] if side == "left" else tip[0] < guard[0]
            box = opponent_record["box"]
            body_height = max(1.0, float(box[3]) - float(box[1]))
            normalized_distance = point_to_rect_distance(tip, box) / body_height
            if toward_opponent and normalized_distance <= threshold_body_heights:
                result[side].append(
                    {
                        "frame": blade_frame["frame"],
                        "time_sec": blade_frame["time_sec"],
                        "tip_to_opponent_box_body_heights": round(normalized_distance, 4),
                        "inside_opponent_box": normalized_distance == 0.0,
                        "blade_source": "geometry",
                        "pose_frame": pose_frame["frame"],
                    }
                )
    return result


def detect_blade_proximity(
    pose_frames: list[dict[str, Any]], blade_frames: list[dict[str, Any]], threshold_body_heights: float
) -> list[dict[str, Any]]:
    events = []
    for blade_frame in blade_frames:
        left = _direct_segment(blade_frame, "left")
        right = _direct_segment(blade_frame, "right")
        if left is None or right is None:
            continue
        pose_frame = nearest_pose_frame(pose_frames, float(blade_frame["time_sec"]))
        body_height = _body_height(pose_frame) if pose_frame else None
        if body_height is None:
            continue
        distance = segment_distance(left[0], left[1], right[0], right[1]) / body_height
        if distance <= threshold_body_heights:
            events.append(
                {
                    "frame": blade_frame["frame"],
                    "time_sec": blade_frame["time_sec"],
                    "segment_distance_body_heights": round(distance, 4),
                    "blade_source": "geometry_both_sides",
                }
            )
    return events


def detect_possible_parries(
    blade_frames: list[dict[str, Any]], proximity_events: list[dict[str, Any]], angle_change_threshold: float = 18.0
) -> list[dict[str, Any]]:
    direct_by_side: dict[str, list[dict[str, Any]]] = {
        side: [frame for frame in blade_frames if _direct_segment(frame, side) is not None] for side in SIDES
    }
    candidates = []
    for event in proximity_events:
        event_time = float(event["time_sec"])
        changes: dict[str, float] = {}
        for side in SIDES:
            current_options = [
                frame for frame in direct_by_side[side] if abs(float(frame["time_sec"]) - event_time) <= 0.06
            ]
            future_options = [
                frame
                for frame in direct_by_side[side]
                if 0.04 <= float(frame["time_sec"]) - event_time <= 0.40
            ]
            if not current_options or not future_options:
                continue
            current = min(current_options, key=lambda frame: abs(float(frame["time_sec"]) - event_time))
            future = min(future_options, key=lambda frame: float(frame["time_sec"]))
            current_angle = (current.get(side) or {}).get("angle_deg")
            future_angle = (future.get(side) or {}).get("angle_deg")
            if current_angle is not None and future_angle is not None:
                changes[side] = round(angle_delta_degrees(float(current_angle), float(future_angle)), 2)
        changed = [side for side, value in changes.items() if value >= angle_change_threshold]
        if changed:
            candidates.append(
                {
                    "time_sec": event["time_sec"],
                    "state": "possible_parry_candidate",
                    "changed_side": changed,
                    "angle_change_deg": changes,
                    "basis": "direct_two_blade_proximity_followed_by_direct_blade_direction_change",
                }
            )
    return candidates


def flatten_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for side in SIDES:
        extension = result["attack_extension_start"][side]
        rows.append(
            {
                "evidence_type": "attack_extension_start",
                "side": side,
                "state": extension["state"],
                "start_sec": extension["start_sec"],
                "end_sec": extension["end_sec"],
                "detail": extension["basis"],
            }
        )
        for event in result["tip_approach_candidates"][side]:
            rows.append(
                {
                    "evidence_type": "tip_approach_candidate",
                    "side": side,
                    "state": "candidate",
                    "start_sec": event["time_sec"],
                    "end_sec": event["time_sec"],
                    "detail": f"tip_distance={event['tip_to_opponent_box_body_heights']} body_heights",
                }
            )
    for event in result["blade_proximity_candidates"]:
        rows.append(
            {
                "evidence_type": "blade_proximity_candidate",
                "side": "both",
                "state": "candidate",
                "start_sec": event["time_sec"],
                "end_sec": event["time_sec"],
                "detail": f"segment_distance={event['segment_distance_body_heights']} body_heights",
            }
        )
    for event in result["possible_parry_candidates"]:
        rows.append(
            {
                "evidence_type": "possible_parry_candidate",
                "side": "+".join(event["changed_side"]),
                "state": event["state"],
                "start_sec": event["time_sec"],
                "end_sec": event["time_sec"],
                "detail": json.dumps(event["angle_change_deg"], ensure_ascii=False),
            }
        )
    return rows


def analyze(
    pose_data: dict[str, Any], blade_data: dict[str, Any], pose_time_offset_sec: float = 0.0
) -> dict[str, Any]:
    # Pose clips are sometimes trimmed/re-encoded excerpts of the blade source
    # video. Work on a copy and place their local timestamps on the source clock.
    pose_frames = copy.deepcopy(pose_data.get("frames", []))
    for frame in pose_frames:
        frame["clip_local_time_sec"] = frame.get("time_sec")
        frame["time_sec"] = round(float(frame.get("time_sec", 0.0)) + pose_time_offset_sec, 6)
    blade_frames = blade_data.get("frames", [])
    blade_fps = float(blade_data.get("fps") or 25.0)
    synchronized_blade_frames = [
        frame for frame in blade_frames
        if nearest_pose_frame(pose_frames, float(frame.get("time_sec", -1e9))) is not None
    ]
    tip_threshold = 0.35
    blade_threshold = 0.12
    tip_approach = detect_tip_approach(pose_frames, blade_frames, tip_threshold)
    blade_proximity = detect_blade_proximity(pose_frames, blade_frames, blade_threshold)
    parries = detect_possible_parries(blade_frames, blade_proximity)
    return {
        "schema": "sabervision_blade_action_timing_v2",
        "inputs": {
            "pose_video": pose_data.get("video"),
            "blade_video": blade_data.get("video"),
            "pose_model": pose_data.get("pose_model"),
            "fencer_model": blade_data.get("fencer_model"),
            "pose_time_offset_sec": pose_time_offset_sec,
        },
        "policy": {
            "outputs_are_candidates_not_referee_decisions": True,
            "temporal_blade_holds_allowed_for_contact_or_parry": False,
            "tip_approach_threshold_body_heights": tip_threshold,
            "blade_proximity_threshold_body_heights": blade_threshold,
        },
        "attack_extension_start": {
            side: detect_attack_extension(pose_frames, side) for side in SIDES
        },
        "blade_visibility": collect_blade_visibility(blade_frames, blade_fps),
        "tip_approach_candidates": tip_approach,
        "blade_proximity_candidates": blade_proximity,
        "possible_contact_windows": [
            {
                "side": side,
                "time_sec": event["time_sec"],
                "state": "possible_contact_candidate",
                "basis": "direct_tip_near_opponent_box",
            }
            for side in SIDES
            for event in tip_approach[side]
        ],
        "possible_parry_candidates": parries,
        "summary": {
            "left_extension_observed": detect_attack_extension(pose_frames, "left")["state"] == "candidate",
            "right_extension_observed": detect_attack_extension(pose_frames, "right")["state"] == "candidate",
            "direct_blade_geometry_frames_left": sum(
                1 for frame in blade_frames if (frame.get("left") or {}).get("source") == "geometry"
            ),
            "direct_blade_geometry_frames_right": sum(
                1 for frame in blade_frames if (frame.get("right") or {}).get("source") == "geometry"
            ),
            "synchronized_window_start_sec": (
                min((float(frame["time_sec"]) for frame in pose_frames), default=None)
            ),
            "synchronized_window_end_sec": (
                max((float(frame["time_sec"]) for frame in pose_frames), default=None)
            ),
            "synchronized_direct_blade_frames_left": sum(
                1 for frame in synchronized_blade_frames
                if (frame.get("left") or {}).get("source") == "geometry"
            ),
            "synchronized_direct_blade_frames_right": sum(
                1 for frame in synchronized_blade_frames
                if (frame.get("right") or {}).get("source") == "geometry"
            ),
            "tip_distance_usable_direct_blade_frames_left": sum(
                1 for frame in synchronized_blade_frames
                if (frame.get("left") or {}).get("source") == "geometry"
                and (
                    (nearest_pose_frame(pose_frames, float(frame["time_sec"])) or {}).get("right")
                    or {}
                ).get("box")
            ),
            "tip_distance_usable_direct_blade_frames_right": sum(
                1 for frame in synchronized_blade_frames
                if (frame.get("right") or {}).get("source") == "geometry"
                and (
                    (nearest_pose_frame(pose_frames, float(frame["time_sec"])) or {}).get("left")
                    or {}
                ).get("box")
            ),
            "tip_approach_candidate_count": sum(len(tip_approach[side]) for side in SIDES),
            "blade_proximity_candidate_count": len(blade_proximity),
            "possible_parry_candidate_count": len(parries),
        },
        "limitations": [
            "A Pose extension candidate does not prove attack priority.",
            "Tip proximity to a 2D person box does not prove a valid hit or target contact.",
            "Blade proximity plus direction change is only a parry candidate; occlusion and perspective can imitate it.",
            "No contact/parry candidate is synthesized from temporal-held blade geometry.",
            "An empty candidate list means not observed in clear frames, not proof that the action did not occur.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-json", type=Path, required=True)
    parser.add_argument("--blade-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--pose-time-offset-sec",
        type=float,
        default=0.0,
        help="Source-video timestamp corresponding to Pose clip time 0.",
    )
    args = parser.parse_args()

    result = analyze(
        load_json(args.pose_json),
        load_json(args.blade_json),
        pose_time_offset_sec=args.pose_time_offset_sec,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    rows = flatten_rows(result)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = ["evidence_type", "side", "state", "start_sec", "end_sec", "detail"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
