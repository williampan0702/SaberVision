#!/usr/bin/env python3
"""Stabilize the physical weapon-wrist trajectory in existing Pose JSON files."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path


WRISTS = ("left_wrist", "right_wrist")
ELBOWS = {"left_wrist": "left_elbow", "right_wrist": "right_elbow"}
SHOULDERS = {"left_wrist": "left_shoulder", "right_wrist": "right_shoulder"}
ANGLE_KEYS = {"left_wrist": "left_elbow_angle", "right_wrist": "right_elbow_angle"}


def point(athlete: dict, name: str) -> tuple[float, float, float, bool] | None:
    item = (athlete.get("keypoints") or {}).get(name)
    if not isinstance(item, dict) or item.get("x") is None or item.get("y") is None:
        return None
    return float(item["x"]), float(item["y"]), float(item.get("confidence", 0.0)), bool(item.get("held"))


def distance(first: tuple[float, float, float, bool], second: tuple[float, float, float, bool]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def emission(athlete: dict, wrist_name: str, side: str) -> float:
    wrist = point(athlete, wrist_name)
    elbow = point(athlete, ELBOWS[wrist_name])
    shoulder = point(athlete, SHOULDERS[wrist_name])
    box = athlete.get("box") or []
    if wrist is None or elbow is None or shoulder is None or len(box) != 4:
        return -100.0
    width = max(1.0, float(box[2]) - float(box[0]))
    height = max(1.0, float(box[3]) - float(box[1]))
    center_x = (float(box[0]) + float(box[2])) / 2.0
    direction = 1.0 if side == "left" else -1.0
    inward = direction * (wrist[0] - center_x) / width
    reach = distance(wrist, shoulder) / height
    forearm = distance(wrist, elbow) / height
    features = athlete.get("features") or {}
    elbow_angle = features.get(ANGLE_KEYS[wrist_name])
    try:
        straightness = float(elbow_angle) / 180.0
    except (TypeError, ValueError):
        straightness = 0.5
    held_penalty = 0.25 if wrist[3] else 0.0
    return (
        1.10 * wrist[2]
        + 0.40 * elbow[2]
        + 0.75 * inward
        + 0.55 * reach
        + 0.20 * forearm
        + 0.25 * straightness
        - held_penalty
    )


def stabilize_side(
    frames: list[dict], side: str, jump_penalty: float, index_switch_penalty: float
) -> tuple[list[str], dict[str, int | float]]:
    scores: list[dict[str, float]] = []
    back: list[dict[str, str | None]] = []
    previous_points: dict[str, tuple[float, float, float, bool] | None] = {name: None for name in WRISTS}
    for frame_index, frame in enumerate(frames):
        athlete = frame.get(side) or {}
        current_scores: dict[str, float] = {}
        current_back: dict[str, str | None] = {}
        box = athlete.get("box") or []
        height = max(1.0, float(box[3]) - float(box[1])) if len(box) == 4 else 1.0
        for name in WRISTS:
            base = emission(athlete, name, side)
            if frame_index == 0:
                current_scores[name] = base
                current_back[name] = None
                continue
            candidate_point = point(athlete, name)
            options: list[tuple[float, str]] = []
            for previous_name in WRISTS:
                transition = 0.0
                previous_point = previous_points.get(previous_name)
                if candidate_point is not None and previous_point is not None:
                    jump = distance(candidate_point, previous_point) / height
                    transition -= jump_penalty * min(jump, 1.25)
                else:
                    transition -= 0.35
                if previous_name != name:
                    # Tiny index-switch cost suppresses noise but still permits a
                    # COCO left/right relabel when the physical point is continuous.
                    transition -= index_switch_penalty
                options.append((scores[-1][previous_name] + transition, previous_name))
            best_score, best_previous = max(options)
            current_scores[name] = base + best_score
            current_back[name] = best_previous
        scores.append(current_scores)
        back.append(current_back)
        previous_points = {name: point(athlete, name) for name in WRISTS}

    selected = max(WRISTS, key=lambda name: scores[-1][name]) if scores else "left_wrist"
    path = [selected]
    for index in range(len(frames) - 1, 0, -1):
        selected = back[index][selected] or selected
        path.append(selected)
    path.reverse()
    old = [(frame.get(side) or {}).get("weapon_wrist") for frame in frames]
    changed = sum(a != b for a, b in zip(old, path))
    switches = sum(path[index] != path[index - 1] for index in range(1, len(path)))
    return path, {"frames": len(path), "changed_frames": changed, "viterbi_switches": switches}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jump-penalty", type=float, default=4.0)
    parser.add_argument("--index-switch-penalty", type=float, default=0.08)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob("*_pose.json"))
    summaries = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        frames = payload.get("frames", [])
        item = {"case_id": path.stem.removesuffix("_pose")}
        for side in ("left", "right"):
            wrist_path, stats = stabilize_side(
                frames, side, args.jump_penalty, args.index_switch_penalty
            )
            for frame, wrist_name in zip(frames, wrist_path):
                athlete = frame.get(side)
                if not isinstance(athlete, dict):
                    continue
                athlete["weapon_wrist"] = wrist_name
                features = athlete.get("features") or {}
                features["weapon_elbow_angle"] = features.get(ANGLE_KEYS[wrist_name])
                athlete["weapon_wrist_tracking"] = "viterbi_physical_trajectory_v1"
            item[side] = stats
        payload["weapon_wrist_tracking"] = {
            "method": "viterbi_physical_trajectory_v1",
            "basis": "spatial continuity, inward reach, visibility and arm extension",
            "jump_penalty": args.jump_penalty,
            "index_switch_penalty": args.index_switch_penalty,
            "original_pose_preserved": True,
        }
        output_path = args.output_dir / path.name
        output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        summaries.append(item)
    summary = {
        "files": len(files),
        "changed_frames": sum(
            int(item[side]["changed_frames"]) for item in summaries for side in ("left", "right")
        ),
        "viterbi_switches": sum(
            int(item[side]["viterbi_switches"]) for item in summaries for side in ("left", "right")
        ),
    }
    (args.output_dir / "stabilization_summary.json").write_text(
        json.dumps({"summary": summary, "cases": summaries}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
