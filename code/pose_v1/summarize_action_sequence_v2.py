from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluate_action_rules import evaluate_clip, find_exchange_phase, load_json, side_features


BALANCED = {
    "dynamic_fallback": True,
    "dynamic_stance": 0.28,
    "dynamic_speed": 0.85,
    "dynamic_elbow": 130.0,
    "dynamic_bent_knee_max": 155.0,
    "dynamic_rear_knee_min": 140.0,
}


def majority_smooth(labels: list[str], radius: int = 1) -> list[str]:
    order = {"steady": 0, "advance": 1, "retreat": 2}
    result = []
    for index in range(len(labels)):
        window = labels[max(0, index - radius): index + radius + 1]
        counts = Counter(window)
        result.append(max(counts, key=lambda value: (counts[value], -order[value])))
    return result


def stabilize_short_runs(labels: list[str], minimum_frames: int = 3) -> list[str]:
    labels = labels.copy()
    for _ in range(8):
        runs = []
        start = 0
        for index in range(1, len(labels) + 1):
            if index < len(labels) and labels[index] == labels[start]:
                continue
            runs.append((start, index, labels[start]))
            start = index
        changed = False
        for run_index, (start, end, _) in enumerate(runs):
            if end - start >= minimum_frames or len(runs) == 1:
                continue
            if run_index == 0:
                replacement = runs[1][2]
            elif run_index == len(runs) - 1:
                replacement = runs[-2][2]
            else:
                previous = runs[run_index - 1]
                following = runs[run_index + 1]
                if previous[2] == following[2]:
                    replacement = previous[2]
                else:
                    previous_length = previous[1] - previous[0]
                    following_length = following[1] - following[0]
                    replacement = (
                        previous[2] if previous_length >= following_length
                        else following[2]
                    )
            labels[start:end] = [replacement] * (end - start)
            changed = True
            break
        if not changed:
            break
    return labels


def movement_timeline(data, side, start_frame, contact_frame):
    samples = []
    for frame in data.get("frames", []):
        frame_no = int(frame.get("frame", -1))
        if start_frame is not None and frame_no < start_frame:
            continue
        if contact_frame is not None and frame_no > contact_frame:
            continue
        features = side_features(frame, side)
        if not features:
            continue
        movement = features.get("movement")
        if movement not in {"advance", "retreat", "steady"}:
            continue
        samples.append({
            "frame": frame_no,
            "time_sec": float(frame.get("time_sec", 0.0)),
            "movement": movement,
            "inward_speed": float(
                features.get("inward_speed_body_heights_per_sec") or 0.0
            ),
        })
    if not samples:
        return [], []
    labels = stabilize_short_runs(
        majority_smooth([sample["movement"] for sample in samples], radius=2)
    )
    segments = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index < len(labels) and labels[index] == labels[start]:
            continue
        part = samples[start:index]
        segments.append({
            "movement": labels[start],
            "start_frame": part[0]["frame"],
            "end_frame": part[-1]["frame"],
            "start_time_sec": round(part[0]["time_sec"], 3),
            "end_time_sec": round(part[-1]["time_sec"], 3),
            "duration_sec": round(
                max(0.0, part[-1]["time_sec"] - part[0]["time_sec"]), 3
            ),
            "mean_inward_speed_body_heights_per_sec": round(
                sum(item["inward_speed"] for item in part) / len(part), 4
            ),
            "sample_frames": len(part),
        })
        start = index
    transitions = [
        {
            "from": segments[index - 1]["movement"],
            "to": segments[index]["movement"],
            "at_frame": segments[index]["start_frame"],
            "at_time_sec": segments[index]["start_time_sec"],
        }
        for index in range(1, len(segments))
        if segments[index - 1]["movement"] != segments[index]["movement"]
    ]
    return segments, transitions


def summarize_one(path: Path, lunge_mode: str):
    data = load_json(path)
    frames = data.get("frames", [])
    total = len(frames)
    stats = data.get("stats") or {}
    detected = int(stats.get("frames_with_two_fencer_detections", 0))
    coverage = detected / total if total else 0.0
    start_frame, contact_frame = find_exchange_phase(data)
    athletes = {}
    for side in ("left", "right"):
        dynamic = BALANCED if lunge_mode == "balanced" else {}
        result = evaluate_clip(
            data=data, side=side, movement_min_votes=3, lunge_min_run=1,
            method="phase_net", net_movement_threshold=0.10,
            lunge_source="leg_geometry", stance_threshold=0.36,
            speed_threshold=0.02, elbow_threshold=110.0,
            bent_knee_max=135.0, rear_knee_min=145.0,
            **dynamic,
        )
        segments, transitions = movement_timeline(
            data, side, start_frame, contact_frame
        )
        athletes[side] = {
            "dominant_pre_contact_movement": result["predicted_movement"],
            "inward_net_body_heights": (
                round(result["inward_net_body_heights"], 4)
                if result["inward_net_body_heights"] is not None else None
            ),
            "lunge_present": result["predicted_lunge"],
            "lunge_candidate_frames": result["lunge_frames"],
            "movement_segments": segments,
            "movement_transitions": transitions,
        }
    pose_counts = [
        int(stats.get("left_pose_frames", 0)),
        int(stats.get("right_pose_frames", 0)),
    ]
    if coverage >= 0.70 and min(pose_counts) >= 10:
        reliability = "high"
    elif coverage >= 0.40 and min(pose_counts) >= 6:
        reliability = "medium"
    else:
        reliability = "low"
    return {
        "source_pose_json": str(path),
        "total_frames": total,
        "two_fencer_detected_frames": detected,
        "two_fencer_detected_coverage": round(coverage, 4),
        "reliability": reliability,
        "phase_start_frame": start_frame,
        "contact_frame": contact_frame,
        "athletes": athletes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lunge-mode", choices=("conservative", "balanced"),
        default="conservative",
    )
    args = parser.parse_args()
    paths = []
    for pose_dir in args.pose_dir:
        paths.extend(sorted(pose_dir.glob("*.json")))
    sequences = {
        path.stem: summarize_one(path, args.lunge_mode) for path in paths
    }
    payload = {
        "schema_version": "2.0",
        "lunge_mode": args.lunge_mode,
        "movement_timeline": {
            "source": "smoothed instantaneous pelvis velocity",
            "advance_threshold_body_heights_per_sec": 0.18,
            "retreat_threshold_body_heights_per_sec": -0.18,
        },
        "sequences": sequences,
        "limitations": [
            "AI-derived timelines are experimental, not referee ground truth.",
            "Very short segments may reflect pose jitter or tracking hold.",
            "Reliability is coverage, not a calibrated probability."
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "sequences": len(sequences), "lunge_mode": args.lunge_mode,
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
