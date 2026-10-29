from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def longest_true_run(values):
    best = cur = 0
    for value in values:
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def side_features(frame, side):
    record = frame.get(side)
    if not isinstance(record, dict):
        return None
    return record.get("features") if isinstance(record.get("features"), dict) else record


def smooth_median(values, radius=1):
    smoothed = []
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        smoothed.append(median(values[lo:hi]))
    return smoothed


def find_exchange_phase(data):
    valid = []
    for frame in data.get("frames", []):
        distance = frame.get("opponent_distance_body_heights")
        if distance is None:
            continue
        if side_features(frame, "left") and side_features(frame, "right"):
            valid.append(frame)
    if not valid:
        return None, None
    distances = [float(f["opponent_distance_body_heights"]) for f in valid]
    smoothed = smooth_median(distances, radius=1)
    contact_pos = min(range(len(valid)), key=lambda i: smoothed[i])
    return valid[0]["frame"], valid[contact_pos]["frame"]


def robust_net_movement(frames, side, start_frame, contact_frame, threshold):
    samples = []
    for frame in frames:
        if start_frame is not None and frame.get("frame", -1) < start_frame:
            continue
        if contact_frame is not None and frame.get("frame", -1) > contact_frame:
            continue
        feat = side_features(frame, side)
        record = frame.get(side)
        if not feat or not isinstance(record, dict):
            continue
        pelvis = feat.get("pelvis")
        box = record.get("box")
        if not pelvis or not box or len(box) != 4:
            continue
        body_height = max(1.0, float(box[3]) - float(box[1]))
        samples.append((float(pelvis[0]), body_height))
    if len(samples) < 4:
        return "steady", 0.0, len(samples)
    edge = max(2, min(5, len(samples) // 3))
    start_x = median(x for x, _ in samples[:edge])
    end_x = median(x for x, _ in samples[-edge:])
    body_height = median(h for _, h in samples)
    screen_delta = (end_x - start_x) / body_height
    inward_delta = screen_delta if side == "left" else -screen_delta
    if inward_delta >= threshold:
        movement = "advance"
    elif inward_delta <= -threshold:
        movement = "retreat"
    else:
        movement = "steady"
    return movement, inward_delta, len(samples)


def evaluate_clip(
    data, side, movement_min_votes, lunge_min_run,
    method, net_movement_threshold, lunge_source,
    stance_threshold, speed_threshold, elbow_threshold,
    bent_knee_max, rear_knee_min,
    dynamic_fallback=False, dynamic_stance=0.28, dynamic_speed=0.55,
    dynamic_elbow=140.0, dynamic_bent_knee_max=160.0,
    dynamic_rear_knee_min=145.0,
):
    frames = data.get("frames", [])
    features = []
    start_frame, contact_frame = find_exchange_phase(data)
    for frame in frames:
        feat = side_features(frame, side)
        if feat:
            features.append(feat)

    movements = [
        f.get("movement") for f in features
        if f.get("movement") in {"advance", "retreat", "steady"}
    ]
    counts = Counter(movements)
    nonsteady = counts["advance"] + counts["retreat"]
    inward_net = None
    phase_pose_records = None
    if method == "phase_net":
        movement, inward_net, phase_pose_records = robust_net_movement(
            frames, side, start_frame, contact_frame, net_movement_threshold
        )
    else:
        if nonsteady < movement_min_votes:
            movement = "steady"
        elif counts["advance"] >= counts["retreat"]:
            movement = "advance"
        else:
            movement = "retreat"

    lunge_flags = []
    for frame in frames:
        frame_no = frame.get("frame", -1)
        if start_frame is not None and frame_no < start_frame:
            continue
        if contact_frame is not None and frame_no > contact_frame + 3:
            continue
        feat = side_features(frame, side)
        if feat:
            if lunge_source == "current":
                flag = bool(feat.get("lunge_candidate", False))
            else:
                knees = [
                    feat.get("left_knee_angle"),
                    feat.get("right_knee_angle"),
                ]
                valid_knees = [float(v) for v in knees if v is not None]
                geometry_flag = (
                    len(valid_knees) == 2
                    and float(feat.get("stance_ratio") or 0.0) >= stance_threshold
                    and float(
                        feat.get("inward_speed_body_heights_per_sec") or 0.0
                    ) >= speed_threshold
                    and float(feat.get("weapon_elbow_angle") or 0.0)
                    >= elbow_threshold
                    and min(valid_knees) <= bent_knee_max
                    and max(valid_knees) >= rear_knee_min
                )
                dynamic_flag = (
                    dynamic_fallback
                    and len(valid_knees) == 2
                    and float(feat.get("stance_ratio") or 0.0)
                    >= dynamic_stance
                    and float(
                        feat.get("inward_speed_body_heights_per_sec") or 0.0
                    ) >= dynamic_speed
                    and float(feat.get("weapon_elbow_angle") or 0.0)
                    >= dynamic_elbow
                    and min(valid_knees) <= dynamic_bent_knee_max
                    and max(valid_knees) >= dynamic_rear_knee_min
                )
                flag = geometry_flag or dynamic_flag
            lunge_flags.append(flag)
    lunge_run = longest_true_run(lunge_flags)
    return {
        "predicted_movement": movement,
        "movement_counts": dict(counts),
        "inward_net_body_heights": inward_net,
        "predicted_lunge": lunge_run >= lunge_min_run,
        "lunge_frames": sum(lunge_flags),
        "lunge_longest_run": lunge_run,
        "pose_records": len(features),
        "phase_pose_records": phase_pose_records,
        "phase_start_frame": start_frame,
        "contact_frame": contact_frame,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--pose-dir", type=Path, required=True, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--movement-min-votes", type=int, default=3)
    parser.add_argument("--lunge-min-run", type=int, default=2)
    parser.add_argument(
        "--lunge-source", choices=("current", "leg_geometry"), default="current"
    )
    parser.add_argument("--stance-threshold", type=float, default=0.42)
    parser.add_argument("--speed-threshold", type=float, default=0.12)
    parser.add_argument("--elbow-threshold", type=float, default=125.0)
    parser.add_argument("--bent-knee-max", type=float, default=150.0)
    parser.add_argument("--rear-knee-min", type=float, default=150.0)
    parser.add_argument("--dynamic-fallback", action="store_true")
    parser.add_argument("--dynamic-stance", type=float, default=0.28)
    parser.add_argument("--dynamic-speed", type=float, default=0.55)
    parser.add_argument("--dynamic-elbow", type=float, default=140.0)
    parser.add_argument("--dynamic-bent-knee-max", type=float, default=160.0)
    parser.add_argument("--dynamic-rear-knee-min", type=float, default=145.0)
    parser.add_argument(
        "--method", choices=("majority", "phase_net"), default="majority"
    )
    parser.add_argument("--net-movement-threshold", type=float, default=0.10)
    args = parser.parse_args()

    labels = load_json(args.labels)
    rows = []
    for clip_id, truth in labels["clips"].items():
        candidates = []
        for pose_dir in args.pose_dir:
            # Historical review sets use exchange_XX_*.json.  Cross-venue
            # clips keep descriptive ids and use <id>_pose.json.  Resolve only
            # exact, anchored forms so ids such as 01 cannot match 010.
            historical = sorted(pose_dir.glob(f"exchange_{clip_id}_*.json"))
            descriptive = pose_dir / f"{clip_id}_pose.json"
            if descriptive.exists():
                candidates.append(descriptive)
            candidates.extend(historical)
        candidates = sorted(candidates)
        if not candidates:
            raise FileNotFoundError(f"No Pose JSON found for exchange {clip_id}")
        pose_json = candidates[0]
        data = load_json(pose_json)
        for side in ("left", "right"):
            pred = evaluate_clip(
                data, side, args.movement_min_votes, args.lunge_min_run,
                args.method, args.net_movement_threshold, args.lunge_source,
                args.stance_threshold, args.speed_threshold,
                args.elbow_threshold, args.bent_knee_max, args.rear_knee_min
                , args.dynamic_fallback, args.dynamic_stance,
                args.dynamic_speed, args.dynamic_elbow,
                args.dynamic_bent_knee_max, args.dynamic_rear_knee_min
            )
            gt = truth[side]
            rows.append({
                "clip_id": clip_id,
                "side": side,
                "label_confidence": gt["confidence"],
                "gt_movement": gt["dominant_movement"],
                "pred_movement": pred["predicted_movement"],
                "movement_correct": int(
                    gt["dominant_movement"] == pred["predicted_movement"]
                ),
                "gt_lunge": int(gt["lunge_present"]),
                "pred_lunge": int(pred["predicted_lunge"]),
                "lunge_correct": int(
                    bool(gt["lunge_present"]) == pred["predicted_lunge"]
                ),
                "movement_counts": json.dumps(
                    pred["movement_counts"], ensure_ascii=False
                ),
                "inward_net_body_heights": pred["inward_net_body_heights"],
                "lunge_frames": pred["lunge_frames"],
                "lunge_longest_run": pred["lunge_longest_run"],
                "pose_records": pred["pose_records"],
                "phase_pose_records": pred["phase_pose_records"],
                "phase_start_frame": pred["phase_start_frame"],
                "contact_frame": pred["contact_frame"],
                "pose_json": str(pose_json),
            })

    movement_correct = sum(r["movement_correct"] for r in rows)
    tp = sum(r["gt_lunge"] and r["pred_lunge"] for r in rows)
    fp = sum((not r["gt_lunge"]) and r["pred_lunge"] for r in rows)
    fn = sum(r["gt_lunge"] and (not r["pred_lunge"]) for r in rows)
    tn = sum((not r["gt_lunge"]) and (not r["pred_lunge"]) for r in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary = {
        "review_status": labels["review_status"],
        "athlete_sequences": len(rows),
        "movement_accuracy": movement_correct / len(rows) if rows else 0.0,
        "movement_correct": movement_correct,
        "movement_total": len(rows),
        "lunge": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
        },
        "settings": {
            "method": args.method,
            "movement_min_votes": args.movement_min_votes,
            "net_movement_threshold_body_heights": args.net_movement_threshold,
            "lunge_min_consecutive_frames": args.lunge_min_run,
            "lunge_source": args.lunge_source,
            "stance_threshold": args.stance_threshold,
            "speed_threshold": args.speed_threshold,
            "elbow_threshold_deg": args.elbow_threshold,
            "bent_knee_max_deg": args.bent_knee_max,
            "rear_knee_min_deg": args.rear_knee_min,
            "dynamic_fallback": args.dynamic_fallback,
            "dynamic_stance": args.dynamic_stance,
            "dynamic_speed": args.dynamic_speed,
            "dynamic_elbow_deg": args.dynamic_elbow,
            "dynamic_bent_knee_max_deg": args.dynamic_bent_knee_max,
            "dynamic_rear_knee_min_deg": args.dynamic_rear_knee_min,
        },
        "warning": "Metrics compare against AI visual reference labels, not expert ground truth."
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "action_rule_evaluation.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "action_rule_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
