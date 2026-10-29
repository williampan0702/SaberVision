from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def scene_hist(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def expand_box(box: np.ndarray, width: int, height: int, scale: float = 1.25) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    half_w = (x2 - x1) * scale * 0.5
    half_h = (y2 - y1) * scale * 0.5
    return (
        max(0, round(cx - half_w)),
        max(0, round(cy - half_h)),
        min(width, round(cx + half_w)),
        min(height, round(cy + half_h)),
    )


def anchor_agrees(box: np.ndarray, candidates: list[dict[str, str]]) -> bool:
    sx = (box[0] + box[2]) * 0.5
    sy = (box[1] + box[3]) * 0.5
    structure_area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    for row in candidates:
        ax1, ay1, ax2, ay2 = (float(row[key]) for key in ("x1", "y1", "x2", "y2"))
        aw = ax2 - ax1
        ah = ay2 - ay1
        if ax1 - aw * 0.08 <= sx <= ax2 + aw * 0.08 and ay1 - ah * 0.12 <= sy <= ay2 + ah * 0.12:
            return True
        intersection_w = max(0.0, min(float(box[2]), ax2) - max(float(box[0]), ax1))
        intersection_h = max(0.0, min(float(box[3]), ay2) - max(float(box[1]), ay1))
        if intersection_w * intersection_h / structure_area >= 0.08:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Stabilize compact structure detections without tracking through cuts.")
    parser.add_argument("input", type=Path)
    parser.add_argument("structure_csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--hold", type=float, default=0.0)
    parser.add_argument("--anchors", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with args.structure_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        samples = list(csv.DictReader(handle))
    anchors_by_time: dict[float, list[dict[str, str]]] = defaultdict(list)
    if args.anchors is not None:
        with args.anchors.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                anchors_by_time[round(float(row["time_sec"]), 3)].append(row)
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    diagonal = math.hypot(width, height)
    writer = cv2.VideoWriter(
        str(args.output / "structure_fused_preview.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )

    previous_hist = None
    track = None
    last_hit_time = None
    previous_accepted_time = None
    outputs = []
    counts = {
        "anchor_structure": 0,
        "structure_continuation": 0,
        "rejected_structure": 0,
        "short_hold": 0,
        "missing": 0,
    }
    for sample in samples:
        timestamp = float(sample["time_sec"])
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        current_hist = scene_hist(frame)
        cut_distance = 0.0 if previous_hist is None else cv2.compareHist(
            previous_hist, current_hist, cv2.HISTCMP_BHATTACHARYYA
        )
        cut = previous_hist is not None and cut_distance >= 0.42
        previous_hist = current_hist
        if cut:
            track = None
            last_hit_time = None
            previous_accepted_time = None

        source = "missing"
        if int(sample["valid"]):
            raw = np.array([float(sample[key]) for key in ("x1", "y1", "x2", "y2")], dtype=np.float32)
            confirmed = args.anchors is None or anchor_agrees(raw, anchors_by_time.get(round(timestamp, 3), []))
            continuous = False
            if track is not None and previous_accepted_time is not None and timestamp - previous_accepted_time <= 1.25 / args.fps:
                old_center = ((track[0] + track[2]) * 0.5, (track[1] + track[3]) * 0.5)
                new_center = ((raw[0] + raw[2]) * 0.5, (raw[1] + raw[3]) * 0.5)
                center_jump = math.hypot(new_center[0] - old_center[0], new_center[1] - old_center[1]) / diagonal
                old_area = max(1.0, (track[2] - track[0]) * (track[3] - track[1]))
                new_area = max(1.0, (raw[2] - raw[0]) * (raw[3] - raw[1]))
                area_ratio = max(old_area, new_area) / min(old_area, new_area)
                # A crossing fencer can temporarily widen the colour-derived
                # box, while its centre still follows the same physical device.
                continuous = not cut and center_jump <= 0.10 and area_ratio <= 3.5
            if confirmed or continuous:
                if track is not None and continuous:
                    track = (1.0 - args.alpha) * track + args.alpha * raw
                else:
                    track = raw
                last_hit_time = timestamp
                previous_accepted_time = timestamp
                source = "anchor_structure" if confirmed else "structure_continuation"
            else:
                track = None
                last_hit_time = None
                previous_accepted_time = None
                source = "rejected_structure"
        elif track is not None and last_hit_time is not None and timestamp - last_hit_time <= args.hold and not cut:
            source = "short_hold"
        else:
            track = None
            last_hit_time = None
            previous_accepted_time = None

        counts[source] += 1
        row = {
            "time_sec": round(timestamp, 3),
            "valid": int(track is not None),
            "source": source,
            "scene_cut": int(cut),
            "cut_distance": round(cut_distance, 5),
            "x1": "",
            "y1": "",
            "x2": "",
            "y2": "",
        }
        if track is not None:
            base = tuple(round(float(value)) for value in track)
            search = expand_box(track, width, height)
            row.update(dict(zip(("x1", "y1", "x2", "y2"), base)))
            cv2.rectangle(frame, (base[0], base[1]), (base[2], base[3]), (255, 255, 0), 2)
            cv2.rectangle(frame, (search[0], search[1]), (search[2], search[3]), (0, 255, 255), 2)
        outputs.append(row)
        cv2.rectangle(frame, (0, 0), (830, 48), (0, 0, 0), -1)
        color = (0, 255, 0) if source in {"anchor_structure", "structure_continuation"} else ((0, 220, 255) if source == "short_hold" else (0, 0, 255))
        cv2.putText(
            frame,
            f"V47 STRUCTURE-FUSED: {source} t={timestamp:.1f}s cut={int(cut)}",
            (8, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            color,
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)

    cap.release()
    writer.release()
    with (args.output / "fused_device_track.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(outputs[0]))
        writer_csv.writeheader()
        writer_csv.writerows(outputs)
    summary = {
        "sampled_frames": len(outputs),
        "valid_frames": sum(int(row["valid"]) for row in outputs),
        "valid_fraction": sum(int(row["valid"]) for row in outputs) / max(1, len(outputs)),
        "sources": counts,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
