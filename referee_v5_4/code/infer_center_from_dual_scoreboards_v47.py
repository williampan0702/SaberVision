from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from detect_scoreboard_structure_v47 import color_components, pair_score


def structure_boxes(frame: np.ndarray) -> list[dict[str, float]]:
    height, width = frame.shape[:2]
    items = sorted(color_components(frame), key=lambda item: float(item["cx"]))
    proposals: list[dict[str, float]] = []
    for left_index, left in enumerate(items):
        for right in items[left_index + 1 :]:
            result = pair_score(left, right, items, width, height)
            if result is None:
                continue
            score, related = result
            if score < 3.0:
                continue
            x1 = min(int(item["x1"]) for item in related)
            x2 = max(int(item["x2"]) for item in related)
            content_width = max(55, x2 - x1)
            center_y = float(np.median([float(item["cy"]) for item in related]))
            padded_x1 = max(0, round(x1 - content_width * 0.16))
            padded_x2 = min(width, round(x2 + content_width * 0.16))
            padded_width = padded_x2 - padded_x1
            box = {
                "x1": float(padded_x1),
                "y1": float(max(0, round(center_y - padded_width * 0.31))),
                "x2": float(padded_x2),
                "y2": float(min(height, round(center_y + padded_width * 0.31))),
                "score": float(score),
            }
            box["w"] = box["x2"] - box["x1"]
            box["h"] = box["y2"] - box["y1"]
            box["cx"] = (box["x1"] + box["x2"]) * 0.5
            box["cy"] = (box["y1"] + box["y2"]) * 0.5
            if not width * 0.08 <= box["w"] <= width * 0.30:
                continue
            if not height * 0.07 <= box["h"] <= height * 0.30:
                continue
            if not height * 0.13 <= box["cy"] <= height * 0.62:
                continue
            proposals.append(box)

    # Many component pairs describe the same screen.  Keep one proposal per
    # horizontal screen region before looking for a left/right pair.
    proposals.sort(key=lambda box: box["score"], reverse=True)
    kept: list[dict[str, float]] = []
    for box in proposals:
        if any(
            abs(box["cx"] - other["cx"]) < width * 0.08
            and abs(box["cy"] - other["cy"]) < height * 0.08
            for other in kept
        ):
            continue
        kept.append(box)
    return kept


def choose_pair(boxes: list[dict[str, float]], width: int, height: int):
    best = None
    for left_index, first in enumerate(boxes):
        for second in boxes[left_index + 1 :]:
            left, right = (first, second) if first["cx"] < second["cx"] else (second, first)
            separation = right["cx"] - left["cx"]
            if not width * 0.28 <= separation <= width * 0.68:
                continue
            if abs(left["cy"] - right["cy"]) > height * 0.075:
                continue
            width_ratio = max(left["w"], right["w"]) / max(1.0, min(left["w"], right["w"]))
            height_ratio = max(left["h"], right["h"]) / max(1.0, min(left["h"], right["h"]))
            if width_ratio > 1.55 or height_ratio > 1.55:
                continue
            midpoint = (left["cx"] + right["cx"]) * 0.5
            if not width * 0.30 <= midpoint <= width * 0.70:
                continue
            symmetry = 1.0 - min(1.0, abs(midpoint / width - 0.5) / 0.20)
            similarity = 2.0 - min(1.0, width_ratio - 1.0) - min(1.0, height_ratio - 1.0)
            score = left["score"] + right["score"] + symmetry + similarity
            if best is None or score > best[0]:
                best = (score, left, right)
    return best


def infer_middle(pair, width: int, height: int) -> tuple[int, int, int, int]:
    _score, left, right = pair
    center_x = (left["cx"] + right["cx"]) * 0.5
    center_y = (left["cy"] + right["cy"]) * 0.5
    mean_width = (left["w"] + right["w"]) * 0.5
    mean_height = (left["h"] + right["h"]) * 0.5
    # Include the external red/green lamp blocks beside the timer.  The old
    # narrow box often contained the numeric display but clipped the actual
    # lamp at its left or right edge during a slow camera pan.
    target_width = min(width * 0.15, max(width * 0.060, mean_width * 0.62))
    target_height = min(height * 0.32, max(height * 0.11, mean_height * 1.38))
    return (
        max(0, round(center_x - target_width * 0.5)),
        max(0, round(center_y - target_height * 0.5)),
        min(width, round(center_x + target_width * 0.5)),
        min(height, round(center_y + target_height * 0.5)),
    )


def smooth_box(previous, current, alpha: float = 0.30):
    if previous is None:
        return current
    return tuple(round((1.0 - alpha) * old + alpha * new) for old, new in zip(previous, current))


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer a narrow scoring device between two symmetric scoreboard screens.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(args.output / "dual_scoreboard_preview.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    previous = None
    missing_streak = 0
    hold_frames = max(0, round(args.hold_seconds * args.fps))
    rows = []
    timestamp = args.start
    while timestamp <= args.end + 1e-6:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = cap.read()
        if not ok:
            timestamp += 1.0 / args.fps
            continue
        boxes = structure_boxes(frame)
        pair = choose_pair(boxes, width, height)
        source = "missing"
        confidence = 0.0
        if pair is not None:
            middle = smooth_box(previous, infer_middle(pair, width, height))
            previous = middle
            missing_streak = 0
            source = "dual_scoreboard_geometry"
            confidence = float(pair[0])
        elif previous is not None and missing_streak < hold_frames:
            missing_streak += 1
            middle = previous
            source = "short_hold"
        else:
            missing_streak += 1
            middle = None

        confirmed = pair is not None
        row = {
            "time_sec": round(timestamp, 3),
            # A held box is useful for a stable preview, but it must never open
            # the lamp-recognition safety gate during a pan or close-up.
            "valid": int(confirmed),
            "source": source,
            "confidence": round(confidence, 6) if confidence else "",
            "x1": "",
            "y1": "",
            "x2": "",
            "y2": "",
            "left_x1": "",
            "left_y1": "",
            "left_x2": "",
            "left_y2": "",
            "right_x1": "",
            "right_y1": "",
            "right_x2": "",
            "right_y2": "",
        }
        if pair is not None:
            left_panel, right_panel = pair[1], pair[2]
            row.update(
                {
                    "left_x1": round(left_panel["x1"]),
                    "left_y1": round(left_panel["y1"]),
                    "left_x2": round(left_panel["x2"]),
                    "left_y2": round(left_panel["y2"]),
                    "right_x1": round(right_panel["x1"]),
                    "right_y1": round(right_panel["y1"]),
                    "right_x2": round(right_panel["x2"]),
                    "right_y2": round(right_panel["y2"]),
                }
            )
            for panel in pair[1:]:
                cv2.rectangle(
                    frame,
                    (round(panel["x1"]), round(panel["y1"])),
                    (round(panel["x2"]), round(panel["y2"])),
                    (255, 0, 255),
                    2,
                )
        if middle is not None:
            row.update(dict(zip(("x1", "y1", "x2", "y2"), middle)))
            box_color = (0, 255, 255) if confirmed else (0, 165, 255)
            cv2.rectangle(frame, (middle[0], middle[1]), (middle[2], middle[3]), box_color, 2)
        rows.append(row)
        cv2.rectangle(frame, (0, 0), (930, 44), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"V47 DUAL-SCOREBOARD: {source} candidates={len(boxes)} t={timestamp:.1f}s",
            (8, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0) if middle is not None else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        timestamp += 1.0 / args.fps

    cap.release()
    writer.release()
    with (args.output / "dual_scoreboard_track.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    source_counts: dict[str, int] = {}
    for row in rows:
        source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1
    valid_frames = sum(int(row["valid"]) for row in rows)
    tracked_frames = sum(int(row["source"] != "missing") for row in rows)
    summary = {
        "sampled_frames": len(rows),
        "valid_frames": valid_frames,
        "valid_fraction": valid_frames / max(1, len(rows)),
        "tracked_frames": tracked_frames,
        "tracked_fraction": tracked_frames / max(1, len(rows)),
        "sources": source_counts,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
