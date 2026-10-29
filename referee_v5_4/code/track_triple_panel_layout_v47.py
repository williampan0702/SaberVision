from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2


def expand_box(box: tuple[int, int, int, int], width: int, height: int, scale: float = 1.25):
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


def candidate(row: dict[str, str]):
    x1, y1, x2, y2 = (float(row[key]) for key in ("x1", "y1", "x2", "y2"))
    return {
        "row": row,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "w": x2 - x1,
        "h": y2 - y1,
        "cx": (x1 + x2) * 0.5,
        "cy": (y1 + y2) * 0.5,
        "confidence": float(row["confidence"]),
    }


def choose_side_pair(rows: list[dict[str, str]], width: int, height: int):
    items = [candidate(row) for row in rows]
    items = [
        item
        for item in items
        if width * 0.09 <= item["w"] <= width * 0.30
        and height * 0.055 <= item["h"] <= height * 0.25
        and item["confidence"] >= 0.08
    ]
    best = None
    for left_index, left in enumerate(items):
        for right in items[left_index + 1 :]:
            if left["cx"] > right["cx"]:
                left, right = right, left
            separation = right["cx"] - left["cx"]
            if not width * 0.30 <= separation <= width * 0.72:
                continue
            if abs(left["cy"] - right["cy"]) > height * 0.08:
                continue
            width_ratio = max(left["w"], right["w"]) / max(1.0, min(left["w"], right["w"]))
            height_ratio = max(left["h"], right["h"]) / max(1.0, min(left["h"], right["h"]))
            if width_ratio > 1.45 or height_ratio > 1.45:
                continue
            symmetry_penalty = abs(separation / width - 0.47)
            score = left["confidence"] + right["confidence"] - symmetry_penalty
            if best is None or score > best[0]:
                best = (score, left, right)
    return best


def synthesize_middle(pair, width: int, height: int):
    score, left, right = pair
    center_x = (left["cx"] + right["cx"]) * 0.5
    center_y = (left["cy"] + right["cy"]) * 0.5
    mean_w = (left["w"] + right["w"]) * 0.5
    mean_h = (left["h"] + right["h"]) * 0.5
    # A clipped/partly occluded side screen biases the midpoint and used to
    # exclude one of the physical machine's tall lamp towers.  The wider box
    # deliberately keeps both towers inside the centre view.
    target_w = min(width * 0.20, max(width * 0.085, mean_w * 0.95))
    target_h = min(height * 0.28, max(height * 0.10, mean_h * 1.35))
    box = (
        max(0, round(center_x - target_w * 0.5)),
        max(0, round(center_y - target_h * 0.5)),
        min(width, round(center_x + target_w * 0.5)),
        min(height, round(center_y + target_h * 0.5)),
    )
    return box, score, left, right


def rounded_box(item) -> tuple[int, int, int, int]:
    return tuple(round(item[key]) for key in ("x1", "y1", "x2", "y2"))


def relative_box(
    outer: tuple[int, int, int, int],
    fractions: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = outer
    fx1, fy1, fx2, fy2 = fractions
    width = x2 - x1
    height = y2 - y1
    return (
        round(x1 + fx1 * width),
        round(y1 + fy1 * height),
        round(x1 + fx2 * width),
        round(y1 + fy2 * height),
    )


def region_geometry(
    left: dict[str, float],
    middle: tuple[int, int, int, int],
    right: dict[str, float],
) -> dict[str, dict[str, tuple[int, int, int, int]]]:
    """Keep all three synchronized scoring views instead of discarding the sides."""
    left_box = rounded_box(left)
    right_box = rounded_box(right)
    return {
        "left_panel": {
            "box": left_box,
            # The score cells illuminate red/green in the lower half.
            "left_light": relative_box(left_box, (0.00, 0.48, 0.31, 1.00)),
            "right_light": relative_box(left_box, (0.69, 0.48, 1.00, 1.00)),
        },
        "center_box": {
            "box": middle,
            # Side-panel clipping can bias the inferred centre horizontally.
            # Search the whole upper band for each colour; colour, rather than
            # a fragile left/right split, identifies the two lamp states.
            "left_light": relative_box(middle, (0.00, 0.00, 1.00, 0.48)),
            "right_light": relative_box(middle, (0.00, 0.00, 1.00, 0.48)),
        },
        "right_panel": {
            "box": right_box,
            "left_light": relative_box(right_box, (0.00, 0.48, 0.31, 1.00)),
            "right_light": relative_box(right_box, (0.69, 0.48, 1.00, 1.00)),
        },
    }


def add_region_columns(row: dict[str, object], regions) -> None:
    for name in ("left_panel", "center_box", "right_panel"):
        region = regions.get(name)
        row[f"{name}_valid"] = int(region is not None)
        for box_name in ("box", "left_light", "right_light"):
            box = region[box_name] if region is not None else ("", "", "", "")
            prefix = name if box_name == "box" else f"{name}_{box_name.removesuffix('_light')}"
            for coordinate, value in zip(("x1", "y1", "x2", "y2"), box):
                row[f"{prefix}_{coordinate}"] = value


def smooth_regions(previous, current, width: int, height: int, alpha: float = 0.24):
    if not previous:
        return current
    output = {}
    for name, region in current.items():
        old = previous.get(name)
        if old is None:
            output[name] = region
            continue
        old_box = old["box"]
        new_box = region["box"]
        old_center = ((old_box[0] + old_box[2]) * 0.5, (old_box[1] + old_box[3]) * 0.5)
        new_center = ((new_box[0] + new_box[2]) * 0.5, (new_box[1] + new_box[3]) * 0.5)
        if abs(old_center[0] - new_center[0]) > width * 0.07 or abs(old_center[1] - new_center[1]) > height * 0.06:
            output[name] = region
            continue
        output[name] = {}
        for box_name, box in region.items():
            output[name][box_name] = tuple(
                round((1.0 - alpha) * old[box_name][index] + alpha * box[index])
                for index in range(4)
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer the middle scoring unit from symmetric left/right score panels.")
    parser.add_argument("input", type=Path)
    parser.add_argument("anchors", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    grouped: dict[float, list[dict[str, str]]] = defaultdict(list)
    with args.anchors.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[round(float(row["time_sec"]), 3)].append(row)

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None if args.no_preview else cv2.VideoWriter(
        str(args.output / "triple_panel_preview.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    rows = []
    previous_regions = {}
    previous_valid_time = float("-inf")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)
    source_index = 0
    timestamp = args.start
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_time = args.start + source_index / source_fps
        source_index += 1
        if frame_time > args.end + 0.5 / source_fps:
            break
        if frame_time + 0.5 / source_fps < timestamp:
            continue
        if timestamp > args.end + 1e-6:
            break
        rounded_time = round(timestamp, 3)
        pair = choose_side_pair(grouped.get(rounded_time, []), width, height)
        row: dict[str, object] = {
            "time_sec": rounded_time,
            "valid": int(pair is not None),
            "source": "triple_panel_layout" if pair is not None else "missing",
            "confidence": "",
            "x1": "",
            "y1": "",
            "x2": "",
            "y2": "",
        }
        regions = {}
        if pair is not None:
            box, score, left, right = synthesize_middle(pair, width, height)
            regions = region_geometry(left, box, right)
            if timestamp - previous_valid_time <= 0.65:
                regions = smooth_regions(previous_regions, regions, width, height)
            previous_regions = regions
            previous_valid_time = timestamp
            box = regions["center_box"]["box"]
            search = expand_box(box, width, height)
            row.update({"confidence": round(score, 6), **dict(zip(("x1", "y1", "x2", "y2"), box))})
            left_box = regions["left_panel"]["box"]
            right_box = regions["right_panel"]["box"]
            cv2.rectangle(frame, left_box[:2], left_box[2:], (255, 0, 255), 1)
            cv2.rectangle(frame, right_box[:2], right_box[2:], (255, 0, 255), 1)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
            cv2.rectangle(frame, (search[0], search[1]), (search[2], search[3]), (0, 255, 255), 2)
            colors = {"left_panel": (255, 0, 255), "center_box": (255, 255, 0), "right_panel": (255, 0, 255)}
            for name, region in regions.items():
                for side in ("left_light", "right_light"):
                    sx1, sy1, sx2, sy2 = region[side]
                    cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), colors[name], 1)
                ox1, oy1, _, _ = region["box"]
                cv2.putText(frame, name, (ox1, max(18, oy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colors[name], 1, cv2.LINE_AA)
        add_region_columns(row, regions)
        rows.append(row)
        cv2.rectangle(frame, (0, 0), (920, 46), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"V47 TRIPLE-PANEL: {'middle' if pair is not None else 'missing'} t={timestamp:.1f}s",
            (8, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 255, 0) if pair is not None else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        if writer is not None:
            writer.write(frame)
        timestamp += 1.0 / args.fps
    cap.release()
    if writer is not None:
        writer.release()

    with (args.output / "triple_panel_track.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    summary = {
        "sampled_frames": len(rows),
        "valid_frames": sum(int(row["valid"]) for row in rows),
        "valid_fraction": sum(int(row["valid"]) for row in rows) / max(1, len(rows)),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
