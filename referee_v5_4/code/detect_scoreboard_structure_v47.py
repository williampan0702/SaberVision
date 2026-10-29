from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def components(mask: np.ndarray, name: str) -> list[dict[str, float | int | str]]:
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    output = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if not 3 <= area <= 1800 or width > 150 or height > 90:
            continue
        cx, cy = centroids[index]
        output.append(
            {
                "color": name,
                "x1": int(x),
                "y1": int(y),
                "x2": int(x + width),
                "y2": int(y + height),
                "cx": float(cx),
                "cy": float(cy),
                "area": int(area),
            }
        )
    return output


def color_components(frame: np.ndarray) -> list[dict[str, float | int | str]]:
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    valid = (saturation >= 105) & (value >= 95)
    valid[: round(height * 0.12)] = False
    valid[round(height * 0.68) :] = False
    masks = {
        "red": valid & ((hue <= 12) | (hue >= 168)),
        "yellow": valid & (hue >= 16) & (hue <= 36),
        "green": valid & (hue >= 38) & (hue <= 78),
        "cyan": valid & (hue >= 79) & (hue <= 108),
    }
    kernel = np.ones((2, 2), dtype=np.uint8)
    output = []
    for name, mask in masks.items():
        cleaned = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
        output.extend(components(cleaned, name))
    return output


def pair_score(left: dict, right: dict, items: list[dict], width: int, height: int):
    dx = float(right["cx"]) - float(left["cx"])
    dy = abs(float(right["cy"]) - float(left["cy"]))
    if not width * 0.035 <= dx <= width * 0.30 or dy > height * 0.10:
        return None
    colors = {str(left["color"]), str(right["color"])}
    valid_pairs = ({"red", "green"}, {"red", "cyan"}, {"cyan", "green"})
    if colors not in valid_pairs:
        return None
    mid_y = (float(left["cy"]) + float(right["cy"])) * 0.5
    between = [
        item
        for item in items
        if float(left["cx"]) - 10 <= float(item["cx"]) <= float(right["cx"]) + 10
        and abs(float(item["cy"]) - mid_y) <= height * 0.09
    ]
    found_colors = {str(item["color"]) for item in between}
    color_bonus = len(found_colors & {"red", "cyan", "green"}) * 1.2
    area_bonus = min(1.5, sum(float(item["area"]) for item in between) / 180.0)
    score = color_bonus + area_bonus - dy / max(1.0, height * 0.10)
    return score, between


def locate(frame: np.ndarray):
    height, width = frame.shape[:2]
    items = color_components(frame)
    best = None
    ordered = sorted(items, key=lambda item: float(item["cx"]))
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            result = pair_score(left, right, items, width, height)
            if result is None:
                continue
            score, related = result
            if best is None or score > best[0]:
                best = (score, related)
    if best is None or best[0] < 3.0:
        return None, best[0] if best else 0.0, items
    score, related = best
    x1 = min(int(item["x1"]) for item in related)
    x2 = max(int(item["x2"]) for item in related)
    content_w = max(55, x2 - x1)
    # Horizontal colour separation is stable even when a fencer crosses the
    # device.  Vertical component extent is not: a red glove or coloured strip
    # can make it several times too tall.  Centre the device on the robust
    # colour-row median and derive its height from its width.
    color_mid_y = float(np.median([float(item["cy"]) for item in related]))
    padded_x1 = max(0, round(x1 - content_w * 0.16))
    padded_x2 = min(width, round(x2 + content_w * 0.16))
    padded_w = padded_x2 - padded_x1
    box = (
        padded_x1,
        max(0, round(color_mid_y - padded_w * 0.31)),
        padded_x2,
        min(height, round(color_mid_y + padded_w * 0.31)),
    )
    # A real scoring device occupies a compact part of a broadcast-wide shot.
    # Large colour arrangements are normally replay screens, lower thirds, or
    # people/clothing in a close-up.  Reject them instead of turning a high raw
    # coverage number into an unsafe lamp-search region.
    box_w = box[2] - box[0]
    box_h = box[3] - box[1]
    box_area_ratio = (box_w * box_h) / float(width * height)
    if box_w > width * 0.32 or box_h > height * 0.40 or box_area_ratio > 0.12:
        return None, score, items
    return box, score, items


def main() -> None:
    parser = argparse.ArgumentParser(description="Find scoreboard color structure without a device detector.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--fps", type=float, default=5.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(args.output / "structure_preview.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    rows = []
    timestamp = args.start
    while timestamp <= args.end + 1e-6:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = cap.read()
        if not ok:
            timestamp += 1.0 / args.fps
            continue
        box, score, items = locate(frame)
        row = {"time_sec": round(timestamp, 3), "valid": int(box is not None), "score": round(score, 4)}
        if box is not None:
            row.update(dict(zip(("x1", "y1", "x2", "y2"), box)))
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), 2)
        else:
            row.update({"x1": "", "y1": "", "x2": "", "y2": ""})
        rows.append(row)
        cv2.rectangle(frame, (0, 0), (720, 42), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"STRUCTURE valid={int(box is not None)} score={score:.2f} components={len(items)} t={timestamp:.1f}s",
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 0, 255) if box is not None else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        timestamp += 1.0 / args.fps
    cap.release()
    writer.release()
    with (args.output / "structure_boxes.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    summary = {
        "sampled_frames": len(rows),
        "valid_frames": sum(int(row["valid"]) for row in rows),
        "raw_coverage": sum(int(row["valid"]) for row in rows) / max(1, len(rows)),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
