from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np


def center(box):
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def scene_hist(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def valid_shape(row: dict[str, str], width: int, height: int) -> bool:
    bw = (float(row["x2"]) - float(row["x1"])) / width
    bh = (float(row["y2"]) - float(row["y1"])) / height
    aspect = bw / max(bh, 1e-6)
    return 0.045 <= bw <= 0.38 and 0.035 <= bh <= 0.30 and 0.65 <= aspect <= 4.5


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stable sparse score-device tracks from v4.6 anchors.")
    parser.add_argument("input", type=Path)
    parser.add_argument("anchors", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--required-hits", type=int, default=3)
    parser.add_argument("--max-gap", type=float, default=2.1)
    parser.add_argument("--alpha", type=float, default=0.35)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with args.anchors.open("r", encoding="utf-8-sig", newline="") as handle:
        anchor_rows = list(csv.DictReader(handle))
    grouped: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in anchor_rows:
        grouped[float(row["time_sec"])].append(row)

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    diagonal = math.hypot(width, height)
    previous_hist = None
    smooth_box = None
    last_time = None
    hit_history: deque[bool] = deque(maxlen=max(5, args.required_hits + 2))
    outputs = []
    review_frames = []

    for timestamp in sorted(grouped):
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        current_hist = scene_hist(frame)
        cut = previous_hist is not None and cv2.compareHist(previous_hist, current_hist, cv2.HISTCMP_BHATTACHARYYA) >= 0.42
        previous_hist = current_hist
        if cut or (last_time is not None and timestamp - last_time > args.max_gap):
            smooth_box = None
            hit_history.clear()

        candidates = [row for row in grouped[timestamp] if valid_shape(row, width, height)]
        candidates.sort(key=lambda row: float(row["confidence"]), reverse=True)
        chosen = None
        if candidates and smooth_box is None:
            chosen = candidates[0]
        elif candidates and smooth_box is not None:
            sx, sy = center(smooth_box)
            scored = []
            for row in candidates:
                box = tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2"))
                cx, cy = center(box)
                distance = math.hypot(cx - sx, cy - sy) / diagonal
                if distance <= 0.12:
                    scored.append((distance - 0.04 * float(row["confidence"]), row))
            if scored:
                chosen = min(scored, key=lambda item: item[0])[1]

        hit_history.append(chosen is not None)
        if chosen is not None:
            raw_box = np.array([float(chosen[key]) for key in ("x1", "y1", "x2", "y2")], dtype=np.float32)
            if smooth_box is None:
                smooth_box = raw_box
            else:
                smooth_box = (1.0 - args.alpha) * np.asarray(smooth_box) + args.alpha * raw_box
            last_time = timestamp
        elif last_time is not None and timestamp - last_time > args.max_gap:
            smooth_box = None
            hit_history.clear()

        stable = smooth_box is not None and sum(hit_history) >= args.required_hits and hit_history[-1]
        if not stable:
            continue
        x1, y1, x2, y2 = [round(value) for value in smooth_box]
        confidence = float(chosen["confidence"]) if chosen is not None else 0.0
        row = {
            "time_sec": timestamp,
            "scene_cut": int(cut),
            "confidence": round(confidence, 6),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "history_hits": sum(hit_history),
        }
        outputs.append(row)
        review = frame.copy()
        cv2.rectangle(review, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(
            review,
            f"STABLE t={timestamp:.1f} c={confidence:.3f} hits={sum(hit_history)}",
            (12, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        review_frames.append((confidence, review))
    cap.release()

    csv_path = args.output / "stable_device_track.csv"
    if outputs:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(outputs[0]))
            writer.writeheader()
            writer.writerows(outputs)
    else:
        csv_path.write_text("time_sec\n", encoding="utf-8-sig")

    chosen_reviews = sorted(review_frames, key=lambda item: item[0], reverse=True)[:48]
    tiles = [cv2.resize(frame, (384, 216), interpolation=cv2.INTER_AREA) for _, frame in chosen_reviews]
    if tiles:
        while len(tiles) % 6:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[index : index + 6]) for index in range(0, len(tiles), 6)])
        cv2.imwrite(str(args.output / "stable_device_track_contact_sheet.jpg"), sheet)
    summary = {
        "input": str(args.input),
        "anchors": len(anchor_rows),
        "stable_rows": len(outputs),
        "required_hits": args.required_hits,
        "max_gap": args.max_gap,
        "smoothing_alpha": args.alpha,
    }
    if outputs:
        summary.update({
            "median_center_x_norm": float(np.median([
                (float(row["x1"]) + float(row["x2"])) / (2 * width) for row in outputs
            ])),
            "median_center_y_norm": float(np.median([
                (float(row["y1"]) + float(row["y2"])) / (2 * height) for row in outputs
            ])),
            "median_area_fraction": float(np.median([
                ((float(row["x2"]) - float(row["x1"])) * (float(row["y2"]) - float(row["y1"])))
                / (width * height) for row in outputs
            ])),
        })
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
