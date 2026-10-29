from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from scan_dual_scoreboard_lamps_v47 import load_track, nearest_track
from validate_dual_scoreboard_lamp_events_v47 import read_csv


def color_score(frame: np.ndarray, box: tuple[int, int, int, int], side: str):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0, 0.0, 0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    vivid = (saturation >= 105) & (value >= 125)
    if side == "left":
        mask = vivid & ((hue <= 12) | (hue >= 168))
    else:
        mask = vivid & (hue >= 36) & (hue <= 88)
    mask_u8 = mask.astype(np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_u8, 8)
    largest = max((int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)), default=0)
    fraction = float(mask.mean())
    largest_fraction = largest / max(1.0, mask.size)
    score = fraction + largest_fraction * 1.8
    return score, fraction, largest


def search_box(device_box: tuple[int, int, int, int], width: int, height: int):
    x1, y1, x2, y2 = device_box
    device_w = x2 - x1
    device_h = y2 - y1
    return (
        max(0, round(x1 - device_w * 0.75)),
        max(0, round(y1 - device_h * 0.55)),
        min(width, round(x2 + device_w * 0.75)),
        min(height, round(y1 + device_h * 0.82)),
    )


def tile_images(images: list[np.ndarray], columns: int, tile_width: int, tile_height: int):
    rows = (len(images) + columns - 1) // columns
    canvas = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        resized = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        canvas[row * tile_height : (row + 1) * tile_height, column * tile_width : (column + 1) * tile_width] = resized
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine expected-colour lamp peaks before persistent score updates.")
    parser.add_argument("input", type=Path)
    parser.add_argument("device_track", type=Path)
    parser.add_argument("score_changes", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--before", type=float, default=5.0)
    parser.add_argument("--after", type=float, default=0.6)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--track-max-gap", type=float, default=0.35)
    parser.add_argument(
        "--interval-mode",
        action="store_true",
        help="Search only after the preceding score update, preventing a delayed update from reusing the previous hit.",
    )
    parser.add_argument(
        "--device-only",
        action="store_true",
        help="Measure colour in the located scoring apparatus itself instead of a wide region that can include fencers.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    track, track_times = load_track(args.device_track)
    changes = read_csv(args.score_changes)
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    peak_rows = []
    annotated = []
    for change_index, change in enumerate(changes, start=1):
        side = change["side"]
        change_time = float(change["time_sec"])
        samples = []
        previous_change = float(changes[change_index - 2]["time_sec"]) if change_index > 1 else 0.0
        search_start = max(0.0, change_time - args.before)
        if args.interval_mode:
            search_start = max(0.0, previous_change + 0.25)
        timestamp = search_start
        while timestamp <= change_time + args.after + 1e-6:
            track_row = nearest_track(track, track_times, timestamp, args.track_max_gap)
            if track_row is not None:
                _track_time, device_box, source = track_row
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                ok, frame = cap.read()
                if ok:
                    region = device_box if args.device_only else search_box(device_box, width, height)
                    expected_score, expected_fraction, expected_largest = color_score(frame, region, side)
                    opposite = "right" if side == "left" else "left"
                    opposite_score, opposite_fraction, _opposite_largest = color_score(frame, region, opposite)
                    samples.append(
                        {
                            "time": timestamp,
                            "frame": frame,
                            "device_box": device_box,
                            "search_box": region,
                            "source": source,
                            "expected_score": expected_score,
                            "expected_fraction": expected_fraction,
                            "expected_largest": expected_largest,
                            "opposite_score": opposite_score,
                            "opposite_fraction": opposite_fraction,
                            "combined_score": max(expected_score, opposite_score),
                        }
                    )
            timestamp += 1.0 / args.fps
        if not samples:
            continue
        best = max(samples, key=lambda sample: sample["combined_score"])
        scores = np.array([sample["combined_score"] for sample in samples], dtype=np.float32)
        baseline = float(np.percentile(scores, 25))
        onset = best["combined_score"] - baseline
        peak_rows.append(
            {
                "index": change_index,
                "score_side": side,
                "score_change_sec": f"{change_time:.4f}",
                "search_start_sec": f"{search_start:.4f}",
                "peak_sec": f"{best['time']:.4f}",
                "lead_sec": f"{change_time - best['time']:.4f}",
                "locator_source": best["source"],
                "expected_score": f"{best['expected_score']:.6f}",
                "expected_fraction": f"{best['expected_fraction']:.6f}",
                "expected_largest_pixels": best["expected_largest"],
                "opposite_score": f"{best['opposite_score']:.6f}",
                "opposite_fraction": f"{best['opposite_fraction']:.6f}",
                "baseline_score": f"{baseline:.6f}",
                "onset": f"{onset:.6f}",
                "combined_score": f"{best['combined_score']:.6f}",
            }
        )
        frame = best["frame"].copy()
        device_box = best["device_box"]
        region = best["search_box"]
        cv2.rectangle(frame, (region[0], region[1]), (region[2], region[3]), (0, 255, 255), 2)
        cv2.rectangle(frame, (device_box[0], device_box[1]), (device_box[2], device_box[3]), (255, 255, 0), 2)
        label_color = (0, 0, 255) if side == "left" else (0, 255, 0)
        cv2.rectangle(frame, (0, 0), (920, 52), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"#{change_index:02d} SCORE->{side.upper()} change={change_time:.1f}s peak={best['time']:.1f}s "
            f"lead={change_time-best['time']:.1f}s score={best['expected_score']:.3f}",
            (8, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.63,
            label_color,
            2,
            cv2.LINE_AA,
        )
        annotated.append(frame)
    cap.release()

    fields = list(peak_rows[0]) if peak_rows else ["index", "score_side", "score_change_sec", "peak_sec"]
    with (args.output / "score_change_lamp_peaks.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(peak_rows)
    if annotated:
        contact = tile_images(annotated, columns=5, tile_width=426, tile_height=240)
        cv2.imwrite(str(args.output / "score_change_lamp_peaks_contact.jpg"), contact)
    summary = {
        "score_changes": len(changes),
        "peaks_mined": len(peak_rows),
        "left": sum(row["score_side"] == "left" for row in peak_rows),
        "right": sum(row["score_side"] == "right" for row in peak_rows),
        "settings": {
            "before": args.before,
            "after": args.after,
            "fps": args.fps,
            "interval_mode": args.interval_mode,
            "device_only": args.device_only,
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
