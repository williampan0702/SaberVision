from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(WORKSPACE / ".ultralytics"))

from ultralytics import YOLO

from analyze_device_candidate_features_v46 import features


DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "score_device_v46_best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan strict score-device anchors at low FPS.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--signal-fraction", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--progress-seconds", type=float, default=60.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / source_fps
    start = max(0.0, args.start)
    end = min(duration, args.end if args.end is not None else duration)
    rows: list[dict[str, object]] = []
    sampled_frames = 0
    batch: list[tuple[float, np.ndarray]] = []
    started = time.perf_counter()

    def consume_batch(samples: list[tuple[float, np.ndarray]]) -> None:
        nonlocal sampled_frames
        if not samples:
            return
        results = model.predict(
            [frame for _timestamp, frame in samples],
            imgsz=960,
            conf=args.conf,
            iou=0.55,
            max_det=30,
            device=0,
            batch=max(1, args.batch_size),
            verbose=False,
        )
        for (timestamp, frame), result in zip(samples, results):
            sampled_frames += 1
            height, width = frame.shape[:2]
            if result.boxes is None:
                continue
            for detection_index, (confidence, box_values) in enumerate(
                zip(result.boxes.conf.cpu().tolist(), result.boxes.xyxy.cpu().tolist())
            ):
                x1, y1, x2, y2 = [max(0, round(value)) for value in box_values]
                x2, y2 = min(width, x2), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                values = features(frame[y1:y2, x1:x2])
                accepted = y2 / height < 0.90 and values["signal_fraction"] >= args.signal_fraction
                if not accepted:
                    continue
                rows.append(
                    {
                        "time_sec": round(timestamp, 3),
                        "detection": detection_index,
                        "confidence": round(float(confidence), 6),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        **{key: round(value, 6) for key, value in values.items()},
                    }
                )
    # Sequential decoding is much faster and more reliable than seeking to
    # every H.264 sample timestamp (the old loop also emitted frequent MMCO
    # decoder errors on long YouTube files).
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    source_index = 0
    next_sample_time = start
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = start + source_index / source_fps
        source_index += 1
        if timestamp > end + 1e-6:
            break
        if timestamp + 0.5 / source_fps < next_sample_time:
            continue
        batch.append((timestamp, frame.copy()))
        if len(batch) >= max(1, args.batch_size):
            consume_batch(batch)
            batch.clear()
            elapsed = time.perf_counter() - started
            if elapsed >= args.progress_seconds and sampled_frames % (max(1, args.batch_size) * 8) == 0:
                processed_seconds = timestamp - start
                rate = processed_seconds / max(0.001, elapsed)
                remaining = max(0.0, end - timestamp) / max(0.001, rate)
                print(
                    f"progress={timestamp:.1f}/{end:.1f}s samples={sampled_frames} "
                    f"speed={rate:.2f}x eta={remaining / 60:.1f}min",
                    flush=True,
                )
        next_sample_time += 1 / args.fps
    consume_batch(batch)
    cap.release()

    csv_path = args.output / "device_anchors.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("time_sec\n", encoding="utf-8-sig")

    selected = sorted(rows, key=lambda row: float(row["confidence"]), reverse=True)[:48]
    tiles = []
    cap = cv2.VideoCapture(str(args.input))
    for row in selected:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(row["time_sec"]) * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        x1, y1, x2, y2 = (int(row[key]) for key in ("x1", "y1", "x2", "y2"))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(
            frame,
            f"t={row['time_sec']:.1f} c={row['confidence']:.3f} sig={row['signal_fraction']:.3f}",
            (12, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(cv2.resize(frame, (384, 216), interpolation=cv2.INTER_AREA))
    cap.release()
    if tiles:
        while len(tiles) % 6:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[index : index + 6]) for index in range(0, len(tiles), 6)])
        cv2.imwrite(str(args.output / "device_anchor_contact_sheet.jpg"), sheet)
    summary = {
        "input": str(args.input),
        "model": str(args.model),
        "start": start,
        "end": end,
        "sample_fps": args.fps,
        "accepted_anchors": len(rows),
        "acceptance": f"y2_norm < 0.90 and signal_fraction >= {args.signal_fraction}",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
