#!/usr/bin/env python3
"""Build lamp-aligned visual packets for double-light action prelabelling."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


def read_specs(paths: list[Path]) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows[row["case_id"]] = row
    return rows


def tile(frame: np.ndarray, text: str, width: int = 480, height: int = 270) -> np.ndarray:
    image = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(image, (0, 0), (width, 34), (0, 0, 0), -1)
    cv2.putText(image, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def sequence_sheet(video: Path, case: dict[str, str], lamp_local: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(video)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps
    times = np.linspace(max(0.0, lamp_local - 1.2), min(duration - 1 / fps, lamp_local + 0.2), 9)
    tiles: list[np.ndarray] = []
    for time_sec in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(time_sec) * 1000)
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((270, 480, 3), dtype=np.uint8)
        relative = float(time_sec) - lamp_local
        marker = "LAMP" if abs(relative) < 0.11 else ""
        tiles.append(tile(frame, f"{relative:+.2f}s {marker}"))
    cap.release()
    grid = np.vstack([np.hstack(tiles[index:index + 3]) for index in range(0, 9, 3)])
    header = np.zeros((70, grid.shape[1], 3), dtype=np.uint8)
    title = f"{case['case_id']} | result={case['official_outcome']} | venue={case['venue']} | AI process label: pending"
    cv2.putText(header, title, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(header, "Frames run left-to-right, top-to-bottom; time is relative to first lamp.", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 220, 255), 2, cv2.LINE_AA)
    return np.vstack([header, grid])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, nargs="+", required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    specs = read_specs(args.spec)
    manifest = json.loads((args.audit_root / "blade_audit_manifest.json").read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    sheets: list[Path] = []
    rows: list[dict[str, object]] = []
    for record in manifest["results"]:
        case_id = record["case_id"]
        case = specs[case_id]
        video = args.audit_root / "clips" / f"{case_id}_lamp_window.mp4"
        lamp_local = float(record.get("lamp_time_in_window", float(record["lamp_time"]) - float(record["window_start"])))
        image = sequence_sheet(video, case, lamp_local)
        path = args.output_root / f"{case_id}_event_sequence.jpg"
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 94])
        sheets.append(path)
        rows.append({
            "case_id": case_id,
            "venue": case["venue"],
            "official_outcome": case["official_outcome"],
            "result_confidence": case.get("confidence", ""),
            "lamp_time": record["lamp_time"],
            "sequence_sheet": str(path.resolve()),
            "process_label": "pending_ai_review",
            "process_confidence": "",
            "attack_initiator": "",
            "parry_riposte_side": "",
            "simultaneous_candidate": "",
            "notes": "",
        })
    with (args.output_root / "event_label_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    overview_dir = args.output_root / "overview"
    overview_dir.mkdir(exist_ok=True)
    for page, start in enumerate(range(0, len(sheets), 4), 1):
        images = [cv2.imread(str(path)) for path in sheets[start:start + 4]]
        images = [cv2.resize(image, (720, 440), interpolation=cv2.INTER_AREA) for image in images]
        while len(images) < 4:
            images.append(np.zeros_like(images[0]))
        canvas = np.vstack([np.hstack(images[:2]), np.hstack(images[2:])])
        cv2.imwrite(str(overview_dir / f"double_event_overview_{page:02d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(json.dumps({"cases": len(rows), "overview_pages": math.ceil(len(rows) / 4)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
