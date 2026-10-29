from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from diagnose_device_locator_v46 import CASES, read_frame


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "v46_device_locator_comparison.json"
OUTPUT = ROOT / "v46_device_candidate_features.csv"


def features(crop: np.ndarray) -> dict[str, float]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    saturated = (saturation >= 90) & (value >= 90)
    lamp_hue = ((hue <= 30) | ((hue >= 35) & (hue <= 100)) | (hue >= 165))
    signal = saturated & lamp_hue
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    return {
        "saturated_fraction": float(np.mean(saturated)),
        "signal_fraction": float(np.mean(signal)),
        "bright_fraction": float(np.mean(value >= 175)),
        "edge_fraction": float(np.mean(edges > 0)),
        "gray_std": float(np.std(gray)),
    }


def main() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    by_case = {row["case"]: row for row in payload["cases"]}
    rows = []
    for case_name, video, timestamp in CASES:
        frame = read_frame(video, timestamp)
        height, width = frame.shape[:2]
        for model_name, detections in by_case[case_name]["models"].items():
            for detection_index, detection in enumerate(detections):
                x1, y1, x2, y2 = detection["box_norm"]
                ix1, iy1 = max(0, round(x1 * width)), max(0, round(y1 * height))
                ix2, iy2 = min(width, round(x2 * width)), min(height, round(y2 * height))
                crop = frame[iy1:iy2, ix1:ix2]
                if crop.size == 0:
                    continue
                rows.append(
                    {
                        "case": case_name,
                        "model": model_name,
                        "detection": detection_index,
                        "confidence": detection["confidence"],
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "width_norm": x2 - x1,
                        "height_norm": y2 - y1,
                        **features(crop),
                    }
                )
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row["model"] == "v46_corrected" or row["model"] == "legacy":
            print(json.dumps(row, ensure_ascii=False))
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
