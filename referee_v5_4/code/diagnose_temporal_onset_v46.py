from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


MODEL = Path(
    r"D:\Codex输出\SaberVision_lamp_state_training_v1\stage60_physical_lamp_yolo_v45b_checkpoint_selection\weights\last.pt"
)
CAIRO = Path(
    r"D:\Codex输出\SaberVision_training_sources\blind_holdout_v4_4_final\sources\cairo_2026_blind.mp4"
)
HONG_KONG = Path(
    r"D:\Codex输出\SaberVision_training_sources\blind_holdout_v4_4_final\sources\hong_kong_2026_blind.mp4"
)
PLOVDIV = Path(
    r"D:\Codex输出\SaberVision_training_sources\blind_holdout_v4_3_final\sources\plovdiv_2025_blind.mp4"
)
TBILISI = Path(
    r"D:\Codex输出\SaberVision_training_sources\full_videos\test\tbilisi_georgia.mp4"
)

CASES = [
    ("cairo_true_green_510", CAIRO, 510.8),
    ("cairo_true_double_564", CAIRO, 564.8),
    ("cairo_true_double_599", CAIRO, 599.8),
    ("cairo_true_double_601", CAIRO, 601.4),
    ("hongkong_false_graphic", HONG_KONG, 170.8),
    ("hongkong_false_spotlights", HONG_KONG, 290.8),
    ("plovdiv_false_scoreboard", PLOVDIV, 123.4),
    ("plovdiv_true_double", PLOVDIV, 242.5),
    ("tbilisi_false_digit_139", TBILISI, 139.6),
    ("tbilisi_false_screen_151", TBILISI, 151.0),
    ("tbilisi_false_digit_220", TBILISI, 220.6),
    ("tbilisi_false_digit_318", TBILISI, 318.5),
    ("tbilisi_false_digit_375", TBILISI, 374.7),
    ("tbilisi_false_screen_388", TBILISI, 387.7),
]

PAST_OFFSETS = [0.20, 0.40, 0.67, 1.00]


def read_frame(video: Path, timestamp: float):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read {video} at {timestamp}")
    return frame


def expanded_box(box, width: int, height: int, expansion: float = 1.15):
    x1, y1, x2, y2 = map(float, box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = max(3.0, (x2 - x1) * expansion / 2)
    half_h = max(3.0, (y2 - y1) * expansion / 2)
    return (
        max(0, int(cx - half_w)), max(0, int(cy - half_h)),
        min(width, int(cx + half_w)), min(height, int(cy + half_h)),
    )


def activation(frame, box, color: str) -> dict:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = expanded_box(box, width, height)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return {"score": 0.0, "colored_fraction": 0.0, "bright_fraction": 0.0, "v90": 0.0}
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    if color == "red_lamp":
        colored = (saturation >= 55) & (value >= 90) & ((hue <= 15) | (hue >= 165))
    else:
        colored = (saturation >= 45) & (value >= 90) & (hue >= 32) & (hue <= 100)
    bright = value >= 220
    colored_fraction = float(np.mean(colored))
    bright_fraction = float(np.mean(bright))
    colored_energy = float(np.mean((value.astype(np.float32) / 255.0) * colored))
    v90 = float(np.percentile(value, 90) / 255.0)
    score = 0.60 * colored_energy + 0.25 * bright_fraction + 0.15 * v90
    return {
        "score": score,
        "colored_fraction": colored_fraction,
        "bright_fraction": bright_fraction,
        "v90": v90,
    }


def main() -> None:
    model = YOLO(str(MODEL))
    report = []
    for name, video, timestamp in CASES:
        current = read_frame(video, timestamp)
        past_frames = {offset: read_frame(video, timestamp - offset) for offset in PAST_OFFSETS}
        height, width = current.shape[:2]
        result = model.predict(
            current, imgsz=960, conf=0.005, iou=0.55, max_det=30,
            device=0, verbose=False,
        )[0]
        detections = []
        for box, confidence, class_id in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
        ):
            color = model.names[int(class_id)]
            current_metrics = activation(current, box, color)
            past_metrics = [activation(past_frames[offset], box, color) for offset in PAST_OFFSETS]
            past_scores = [item["score"] for item in past_metrics]
            baseline = float(np.median(past_scores))
            onset = current_metrics["score"] - baseline
            ratio = current_metrics["score"] / max(0.01, baseline)
            x1, y1, x2, y2 = map(float, box)
            detections.append({
                "color": color,
                "locator_conf": round(float(confidence), 5),
                "center_norm": [round((x1 + x2) / 2 / width, 4), round((y1 + y2) / 2 / height, 4)],
                "current": {key: round(value, 5) for key, value in current_metrics.items()},
                "past_scores": [round(value, 5) for value in past_scores],
                "baseline": round(baseline, 5),
                "onset": round(onset, 5),
                "ratio": round(ratio, 3),
                "onset_product": round(onset * float(confidence), 6),
            })
        detections.sort(key=lambda item: (item["onset"], item["locator_conf"]), reverse=True)
        report.append({"case": name, "time_sec": timestamp, "detections": detections})
    output = Path("work/v46_temporal_diagnostic.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = [
        {
            "case": row["case"],
            "top": row["detections"][:8],
        }
        for row in report
    ]
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
