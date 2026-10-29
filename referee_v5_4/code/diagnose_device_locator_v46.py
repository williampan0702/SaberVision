from __future__ import annotations

import json
from pathlib import Path

import cv2
from ultralytics import YOLO


MODELS = {
    "confirm": Path(r"D:\Codex输出\SaberVision_referee_recognition_v4_5_experimental\models\score_device_hardneg_confirm.pt"),
    "recall": Path(r"D:\Codex输出\SaberVision_referee_recognition_v4_5_experimental\models\score_device_high_recall.pt"),
    "legacy": Path(r"D:\Codex输出\SaberVision_referee_recognition_v4_5_experimental\models\score_device_high_recall_legacy.pt"),
    "v46_corrected": Path(
        r"D:\Codex输出\SaberVision_referee_training_v1\runs"
        r"\score_device_v46_corrected_finetune\weights\best.pt"
    ),
}
OUTPUT = Path(__file__).resolve().parent / "v46_device_locator_comparison.json"
CAIRO = Path(r"D:\Codex输出\SaberVision_training_sources\blind_holdout_v4_4_final\sources\cairo_2026_blind.mp4")
TBILISI = Path(r"D:\Codex输出\SaberVision_training_sources\full_videos\test\tbilisi_georgia.mp4")
HONG_KONG = Path(r"D:\Codex输出\SaberVision_training_sources\blind_holdout_v4_4_final\sources\hong_kong_2026_blind.mp4")

CASES = [
    ("cairo_wide_510", CAIRO, 510.8),
    ("cairo_pan_599", CAIRO, 599.8),
    ("cairo_pan_edge_601", CAIRO, 601.4),
    ("tbilisi_wide_139", TBILISI, 139.6),
    ("tbilisi_wide_220", TBILISI, 220.6),
    ("tbilisi_wide_375", TBILISI, 374.7),
    ("tbilisi_closeup_388", TBILISI, 387.7),
    ("hongkong_no_device_171", HONG_KONG, 170.8),
]


def read_frame(video: Path, timestamp: float):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError((video, timestamp))
    return frame


def main() -> None:
    models = {name: YOLO(str(path)) for name, path in MODELS.items()}
    report = []
    for case_name, video, timestamp in CASES:
        frame = read_frame(video, timestamp)
        height, width = frame.shape[:2]
        row = {"case": case_name, "time_sec": timestamp, "models": {}}
        for model_name, model in models.items():
            result = model.predict(
                frame, imgsz=960, conf=0.01, iou=0.55,
                max_det=30, device=0, verbose=False,
            )[0]
            detections = []
            for box, confidence, class_id in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
            ):
                x1, y1, x2, y2 = map(float, box)
                detections.append({
                    "class": model.names[int(class_id)],
                    "confidence": round(float(confidence), 5),
                    "box_norm": [round(x1 / width, 4), round(y1 / height, 4),
                                 round(x2 / width, 4), round(y2 / height, 4)],
                })
            row["models"][model_name] = detections
        report.append(row)
    payload = {"model_names": {name: model.names for name, model in models.items()}, "cases": report}
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    print(f"output={OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
