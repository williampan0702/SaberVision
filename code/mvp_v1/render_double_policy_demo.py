#!/usr/bin/env python3
"""Render a compact visual demo of the frozen conservative double-light rule."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CASES = [
    "seoulh2_double_20",  # automatic right
    "seoulh2_double_33",  # automatic left
    "seoulh2_double_22",  # no-point reference, rule abstains
    "seoulh2_double_31",  # no-point reference, balanced motion
]
AUTO_THRESHOLD = 1.70
OUTPUT_SIZE = (1280, 720)
OUTPUT_FPS = 30.0


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["case_id"]: row for row in csv.DictReader(handle)}


def fit_frame(frame: np.ndarray) -> np.ndarray:
    target_w, target_h = OUTPUT_SIZE
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    resized = cv2.resize(frame, (round(w * scale), round(h * scale)))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y = (target_h - resized.shape[0]) // 2
    x = (target_w - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def put_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.72,
    color: tuple[int, int, int] = (245, 245, 245),
    thickness: int = 2,
) -> None:
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def describe_review(delta_035: float, delta_075: float) -> str:
    short_pass = abs(delta_035) >= AUTO_THRESHOLD
    long_pass = abs(delta_075) >= AUTO_THRESHOLD
    same_direction = delta_035 * delta_075 > 0
    if short_pass and not long_pass:
        return "REVIEW: 0.75s evidence is below the strong threshold"
    if long_pass and not short_pass:
        return "REVIEW: 0.35s evidence is below the strong threshold"
    if short_pass and long_pass and not same_direction:
        return "REVIEW: the two time windows disagree"
    return "REVIEW: no stable strong speed advantage"


def overlay(
    frame: np.ndarray,
    case_id: str,
    evaluation: dict[str, str],
    current_sec: float,
    duration_sec: float,
    lamp_sec: float,
) -> np.ndarray:
    result = frame.copy()
    dark = result.copy()
    cv2.rectangle(dark, (0, 0), (OUTPUT_SIZE[0], 176), (0, 0, 0), -1)
    cv2.addWeighted(dark, 0.72, result, 0.28, 0, result)

    delta_035 = float(evaluation["speed_delta"])
    delta_075 = float(evaluation["speed_delta_075"])
    prediction = evaluation["prediction"]
    official = evaluation["official_outcome"]

    put_text(result, f"DOUBLE-LIGHT FROZEN POLICY | {case_id}", (28, 34), 0.74)
    put_text(
        result,
        f"speed delta  0.35s: {delta_035:+.3f}   0.75s: {delta_075:+.3f}   auto threshold: +/-{AUTO_THRESHOLD:.3f}",
        (28, 72),
        0.66,
        (210, 230, 255),
    )
    put_text(result, "positive = LEFT faster | negative = RIGHT faster", (28, 106), 0.57, (190, 190, 190), 1)

    if current_sec < lamp_sec:
        verdict = "WAITING FOR DOUBLE LIGHT"
        verdict_color = (210, 210, 210)
        explanation = "Decision is emitted only after the lamp event"
    elif prediction in {"left", "right"}:
        verdict = f"MVP: AUTO {prediction.upper()}"
        verdict_color = (80, 235, 110)
        explanation = "Both speed windows pass the threshold in the same direction"
    else:
        verdict = "MVP: REVIEW REQUIRED"
        verdict_color = (0, 185, 255)
        explanation = describe_review(delta_035, delta_075)
    put_text(result, verdict, (28, 143), 0.82, verdict_color, 2)
    put_text(result, explanation, (390, 143), 0.58, verdict_color, 1)

    if current_sec >= lamp_sec:
        reference = official.replace("_", " ").upper()
        label = f"VALIDATION REFERENCE: {reference}"
        if official == "no_point" and prediction == "review":
            label += "  (correctly abstained)"
        put_text(result, label, (28, 662), 0.64, (240, 240, 240), 2)

    x0, x1, y = 28, 1252, 695
    cv2.line(result, (x0, y), (x1, y), (110, 110, 110), 6, cv2.LINE_AA)
    progress = min(1.0, max(0.0, current_sec / max(duration_sec, 0.001)))
    lamp_progress = min(1.0, max(0.0, lamp_sec / max(duration_sec, 0.001)))
    cv2.line(result, (x0, y), (round(x0 + (x1 - x0) * progress), y), (238, 238, 238), 6, cv2.LINE_AA)
    lamp_x = round(x0 + (x1 - x0) * lamp_progress)
    cv2.line(result, (lamp_x, y - 13), (lamp_x, y + 13), (0, 210, 255), 3, cv2.LINE_AA)
    put_text(result, "LAMP", (max(x0, lamp_x - 28), y - 18), 0.42, (0, 210, 255), 1)
    return result


def make_title_card(case_id: str, evaluation: dict[str, str], ordinal: int, total: int) -> np.ndarray:
    card = np.full((OUTPUT_SIZE[1], OUTPUT_SIZE[0], 3), (16, 18, 22), dtype=np.uint8)
    prediction = evaluation["prediction"]
    official = evaluation["official_outcome"].replace("_", " ").upper()
    kind = f"AUTO {prediction.upper()}" if prediction in {"left", "right"} else "REVIEW REQUIRED"
    color = (80, 235, 110) if prediction in {"left", "right"} else (0, 185, 255)
    put_text(card, f"EXAMPLE {ordinal}/{total}", (75, 110), 0.8, (175, 175, 175), 2)
    put_text(card, case_id, (75, 208), 1.35, (245, 245, 245), 3)
    put_text(card, kind, (75, 320), 1.55, color, 4)
    put_text(card, f"validation reference: {official}", (75, 395), 0.85, (215, 215, 215), 2)
    put_text(
        card,
        f"Frozen policy: both 0.35s and 0.75s speed windows must agree and pass +/-{AUTO_THRESHOLD:.2f}",
        (75, 560),
        0.69,
        (190, 210, 230),
        2,
    )
    return card


def transcode(temp_path: Path, output_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        temp_path.replace(output_path)
        return
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(temp_path),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )
    temp_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--pose-video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    args = parser.parse_args()

    specs = read_rows(args.spec)
    evaluations = read_rows(args.cases)
    case_ids = args.case_ids or DEFAULT_CASES
    missing = [case_id for case_id in case_ids if case_id not in specs or case_id not in evaluations]
    if missing:
        raise SystemExit(f"Missing cases: {', '.join(missing)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = args.output.with_name(args.output.stem + "_intermediate.mp4")
    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        OUTPUT_FPS,
        OUTPUT_SIZE,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {temp_path}")

    preview_frame: np.ndarray | None = None
    title_frames = round(0.8 * OUTPUT_FPS)
    freeze_frames = round(1.25 * OUTPUT_FPS)
    for ordinal, case_id in enumerate(case_ids, start=1):
        evaluation = evaluations[case_id]
        spec = specs[case_id]
        title = make_title_card(case_id, evaluation, ordinal, len(case_ids))
        for _ in range(title_frames):
            writer.write(title)

        video_path = args.pose_video_dir / f"{case_id}_pose.mp4"
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        source_fps = capture.get(cv2.CAP_PROP_FPS) or OUTPUT_FPS
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / source_fps
        lamp_sec = float(spec["lamp_time_override"])
        last: np.ndarray | None = None
        source_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            current_sec = source_index / source_fps
            rendered = overlay(fit_frame(frame), case_id, evaluation, current_sec, duration, lamp_sec)
            writer.write(rendered)
            last = rendered
            if case_id == "seoulh2_double_22" and current_sec >= lamp_sec + 0.15 and preview_frame is None:
                preview_frame = rendered.copy()
            source_index += 1
        capture.release()
        if last is not None:
            for _ in range(freeze_frames):
                writer.write(last)
    writer.release()

    transcode(temp_path, args.output)
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        if preview_frame is None:
            preview_frame = make_title_card(case_ids[0], evaluations[case_ids[0]], 1, len(case_ids))
        suffix = args.preview.suffix or ".jpg"
        encoded, buffer = cv2.imencode(suffix, preview_frame)
        if not encoded:
            raise RuntimeError(f"Could not encode preview: {args.preview}")
        buffer.tofile(str(args.preview))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
