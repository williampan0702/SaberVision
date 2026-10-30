#!/usr/bin/env python3
"""Create compact lamp-aligned review sheets for unresolved MVP events."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
import statistics
import subprocess
from pathlib import Path

import cv2
import numpy as np


def finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_write(path: Path, image: np.ndarray) -> None:
    encoded, buffer = cv2.imencode(path.suffix or ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not encoded:
        raise RuntimeError(f"Could not encode {path}")
    buffer.tofile(str(path))


def nearest_pose(frames: list[dict], time_sec: float) -> dict:
    candidates = [
        (frame_time, frame)
        for frame in frames
        if isinstance(frame, dict)
        and (frame_time := finite_float(frame.get("time_sec"))) is not None
    ]
    if not candidates:
        return {}
    return min(candidates, key=lambda item: abs(item[0] - time_sec))[1]


def speed(frame: dict, side: str) -> float | None:
    value = (((frame.get(side) or {}).get("features") or {}).get("inward_speed_body_heights_per_sec"))
    return finite_float(value)


def sustained_start(samples: list[tuple[float, float]]) -> float | None:
    for index in range(max(0, len(samples) - 2)):
        window = samples[index : index + 3]
        if len(window) == 3 and sum(value >= 0.20 for _, value in window) >= 2:
            return window[0][0]
    return None


def motion_summary(pose_frames: list[dict], lamp_time: float) -> dict[str, object]:
    lamp_time_value = finite_float(lamp_time)
    if lamp_time_value is None:
        return {
            "cue": "initiation unclear; final approach unavailable",
            "left_onset_lead_sec": None,
            "right_onset_lead_sec": None,
            "final_speed_left_minus_right": None,
            "capability": "motion_review_cue_not_fencing_ruling",
        }
    lamp_time = lamp_time_value
    window: list[tuple[float, dict]] = []
    for frame in pose_frames:
        if not isinstance(frame, dict):
            continue
        time_sec = finite_float(frame.get("time_sec"))
        if time_sec is not None and lamp_time - 1.20 <= time_sec <= lamp_time:
            window.append((time_sec, frame))
    samples: dict[str, list[tuple[float, float]]] = {"left": [], "right": []}
    for time_sec, frame in window:
        for side in ("left", "right"):
            value = speed(frame, side)
            if value is not None:
                samples[side].append((time_sec, value))
    starts = {side: sustained_start(values) for side, values in samples.items()}
    onset_leads = {
        side: round(lamp_time - value, 3) if value is not None else None
        for side, value in starts.items()
    }
    if all(starts[side] is not None for side in ("left", "right")):
        assert starts["left"] is not None and starts["right"] is not None
        delta = starts["right"] - starts["left"]
        if abs(delta) <= 0.12:
            start_cue = "both initiate together"
        elif delta > 0:
            start_cue = f"left initiates {abs(delta):.2f}s earlier"
        else:
            start_cue = f"right initiates {abs(delta):.2f}s earlier"
    elif starts["left"] is not None:
        start_cue = "only left has sustained initiation"
    elif starts["right"] is not None:
        start_cue = "only right has sustained initiation"
    else:
        start_cue = "initiation unclear"

    final_means = {}
    for side in ("left", "right"):
        final = [value for time_sec, value in samples[side] if time_sec >= lamp_time - 0.35]
        final_means[side] = statistics.fmean(final) if final else None
    if all(final_means[side] is not None for side in ("left", "right")):
        final_delta = float(final_means["left"]) - float(final_means["right"])
        if final_delta >= 0.80:
            pressure_cue = "left has stronger final approach"
        elif final_delta <= -0.80:
            pressure_cue = "right has stronger final approach"
        else:
            pressure_cue = "final approach is balanced"
    else:
        final_delta = None
        pressure_cue = "final approach unavailable"
    return {
        "cue": f"{start_cue}; {pressure_cue}",
        "left_onset_lead_sec": onset_leads["left"],
        "right_onset_lead_sec": onset_leads["right"],
        "final_speed_left_minus_right": round(final_delta, 4) if final_delta is not None else None,
        "capability": "motion_review_cue_not_fencing_ruling",
    }


def device_summary(event: dict) -> dict[str, str]:
    state = str(event.get("touch_state") or "").strip()
    status = str(event.get("touch_status") or "").strip()
    evidence = event.get("device_evidence") or {}
    results = evidence.get("automatic_source_results") or {}
    resolution_rule = str(evidence.get("resolution_rule") or "")
    if state:
        display = state.upper()
    elif status == "automatic_source_conflict":
        display = "SOURCE CONFLICT"
    else:
        display = "NO RESOLVED SOURCE"
    if results:
        aliases = {
            "physical_temporal_lamp_events": "physical",
            "device_gated_lamp_events": "device-gated",
        }
        detail = " | ".join(
            f"{aliases.get(str(source), str(source))}={str(value).upper()}"
            for source, value in results.items()
        )
    else:
        detail = "no automatic lamp-source result"
    resolution_labels = {
        "physical_near_simultaneous_double_over_single_subset": (
            "fused=DOUBLE (physical double covers a single-colour subset)"
        ),
        "explicit_double_with_cross_source_colour_support": (
            "fused=DOUBLE (explicit double supported by both physical colours)"
        ),
    }
    if resolution_rule in resolution_labels:
        detail += " | " + resolution_labels[resolution_rule]
    return {"display": display, "detail": detail}


def tile(frame: np.ndarray, relative: float, pose_frame: dict, lamp: bool) -> np.ndarray:
    image = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (320, 42), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.78, image, 0.22, 0)
    left_speed, right_speed = speed(pose_frame, "left"), speed(pose_frame, "right")
    timing = f"{relative:+.2f}s" + ("  LAMP" if lamp else "")
    values = f"L {left_speed:+.2f}" if left_speed is not None else "L n/a"
    values += f"   R {right_speed:+.2f}" if right_speed is not None else "   R n/a"
    cv2.putText(image, timing, (7, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255) if lamp else (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, values, (7, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (190, 230, 255), 1, cv2.LINE_AA)
    return image


def build_sheet(
    cap: cv2.VideoCapture,
    pose_frames: list[dict],
    event: dict,
    motion: dict[str, object],
    device: dict[str, str],
) -> np.ndarray:
    start = float(event["start_sec"])
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps
    times = np.linspace(max(0.0, start - 1.20), min(duration - 1 / fps, start + 0.20), 12)
    panels = []
    for time_sec in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(time_sec) * 1000)
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((180, 320, 3), dtype=np.uint8)
        relative = float(time_sec) - start
        panels.append(tile(frame, relative, nearest_pose(pose_frames, float(time_sec)), abs(relative) <= 0.08))
    grid = np.vstack([np.hstack(panels[index : index + 4]) for index in range(0, 12, 4)])
    header = np.full((134, grid.shape[1], 3), (13, 15, 18), dtype=np.uint8)
    event_id = event.get("event_id", "?")
    verdict = event.get("verdict") or {}
    title = f"REVIEW EVENT {event_id} | {device['display']} | lamp {start:.2f}s"
    reason = str(verdict.get("reason") or "Unresolved event")
    cv2.putText(header, title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 205, 255), 2, cv2.LINE_AA)
    cv2.putText(header, reason[:118], (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (225, 225, 225), 1, cv2.LINE_AA)
    cv2.putText(header, f"SOURCES: {device['detail']}"[:135], (14, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 200, 120), 1, cv2.LINE_AA)
    cv2.putText(header, f"MOTION CUE (not ruling): {motion['cue']}"[:135], (14, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (120, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(header, "Order: left-to-right, top-to-bottom | speed units: body-heights/s toward opponent", (14, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (165, 190, 210), 1, cv2.LINE_AA)
    return np.vstack([header, grid])


def build_review_clip(
    video: Path,
    output: Path,
    pose_frames: list[dict],
    event: dict,
    device: dict[str, str],
) -> str:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(video)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    scale = min(1.0, 1280.0 / max(1, width))
    target_size = (max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2))
    lamp_time = float(event["start_sec"])
    start_time = max(0.0, lamp_time - 1.20)
    end_time = lamp_time + 0.30
    raw = output.with_name(f"{output.stem}_raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, target_size)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create review clip {raw}")
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        time_sec = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        if time_sec > end_time:
            break
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (target_size[0], 68), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.80, frame, 0.20, 0)
        relative = time_sec - lamp_time
        pose_frame = nearest_pose(pose_frames, time_sec)
        left_speed, right_speed = speed(pose_frame, "left"), speed(pose_frame, "right")
        timing = f"{relative:+.2f}s" + ("  LAMP" if abs(relative) <= 0.06 else "")
        speeds = f"L {left_speed:+.2f}" if left_speed is not None else "L n/a"
        speeds += f"   R {right_speed:+.2f}" if right_speed is not None else "   R n/a"
        color = (0, 220, 255) if abs(relative) <= 0.06 else (255, 255, 255)
        cv2.putText(frame, timing, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{device['display']} | {device['detail']} | {speeds}"[:120], (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 225, 255), 1, cv2.LINE_AA)
        # Duplicate every source frame so playback is 0.5x without interpolating evidence.
        writer.write(frame)
        writer.write(frame)
        written += 2
    writer.release()
    cap.release()
    if written == 0:
        raw.unlink(missing_ok=True)
        raise RuntimeError(f"No frames written for review event {event.get('event_id')}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        result = subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error", "-i", str(raw), "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and output.exists():
            raw.unlink(missing_ok=True)
            return "h264"
    raw.replace(output)
    return "mp4v_fallback"


def render_review_index(rows: list[dict]) -> str:
    cards = []
    for row in rows:
        sheet_name = Path(str(row["review_sheet"])).name
        clip = row.get("review_clip")
        clip_html = ""
        if clip:
            clip_name = Path(str(clip)).name
            clip_html = (
                f"<video controls preload='metadata' src='{html.escape(clip_name)}'>"
                "Your browser does not support embedded video. Use the contact sheet below.</video>"
            )
        cards.append(
            "<article class='card'>"
            f"<h2>Event {html.escape(str(row['event_id']))} · "
            f"{html.escape(str(row.get('device_display') or row.get('touch_state') or 'unknown').upper())}</h2>"
            f"<p><b>Lamp onset:</b> {html.escape(str(row.get('start_sec')))} s &nbsp;"
            f"<b>Status:</b> {html.escape(str(row.get('verdict_status') or 'review_required'))}</p>"
            f"<p><b>Score-box evidence:</b> {html.escape(str(row.get('device_summary') or 'unavailable'))}</p>"
            f"<p class='cue'><b>Motion cue:</b> {html.escape(str(row['motion_cue']))}</p>"
            f"{clip_html}"
            f"<a href='{html.escape(sheet_name)}'><img src='{html.escape(sheet_name)}' "
            f"alt='12-frame review sheet for event {html.escape(str(row['event_id']))}'></a>"
            "</article>"
        )
    content = "\n".join(cards) if cards else (
        "<section class='empty'><h2>No events require manual review</h2>"
        "<p>Every event has a final result, or no valid lamp event was detected.</p></section>"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SaberVision Double-Light Review</title>
<style>
body {{ margin:0; background:#11151a; color:#eaf0f5; font-family:system-ui,sans-serif; }}
main {{ width:min(1320px,94vw); margin:30px auto 60px; }}
h1 {{ margin-bottom:8px; }} .notice {{ color:#ffd36a; margin:0 0 24px; }}
.card,.empty {{ background:#1b222a; border:1px solid #354351; border-radius:12px; padding:18px; margin:18px 0; }}
.card h2 {{ margin:0 0 8px; color:#5bd9ff; }} .cue {{ color:#ffd36a; }}
img,video {{ width:100%; height:auto; border-radius:6px; display:block; margin-top:14px; background:#090b0d; }}
a {{ color:inherit; }}
</style>
</head>
<body><main>
<h1>SaberVision Double-Light Events Requiring Review</h1>
<p class="notice">Motion cues help locate visual evidence; they are not official sabre priority decisions.</p>
{content}
</main></body></html>
"""


def write_review_index(path: Path, rows: list[dict]) -> None:
    path.write_text(render_review_index(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-video", type=Path, required=True)
    parser.add_argument("--pose-json", type=Path, required=True)
    parser.add_argument("--verdict-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # These images are generated artifacts owned by this stage.  Remove stale
    # packets when a rerun changes an event from review to final.
    for pattern in ("event_*_review.jpg", "event_*_review*.mp4"):
        for stale in args.output_dir.glob(pattern):
            stale.unlink()
    pose = read_json(args.pose_json)
    verdict = read_json(args.verdict_json)
    events = []
    for event in verdict.get("events", []):
        decision = event.get("verdict") or {}
        status = str(decision.get("status") or "")
        if status.startswith("review_required") or (
            not status and decision.get("point_awarded") is None
        ):
            events.append(event)
    cap = cv2.VideoCapture(str(args.pose_video))
    if not cap.isOpened():
        raise RuntimeError(args.pose_video)
    rows = []
    for index, event in enumerate(events, start=1):
        event_id = event.get("event_id", index)
        # File names use the local packet index, never untrusted event IDs.
        # This prevents duplicate IDs from overwriting each other and keeps an
        # arbitrary string ID from escaping the requested output directory.
        path = args.output_dir / f"event_{index}_review.jpg"
        clip_path = args.output_dir / f"event_{index}_review.mp4"
        motion = motion_summary(pose.get("frames", []), float(event["start_sec"]))
        device = device_summary(event)
        safe_write(path, build_sheet(cap, pose.get("frames", []), event, motion, device))
        clip_codec = build_review_clip(args.pose_video, clip_path, pose.get("frames", []), event, device)
        rows.append({
            "event_id": event_id,
            "start_sec": event.get("start_sec"),
            "touch_state": event.get("touch_state"),
            "verdict_status": (event.get("verdict") or {}).get("status"),
            "device_display": device["display"],
            "device_summary": device["detail"],
            "motion_cue": motion["cue"],
            "left_onset_lead_sec": motion["left_onset_lead_sec"],
            "right_onset_lead_sec": motion["right_onset_lead_sec"],
            "final_speed_left_minus_right": motion["final_speed_left_minus_right"],
            "cue_capability": motion["capability"],
            "review_clip": str(clip_path.resolve()),
            "review_clip_codec": clip_codec,
            "review_clip_playback_speed": 0.5,
            "review_sheet": str(path.resolve()),
        })
    cap.release()
    manifest = args.output_dir / "review_manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "event_id", "start_sec", "touch_state", "verdict_status", "device_display",
            "device_summary", "motion_cue",
            "left_onset_lead_sec", "right_onset_lead_sec", "final_speed_left_minus_right",
            "cue_capability", "review_clip", "review_clip_codec", "review_clip_playback_speed",
            "review_sheet",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    review_index = args.output_dir / "review_index.html"
    write_review_index(review_index, rows)
    payload = {
        "review_events": len(rows),
        "events": rows,
        "manifest": str(manifest.resolve()),
        "review_index": str(review_index.resolve()),
        "output_dir": str(args.output_dir.resolve()),
    }
    args.summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
