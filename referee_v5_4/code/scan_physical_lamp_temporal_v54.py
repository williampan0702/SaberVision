from __future__ import annotations

import argparse
import bisect
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def load_track(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("x1")]
    track = [
        (float(row["time_sec"]), np.array([float(row[key]) for key in ("x1", "y1", "x2", "y2")]))
        for row in rows
    ]
    track.sort(key=lambda item: item[0])
    return track, [item[0] for item in track]


def interpolate_box(track, times, timestamp: float, max_gap: float = 1.2):
    if not track:
        return None
    index = bisect.bisect_left(times, timestamp)
    if index == 0:
        return track[0][1] if abs(times[0] - timestamp) <= max_gap else None
    if index == len(track):
        return track[-1][1] if abs(times[-1] - timestamp) <= max_gap else None
    before_time, before_box = track[index - 1]
    after_time, after_box = track[index]
    if timestamp - before_time > max_gap or after_time - timestamp > max_gap:
        return None
    weight = (timestamp - before_time) / max(1e-6, after_time - before_time)
    return before_box * (1 - weight) + after_box * weight


def expand(box, width: int, height: int, factor: float = 1.24):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = (x2 - x1) * factor / 2, (y2 - y1) * factor / 2
    return (
        max(0, int(round(cx - half_w))), max(0, int(round(cy - half_h))),
        min(width, int(round(cx + half_w))), min(height, int(round(cy + half_h))),
    )


def color_fractions(crop: np.ndarray):
    normalized = cv2.resize(crop, (256, 192), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)
    top = hsv[:128]
    left, right = top[:, :128], top[:, 128:]
    red = (((left[..., 0] <= 12) | (left[..., 0] >= 168)) & (left[..., 1] >= 100) & (left[..., 2] >= 145))
    green = ((right[..., 0] >= 35) & (right[..., 0] <= 100) & (right[..., 1] >= 75) & (right[..., 2] >= 120))
    return float(red.mean()), float(green.mean())


def spans(active: list[bool], times: list[float], minimum_frames: int, maximum_gap_frames: int):
    indices = [index for index, value in enumerate(active) if value]
    if not indices:
        return []
    groups = [[indices[0]]]
    for index in indices[1:]:
        if index - groups[-1][-1] <= maximum_gap_frames + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    return [(times[group[0]], times[group[-1]]) for group in groups if len(group) >= minimum_frames]


def fuse_color_events(events: list[dict], maximum_onset_gap: float = 0.25) -> list[dict]:
    """Fuse opposite colours only when their lamp onsets are near-simultaneous.

    Lamp states can remain visible for several seconds.  Simple interval overlap
    incorrectly turns a later colour reflection/scoreboard animation into a
    double light.  Priority depends on the touch onset, so late colours remain
    separate events and are resolved from the earliest onset downstream.
    """
    ordered = sorted(events, key=lambda item: item["start_sec"])
    used: set[int] = set()
    fused: list[dict] = []
    for index, event in enumerate(ordered):
        if index in used:
            continue
        match_index = next((
            other_index
            for other_index in range(index + 1, len(ordered))
            if other_index not in used
            and ordered[other_index]["state"] != event["state"]
            and abs(ordered[other_index]["start_sec"] - event["start_sec"]) <= maximum_onset_gap
            and ordered[other_index]["start_sec"] <= event["end_sec"]
            and event["start_sec"] <= ordered[other_index]["end_sec"]
        ), None)
        if match_index is None:
            fused.append(dict(event))
            continue
        other = ordered[match_index]
        used.add(match_index)
        fused.append({
            "state": "double",
            "start_sec": min(event["start_sec"], other["start_sec"]),
            "end_sec": max(event["end_sec"], other["end_sec"]),
            "peak_score": max(event["peak_score"], other["peak_score"]),
        })
    return sorted(fused, key=lambda item: item["start_sec"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal colour fallback inside a stable physical-device track.")
    parser.add_argument("input", type=Path)
    parser.add_argument("track", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--analysis-fps", type=float, default=15.0)
    parser.add_argument("--red-delta", type=float, default=0.006)
    parser.add_argument("--green-delta", type=float, default=0.004)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    track, track_times = load_track(args.track)
    cap = cv2.VideoCapture(str(args.input))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_times, red_values, green_values, valid = [], [], [], []
    timestamp = args.start
    step = 1.0 / args.analysis_fps
    while timestamp <= args.end + 1e-6:
        box = interpolate_box(track, track_times, timestamp)
        if box is None:
            sample_times.append(timestamp); red_values.append(0.0); green_values.append(0.0); valid.append(False)
            timestamp += step
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        x1, y1, x2, y2 = expand(box, width, height)
        red, green = color_fractions(frame[y1:y2, x1:x2])
        sample_times.append(timestamp); red_values.append(red); green_values.append(green); valid.append(True)
        timestamp += step
    cap.release()
    valid_red = [value for value, is_valid in zip(red_values, valid) if is_valid]
    valid_green = [value for value, is_valid in zip(green_values, valid) if is_valid]
    red_baseline = float(np.quantile(valid_red, 0.10)) if valid_red else 0.0
    green_baseline = float(np.quantile(valid_green, 0.10)) if valid_green else 0.0
    red_scores = [max(0.0, value - red_baseline) if is_valid else 0.0 for value, is_valid in zip(red_values, valid)]
    green_scores = [max(0.0, value - green_baseline) if is_valid else 0.0 for value, is_valid in zip(green_values, valid)]
    red_active = [score >= args.red_delta for score in red_scores]
    green_active = [score >= args.green_delta for score in green_scores]
    minimum_frames = max(3, int(round(args.analysis_fps * 0.25)))
    gap_frames = max(1, int(round(args.analysis_fps * 0.20)))
    events = []
    for state, active, scores in (("left", red_active, red_scores), ("right", green_active, green_scores)):
        for start, end in spans(active, sample_times, minimum_frames, gap_frames):
            peak = max(score for time, score in zip(sample_times, scores) if start <= time <= end)
            events.append({"state": state, "start_sec": start, "end_sec": end, "peak_score": peak})
    # Convert only near-simultaneous opposite-colour onsets into a double.
    fused = fuse_color_events(events)
    fields = ["state", "start_sec", "end_sec", "duration_sec", "peak_score"]
    with (args.output / "physical_temporal_lamp_events.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for event in fused:
            writer.writerow({**event, "duration_sec": event["end_sec"] - event["start_sec"]})
    with (args.output / "physical_temporal_lamp_frames.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_sec", "valid", "red_fraction", "green_fraction", "red_score", "green_score"])
        writer.writerows(zip(sample_times, valid, red_values, green_values, red_scores, green_scores))
    summary = {
        "analyzed_frames": len(sample_times), "valid_frames": sum(valid),
        "red_baseline": red_baseline, "green_baseline": green_baseline,
        "red_delta": args.red_delta, "green_delta": args.green_delta,
        "double_onset_max_gap_sec": 0.25, "events": fused,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
