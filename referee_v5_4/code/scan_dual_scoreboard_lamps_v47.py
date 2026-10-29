from __future__ import annotations

import argparse
import bisect
import csv
import json
import time
from pathlib import Path

import cv2

from validate_dual_scoreboard_lamp_events_v47 import measure, read_csv


def load_track(path: Path):
    track = []
    for row in read_csv(path):
        if not row.get("x1"):
            continue
        track.append(
            (
                float(row["time_sec"]),
                tuple(round(float(row[key])) for key in ("x1", "y1", "x2", "y2")),
                row.get("source", ""),
            )
        )
    track.sort()
    return track, [row[0] for row in track]


def nearest_track(track, times, timestamp: float, max_gap: float):
    index = bisect.bisect_left(times, timestamp)
    candidates = []
    if index < len(track):
        candidates.append(track[index])
    if index > 0:
        candidates.append(track[index - 1])
    if not candidates:
        return None
    nearest = min(candidates, key=lambda item: abs(item[0] - timestamp))
    return nearest if abs(nearest[0] - timestamp) <= max_gap else None


def majority_filter(values: list[bool], radius: int = 1) -> list[bool]:
    output = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        window = values[start:end]
        output.append(sum(window) >= 2)
    return output


def events_from_rows(
    rows: list[dict],
    gap_seconds: float,
    minimum_seconds: float,
    refractory_seconds: float,
):
    events = []
    current = None
    for row in rows:
        state = row["state"]
        timestamp = float(row["time_sec"])
        if state == "none":
            if current and timestamp - current["last_on"] > gap_seconds:
                events.append(current)
                current = None
            continue
        score = max(float(row["red_fraction"]), float(row["green_fraction"]))
        if current is None or timestamp - current["last_on"] > gap_seconds:
            if current:
                events.append(current)
            current = {
                "state": state,
                "start_sec": timestamp,
                "last_on": timestamp,
                "peak_score": score,
                "states": {state},
            }
        else:
            current["last_on"] = timestamp
            current["peak_score"] = max(current["peak_score"], score)
            current["states"].add(state)
            if "double" in current["states"] or {"left", "right"} <= current["states"]:
                current["state"] = "double"
    if current:
        events.append(current)
    output = []
    for event in events:
        duration = event["last_on"] - event["start_sec"]
        if duration < minimum_seconds:
            continue
        output.append(
            {
                "state": event["state"],
                "start_sec": round(event["start_sec"], 4),
                "end_sec": round(event["last_on"], 4),
                "duration_sec": round(duration, 4),
                "peak_score": f"{event['peak_score']:.6f}",
            }
        )
    # A scoring machine cannot represent two independent fencing exchanges a
    # couple of seconds apart: the referee must halt and restart first.  Short
    # follow-up colour fragments are normally the same lamp being occluded or
    # the broadcast wipe crossing the box, so keep the first consolidated
    # event and suppress those fragments.
    deduplicated = []
    for event in output:
        if (
            deduplicated
            and event["start_sec"] - deduplicated[-1]["end_sec"] < refractory_seconds
        ):
            continue
        deduplicated.append(event)
    return deduplicated


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct lamp-colour scanner for the v4.7 dual-scoreboard route.")
    parser.add_argument("input", type=Path)
    parser.add_argument("device_track", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--analysis-fps", type=float, default=15.0)
    parser.add_argument("--track-max-gap", type=float, default=0.35)
    parser.add_argument("--red-threshold", type=float, default=0.018)
    parser.add_argument("--green-threshold", type=float, default=0.008)
    parser.add_argument("--dark-threshold", type=float, default=0.16)
    parser.add_argument("--required-lamp-components", type=int, default=3)
    parser.add_argument("--event-gap", type=float, default=0.28)
    parser.add_argument("--minimum-event", type=float, default=0.20)
    parser.add_argument("--refractory", type=float, default=3.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    track, times = load_track(args.device_track)
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, round(source_fps / min(source_fps, args.analysis_fps)))
    output_fps = source_fps / stride
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)
    writer = cv2.VideoWriter(
        str(args.output / "dual_scoreboard_lamp_preview.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )

    raw_frames = []
    read_index = 0
    started = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = args.start + read_index / source_fps
        read_index += 1
        if timestamp > args.end:
            break
        if (read_index - 1) % stride:
            continue
        track_row = nearest_track(track, times, timestamp, args.track_max_gap)
        red_fraction = green_fraction = broad_red_fraction = broad_green_fraction = 0.0
        lower_red_lamp_blob = lower_green_lamp_blob = dark_fraction = lamp_row = 0.0
        box = None
        source = "missing"
        visible = False
        if track_row is not None:
            _track_time, box, source = track_row
            metrics = measure(frame, box)
            broad_red_fraction = metrics.get("broad_red_fraction", 0.0)
            broad_green_fraction = metrics.get("broad_green_fraction", 0.0)
            lower_red_lamp_blob = metrics.get("lower_red_lamp_blob", 0.0)
            lower_green_lamp_blob = metrics.get("lower_green_lamp_blob", 0.0)
            # Newer scoring units use two tall lamp towers instead of the
            # older four-lamp horizontal row.  Preserve the selective row
            # score when present, but allow the broad left/right lamp bands
            # to recover those vertical towers.
            red_fraction = lower_red_lamp_blob
            green_fraction = max(
                lower_green_lamp_blob,
                metrics.get("lower_green_fraction", 0.0),
            )
            dark_fraction = metrics["dark_fraction"]
            lamp_row = metrics["lamp_row"]
            visible = dark_fraction >= args.dark_threshold and (
                lamp_row >= args.required_lamp_components
                or red_fraction >= args.red_threshold
                or green_fraction >= args.green_threshold
            )
        raw_row = {
                "frame": len(raw_frames),
                "time_sec": f"{timestamp:.4f}",
                "source": source,
                "device_visible": int(visible),
                "lamp_components": round(lamp_row),
                "dark_fraction": f"{dark_fraction:.6f}",
                "red_fraction": f"{red_fraction:.6f}",
                "green_fraction": f"{green_fraction:.6f}",
                "broad_red_fraction": f"{broad_red_fraction:.6f}",
                "broad_green_fraction": f"{broad_green_fraction:.6f}",
                "lower_red_lamp_blob": f"{lower_red_lamp_blob:.6f}",
                "lower_green_lamp_blob": f"{lower_green_lamp_blob:.6f}",
                "red_raw": int(visible and red_fraction >= args.red_threshold),
                "green_raw": int(visible and green_fraction >= args.green_threshold),
            }
        raw_state = "double" if raw_row["red_raw"] and raw_row["green_raw"] else (
            "left" if raw_row["red_raw"] else "right" if raw_row["green_raw"] else "none"
        )
        if box is not None:
            color = (0, 255, 255) if visible else (0, 165, 255)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        state_color = (255, 255, 255)
        if raw_state == "left":
            state_color = (0, 0, 255)
        elif raw_state == "right":
            state_color = (0, 255, 0)
        elif raw_state == "double":
            state_color = (0, 255, 255)
        cv2.rectangle(frame, (0, 0), (930, 48), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"V47 DUAL LAMPS: {raw_state.upper()} device={int(visible)} "
            f"R={red_fraction:.3f} G={green_fraction:.3f} t={timestamp:.2f}",
            (8, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            state_color,
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        raw_frames.append(raw_row)

    red_filtered = majority_filter([bool(int(row["red_raw"])) for row in raw_frames])
    green_filtered = majority_filter([bool(int(row["green_raw"])) for row in raw_frames])
    for index, row in enumerate(raw_frames):
        red_on = red_filtered[index]
        green_on = green_filtered[index]
        row["red_on"] = int(red_on)
        row["green_on"] = int(green_on)
        row["state"] = "double" if red_on and green_on else "left" if red_on else "right" if green_on else "none"
    cap.release()
    writer.release()

    events = events_from_rows(
        raw_frames,
        args.event_gap,
        args.minimum_event,
        args.refractory,
    )
    with (args.output / "dual_scoreboard_lamp_frames.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(raw_frames[0]))
        csv_writer.writeheader()
        csv_writer.writerows(raw_frames)
    event_fields = ["state", "start_sec", "end_sec", "duration_sec", "peak_score"]
    with (args.output / "dual_scoreboard_lamp_events.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=event_fields)
        csv_writer.writeheader()
        csv_writer.writerows(events)
    summary = {
        "input": str(args.input),
        "analyzed_frames": len(raw_frames),
        "device_visible_frames": sum(int(row["device_visible"]) for row in raw_frames),
        "events": events,
        "elapsed_sec": time.perf_counter() - started,
        "thresholds": {
            "red_fraction": args.red_threshold,
            "green_fraction": args.green_threshold,
            "dark_fraction": args.dark_threshold,
            "required_lamp_components": args.required_lamp_components,
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
