from __future__ import annotations

import argparse
import bisect
import csv
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np


REGIONS = ("left_panel", "center_box", "right_panel")
SIDES = ("left", "right")


def load_track(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = []
    for row in source_rows:
        regions = {}
        for region in REGIONS:
            if row.get(f"{region}_valid") != "1":
                continue
            regions[region] = {
                "box": tuple(round(float(row[f"{region}_{key}"])) for key in ("x1", "y1", "x2", "y2")),
                "left": tuple(round(float(row[f"{region}_left_{key}"])) for key in ("x1", "y1", "x2", "y2")),
                "right": tuple(round(float(row[f"{region}_right_{key}"])) for key in ("x1", "y1", "x2", "y2")),
            }
        if regions:
            rows.append((float(row["time_sec"]), regions, row.get("source", "")))
    rows.sort(key=lambda item: item[0])
    return rows, [item[0] for item in rows]


def load_events(path: Path | None):
    if path is None:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def event_at(events, timestamp: float):
    for event in events:
        # CSV boundaries are rounded to 4 decimals while decoded frame times
        # are repeating fractions (for example 376.733333...).
        if float(event["start_sec"]) - 0.08 <= timestamp <= float(event["end_sec"]) + 0.08:
            return event
    return None


def nearest_layout(track, times, timestamp: float, maximum_gap: float):
    index = bisect.bisect_left(times, timestamp)
    candidates = []
    if index < len(track):
        candidates.append(track[index])
    if index:
        candidates.append(track[index - 1])
    if not candidates:
        return None
    nearest = min(candidates, key=lambda item: abs(item[0] - timestamp))
    return nearest if abs(nearest[0] - timestamp) <= maximum_gap else None


def colour_fraction(frame: np.ndarray, box, colour: str) -> float:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    vivid = (saturation >= 90) & (value >= 115)
    if colour == "red":
        mask = vivid & ((hue <= 15) | (hue >= 165))
    else:
        mask = vivid & (hue >= 34) & (hue <= 98)
    return float(mask.mean())


def rolling_baseline(history: deque[float], value: float) -> float:
    baseline = float(np.quantile(np.asarray(history, dtype=np.float32), 0.25)) if history else value
    history.append(value)
    return baseline


def majority(values: list[bool], radius: int = 1) -> list[bool]:
    filtered = []
    for index in range(len(values)):
        window = values[max(0, index - radius):min(len(values), index + radius + 1)]
        filtered.append(sum(window) >= 2)
    return filtered


def make_events(rows: list[dict], gap_seconds: float = 0.34, minimum_seconds: float = 0.12):
    events = []
    current = None
    for row in rows:
        timestamp = float(row["time_sec"])
        state = row["fused_state"]
        if state == "none":
            if current is not None and timestamp - current["last"] > gap_seconds:
                events.append(current)
                current = None
            continue
        if current is None or timestamp - current["last"] > gap_seconds:
            if current is not None:
                events.append(current)
            current = {"start": timestamp, "last": timestamp, "states": {state}, "regions": set(row["supporting_regions"].split(";"))}
        else:
            current["last"] = timestamp
            current["states"].add(state)
            current["regions"].update(row["supporting_regions"].split(";"))
    if current is not None:
        events.append(current)
    output = []
    for event in events:
        duration = event["last"] - event["start"]
        if duration < minimum_seconds:
            continue
        states = event["states"]
        state = "double" if "double" in states or {"left", "right"} <= states else next(iter(states))
        output.append({
            "state": state,
            "start_sec": round(event["start"], 4),
            "end_sec": round(event["last"], 4),
            "duration_sec": round(duration, 4),
            "supporting_regions": ";".join(sorted(region for region in event["regions"] if region)),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and fuse three synchronized scoring views.")
    parser.add_argument("input", type=Path)
    parser.add_argument("track", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--analysis-fps", type=float, default=15.0)
    parser.add_argument("--track-max-gap", type=float, default=0.32)
    parser.add_argument("--side-delta", type=float, default=0.055)
    parser.add_argument("--center-delta", type=float, default=0.075)
    parser.add_argument("--validated-events", type=Path)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    track, times = load_track(args.track)
    validated_events = load_events(args.validated_events)
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, round(source_fps / min(source_fps, args.analysis_fps)))
    output_fps = source_fps / stride
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)
    writer = None if args.no_preview else cv2.VideoWriter(
        str(args.output / "three_region_preview.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )

    history = defaultdict(lambda: deque(maxlen=max(12, round(output_fps * 4))))
    rows = []
    read_index = 0
    colours = {"left_panel": (255, 0, 255), "center_box": (255, 255, 0), "right_panel": (255, 0, 255)}
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
        located = nearest_layout(track, times, timestamp, args.track_max_gap)
        row = {"time_sec": f"{timestamp:.4f}", "layout_valid": int(located is not None)}
        supporting = []
        region_states = []
        if located is not None:
            _track_time, regions, source = located
            row["source"] = source
            for region_name, region in regions.items():
                detections = {}
                for side in SIDES:
                    fraction = colour_fraction(frame, region[side], "red" if side == "left" else "green")
                    baseline = rolling_baseline(history[(region_name, side)], fraction)
                    delta = fraction - baseline
                    threshold = args.center_delta if region_name == "center_box" else args.side_delta
                    detections[side] = delta >= threshold and fraction >= threshold + 0.015
                    row[f"{region_name}_{side}_fraction"] = f"{fraction:.6f}"
                    row[f"{region_name}_{side}_baseline"] = f"{baseline:.6f}"
                    row[f"{region_name}_{side}_delta"] = f"{delta:.6f}"
                    row[f"{region_name}_{side}_raw"] = int(detections[side])
                    x1, y1, x2, y2 = region[side]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colours[region_name], 1)
                state = "double" if detections["left"] and detections["right"] else "left" if detections["left"] else "right" if detections["right"] else "none"
                row[f"{region_name}_raw_state"] = state
                if state != "none":
                    supporting.append(region_name)
                    region_states.append(state)
                x1, y1, x2, y2 = region["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), colours[region_name], 2)
                cv2.putText(frame, f"{region_name}:{state}", (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colours[region_name], 1, cv2.LINE_AA)
        else:
            row["source"] = "missing"
            for region_name in REGIONS:
                for side in SIDES:
                    row[f"{region_name}_{side}_fraction"] = ""
                    row[f"{region_name}_{side}_baseline"] = ""
                    row[f"{region_name}_{side}_delta"] = ""
                    row[f"{region_name}_{side}_raw"] = 0
                row[f"{region_name}_raw_state"] = "none"
        fused = "double" if "double" in region_states or {"left", "right"} <= set(region_states) else (region_states[0] if region_states else "none")
        row["fused_raw_state"] = fused
        row["supporting_regions"] = ";".join(supporting)
        validated = event_at(validated_events, timestamp)
        row["validated_state"] = validated["state"] if validated is not None else "none"
        row["decision"] = "accepted_validated" if validated is not None else "blocked_side_only" if fused != "none" else "none"
        rows.append(row)
        cv2.rectangle(frame, (0, 0), (1050, 46), (0, 0, 0), -1)
        display_state = row["validated_state"]
        status = "BLOCKED-SIDE-ONLY" if row["decision"] == "blocked_side_only" else display_state.upper()
        cv2.putText(frame, f"V49 THREE-REGION: {status} views={len(supporting)} t={timestamp:.2f}s", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        if writer is not None:
            writer.write(frame)

    for state in ("left", "right"):
        filtered = majority([row["fused_raw_state"] in (state, "double") for row in rows])
        for row, value in zip(rows, filtered):
            row[f"{state}_on"] = int(value)
    for row in rows:
        row["fused_state"] = "double" if row["left_on"] and row["right_on"] else "left" if row["left_on"] else "right" if row["right_on"] else "none"

    cap.release()
    if writer is not None:
        writer.release()
    with (args.output / "three_region_frames.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    raw_events = make_events(rows)
    with (args.output / "three_region_events.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ("state", "start_sec", "end_sec", "duration_sec", "supporting_regions")
        csv_writer = csv.DictWriter(handle, fieldnames=fields)
        csv_writer.writeheader()
        csv_writer.writerows(raw_events)
    validated_audit = []
    for event in validated_events:
        start = float(event["start_sec"]) - 0.35
        end = float(event["end_sec"]) + 0.35
        nearby = [row for row in rows if start <= float(row["time_sec"]) <= end]
        region_support = []
        for region in REGIONS:
            states = {row[f"{region}_raw_state"] for row in nearby}
            if states - {"none"}:
                region_support.append(region)
        validated_audit.append({
            **event,
            "regional_support": ";".join(region_support),
            "regional_support_count": len(region_support),
        })
    if validated_audit:
        with (args.output / "validated_event_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            csv_writer = csv.DictWriter(handle, fieldnames=list(validated_audit[0]))
            csv_writer.writeheader()
            csv_writer.writerows(validated_audit)
    rejected_side_only = []
    for event in raw_events:
        overlaps = any(
            float(event["start_sec"]) <= float(primary["end_sec"]) + 0.35
            and float(event["end_sec"]) >= float(primary["start_sec"]) - 0.35
            for primary in validated_events
        )
        if validated_events and not overlaps:
            rejected_side_only.append(event)
    valid_frames = sum(int(row["layout_valid"]) for row in rows)
    summary = {
        "analyzed_frames": len(rows),
        "layout_valid_frames": valid_frames,
        "all_three_regions_available_frames": valid_frames,
        "raw_frame_coverage": valid_frames / max(1, len(rows)),
        "validated_events": validated_audit,
        "raw_colour_candidates": raw_events,
        "rejected_side_only_candidates": rejected_side_only,
        "fusion_rule": "validated physical-lamp events are final; side-panel-only colour changes are review-only",
        "note": "raw_frame_coverage includes replay/close-up frames where the three devices are not visible",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
