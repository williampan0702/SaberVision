from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scan_dual_scoreboard_lamps_v47 import events_from_rows, majority_filter
from validate_dual_scoreboard_lamp_events_v47 import read_csv


def merge_fragments(events: list[dict], maximum_gap: float) -> list[dict]:
    merged: list[dict] = []
    for event in events:
        if merged and float(event["start_sec"]) - float(merged[-1]["end_sec"]) <= maximum_gap:
            previous = merged[-1]
            states = {previous["state"], event["state"]}
            if len(states) == 1 or "double" in states:
                previous["state"] = "double" if "double" in states else previous["state"]
                previous["end_sec"] = event["end_sec"]
                previous["duration_sec"] = round(
                    float(previous["end_sec"]) - float(previous["start_sec"]), 4
                )
                previous["peak_score"] = f"{max(float(previous['peak_score']), float(event['peak_score'])):.6f}"
                previous["fragments"] = int(previous.get("fragments", 1)) + 1
                continue
        merged.append({**event, "fragments": 1})
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-threshold cached v4.7 lamp-frame measurements.")
    parser.add_argument("frames", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--red-threshold", type=float, default=0.03)
    parser.add_argument("--green-threshold", type=float, default=0.01)
    parser.add_argument("--dark-threshold", type=float, default=0.16)
    parser.add_argument("--required-lamp-components", type=int, default=3)
    parser.add_argument("--event-gap", type=float, default=0.5)
    parser.add_argument("--minimum-event", type=float, default=0.12)
    parser.add_argument("--fragment-gap", type=float, default=2.5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.frames)
    for row in rows:
        red_fraction = float(row["red_fraction"])
        green_fraction = float(row["green_fraction"])
        dark_fraction = float(row["dark_fraction"])
        components = int(round(float(row["lamp_components"])))
        visible = dark_fraction >= args.dark_threshold and (
            components >= args.required_lamp_components
            or red_fraction >= args.red_threshold
            or green_fraction >= args.green_threshold
        )
        row["device_visible"] = int(visible)
        row["red_raw"] = int(visible and red_fraction >= args.red_threshold)
        row["green_raw"] = int(visible and green_fraction >= args.green_threshold)

    red_filtered = majority_filter([bool(int(row["red_raw"])) for row in rows])
    green_filtered = majority_filter([bool(int(row["green_raw"])) for row in rows])
    for index, row in enumerate(rows):
        red_on = red_filtered[index]
        green_on = green_filtered[index]
        row["red_on"] = int(red_on)
        row["green_on"] = int(green_on)
        row["state"] = "double" if red_on and green_on else "left" if red_on else "right" if green_on else "none"

    raw_events = events_from_rows(rows, args.event_gap, args.minimum_event)
    events = merge_fragments(raw_events, args.fragment_gap)
    fields = ["state", "start_sec", "end_sec", "duration_sec", "peak_score", "fragments"]
    with (args.output / "lamp_events_rethresholded.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)
    summary = {
        "frames": len(rows),
        "raw_events": len(raw_events),
        "merged_events": len(events),
        "events": events,
        "thresholds": {
            "red": args.red_threshold,
            "green": args.green_threshold,
            "dark": args.dark_threshold,
            "event_gap": args.event_gap,
            "fragment_gap": args.fragment_gap,
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
