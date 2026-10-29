from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from audit_three_region_signals_v48 import REGIONS, SIDES, majority, make_events


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-threshold saved V4.8 three-region colour features.")
    parser.add_argument("frames", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--side-delta", type=float, default=0.055)
    parser.add_argument("--center-delta", type=float, default=0.035)
    parser.add_argument("--side-margin", type=float, default=0.015)
    parser.add_argument("--center-margin", type=float, default=0.010)
    parser.add_argument("--center-left-margin", type=float)
    parser.add_argument("--center-right-margin", type=float)
    args = parser.parse_args()
    rows = read_csv(args.frames)
    raw_positive_frames = 0
    for row in rows:
        supporting = []
        region_states = []
        for region in REGIONS:
            detections = {}
            for side in SIDES:
                fraction_text = row.get(f"{region}_{side}_fraction", "")
                baseline_text = row.get(f"{region}_{side}_baseline", "")
                if not fraction_text or not baseline_text:
                    detections[side] = False
                    continue
                fraction = float(fraction_text)
                baseline = float(baseline_text)
                delta = fraction - baseline
                threshold = args.center_delta if region == "center_box" else args.side_delta
                if region == "center_box" and side == "left" and args.center_left_margin is not None:
                    margin = args.center_left_margin
                elif region == "center_box" and side == "right" and args.center_right_margin is not None:
                    margin = args.center_right_margin
                else:
                    margin = args.center_margin if region == "center_box" else args.side_margin
                detections[side] = delta >= threshold and fraction >= threshold + margin
                row[f"{region}_{side}_rethreshold_raw"] = int(detections[side])
            state = (
                "double" if detections["left"] and detections["right"]
                else "left" if detections["left"]
                else "right" if detections["right"]
                else "none"
            )
            row[f"{region}_rethreshold_state"] = state
            if state != "none":
                supporting.append(region)
                region_states.append(state)
        fused = (
            "double" if "double" in region_states or {"left", "right"} <= set(region_states)
            else region_states[0] if region_states
            else "none"
        )
        row["fused_raw_state"] = fused
        row["supporting_regions"] = ";".join(supporting)
        raw_positive_frames += fused != "none"
    for side in SIDES:
        filtered = majority([row["fused_raw_state"] in (side, "double") for row in rows])
        for row, value in zip(rows, filtered):
            row[f"{side}_on"] = int(value)
    for row in rows:
        row["fused_state"] = (
            "double" if row["left_on"] and row["right_on"]
            else "left" if row["left_on"]
            else "right" if row["right_on"]
            else "none"
        )
    events = make_events(rows)
    for event in events:
        event_rows = [
            row for row in rows
            if float(event["start_sec"]) - 1e-6 <= float(row["time_sec"]) <= float(event["end_sec"]) + 1e-6
        ]
        counts = {
            (region, side): sum(int(row.get(f"{region}_{side}_rethreshold_raw", 0)) for row in event_rows)
            for region in REGIONS for side in SIDES
        }
        peak_deltas = {
            (region, side): max(
                (
                    float(row.get(f"{region}_{side}_delta") or 0.0)
                    for row in event_rows
                ),
                default=0.0,
            )
            for region in REGIONS for side in SIDES
        }
        center_left_frames = counts[("center_box", "left")]
        center_right_frames = counts[("center_box", "right")]
        center_left = center_left_frames >= 2
        center_right = center_right_frames >= 2
        center_anchor_state = (
            "double" if center_left and center_right
            else "left" if center_left
            else "right" if center_right
            else "none"
        )
        # The centre must confirm at least one colour.  A missing second colour
        # may be restored only when both remote panels independently show it
        # for at least five analyzed frames.  This preserves real double hits
        # without allowing side-only colour changes to create an event.
        # The red tower is systematically dimmer in this broadcast.  When the
        # centre clearly confirms green and the multi-view aggregate is double,
        # retain the red side instead of downgrading a visible double hit.
        consensus_left = center_left or (
            center_right and event["state"] == "double"
        )
        consensus_right = center_right or (
            center_anchor_state != "none"
            and counts[("left_panel", "right")] >= 5
            and counts[("right_panel", "right")] >= 5
            and peak_deltas[("left_panel", "right")] >= 0.18
            and peak_deltas[("right_panel", "right")] >= 0.18
        )
        consensus_state = (
            "double" if consensus_left and consensus_right
            else "left" if consensus_left
            else "right" if consensus_right
            else "none"
        )
        event["aggregate_state"] = event["state"]
        event["center_anchor_state"] = center_anchor_state
        event["multiview_confirmed_state"] = consensus_state
        event["center_left_frames"] = center_left_frames
        event["center_right_frames"] = center_right_frames
        event["left_panel_left_frames"] = counts[("left_panel", "left")]
        event["left_panel_right_frames"] = counts[("left_panel", "right")]
        event["right_panel_left_frames"] = counts[("right_panel", "left")]
        event["right_panel_right_frames"] = counts[("right_panel", "right")]
        for region in ("left_panel", "right_panel"):
            for side in SIDES:
                event[f"{region}_{side}_peak_delta"] = round(peak_deltas[(region, side)], 6)
        if consensus_state != "none":
            event["state"] = consensus_state
    args.output.mkdir(parents=True, exist_ok=True)
    event_fields = [
        "state", "start_sec", "end_sec", "duration_sec", "supporting_regions",
        "aggregate_state", "center_anchor_state", "multiview_confirmed_state",
        "center_left_frames", "center_right_frames", "left_panel_left_frames",
        "left_panel_right_frames", "right_panel_left_frames", "right_panel_right_frames",
        "left_panel_left_peak_delta", "left_panel_right_peak_delta",
        "right_panel_left_peak_delta", "right_panel_right_peak_delta",
    ]
    with (args.output / "three_region_events.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(events[0]) if events else event_fields)
        writer.writeheader()
        writer.writerows(events)
    summary = {
        "frames": len(rows),
        "raw_positive_frames": raw_positive_frames,
        "events": len(events),
        "side_delta": args.side_delta,
        "center_delta": args.center_delta,
        "side_margin": args.side_margin,
        "center_margin": args.center_margin,
        "center_left_margin": args.center_left_margin,
        "center_right_margin": args.center_right_margin,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
