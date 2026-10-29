from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def state(row: dict[str, str]) -> str:
    red = row.get("red_on") == "1"
    green = row.get("green_on") == "1"
    return "double" if red and green else "left" if red else "right" if green else "none"


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine low-threshold lamp windows for visual review only.")
    parser.add_argument("frames", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--maximum-gap", type=float, default=0.50)
    parser.add_argument("--minimum-peak", type=float, default=0.018)
    args = parser.parse_args()

    with args.frames.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if state(row) != "none"]

    groups: list[list[dict[str, str]]] = []
    for row in rows:
        timestamp = float(row["time_sec"])
        if groups and timestamp - float(groups[-1][-1]["time_sec"]) <= args.maximum_gap:
            groups[-1].append(row)
        else:
            groups.append([row])

    events = []
    for group in groups:
        strongest = max(group, key=lambda row: float(row["peak_score"]))
        peak = float(strongest["peak_score"])
        if peak < args.minimum_peak:
            continue
        states = {state(row) for row in group}
        event_state = "double" if "double" in states or {"left", "right"} <= states else next(iter(states))
        start = float(group[0]["time_sec"])
        end = float(group[-1]["time_sec"])
        events.append(
            {
                "state": event_state,
                "start_sec": f"{start:.4f}",
                "end_sec": f"{end:.4f}",
                "duration_sec": f"{max(0.0, end - start):.4f}",
                "peak_score": f"{peak:.5f}",
                "peak_time": strongest["time_sec"],
                "positive_frames": len(group),
                "review_only": 1,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "state",
        "start_sec",
        "end_sec",
        "duration_sec",
        "peak_score",
        "peak_time",
        "positive_frames",
        "review_only",
    ]
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)
    print(json.dumps({"review_events": len(events), "events": events}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
