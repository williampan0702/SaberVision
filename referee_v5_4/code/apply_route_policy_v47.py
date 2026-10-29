from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SAFE_ROUTES = {
    "primary_v46",
    "primary_flow_v472",
    "primary_flow_v473",
    "triple_panel",
    "dual_scoreboard",
    "structure_assisted",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the v4.7 locator safety gate to lamp events.")
    parser.add_argument("route_json", type=Path)
    parser.add_argument("candidate_events", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    route_data = json.loads(args.route_json.read_text(encoding="utf-8-sig"))
    route = route_data["route"]
    with args.candidate_events.open("r", encoding="utf-8-sig", newline="") as handle:
        candidates = list(csv.DictReader(handle))
    accepted = candidates if route in SAFE_ROUTES else []
    fields = list(candidates[0].keys()) if candidates else [
        "state", "start_sec", "end_sec", "duration_sec", "peak_score"
    ]
    if "route" not in fields:
        fields.append("route")
    rows = [{**row, "route": route} for row in accepted]
    with (args.output / "final_events.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "route": route,
        "safety_gate": "enabled" if route in SAFE_ROUTES else "blocked",
        "candidate_events": len(candidates),
        "final_events": len(rows),
        "reason": route_data.get("reason", ""),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
