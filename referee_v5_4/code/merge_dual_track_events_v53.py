from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = ["state", "start_sec", "end_sec", "duration_sec", "peak_score"]


def read_events(path: Path, source: str) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    events = []
    for row in rows:
        events.append(
            {
                "state": row["state"],
                "start": float(row["start_sec"]),
                "end": float(row["end_sec"]),
                "peak": float(row.get("peak_score") or 0.0),
                "sources": {source},
            }
        )
    return events


def combine_state(states: set[str]) -> str:
    if "double" in states or {"left", "right"} <= states:
        return "double"
    return next(iter(states))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge flow-smoothed and current-geometry dual-device events."
    )
    parser.add_argument("flow_events", type=Path)
    parser.add_argument("geometry_events", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--merge-gap", type=float, default=0.60)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    events = read_events(args.flow_events, "flow") + read_events(
        args.geometry_events, "geometry"
    )
    events.sort(key=lambda item: (item["start"], item["end"]))
    clusters: list[list[dict]] = []
    for event in events:
        if clusters and event["start"] <= max(item["end"] for item in clusters[-1]) + args.merge_gap:
            clusters[-1].append(event)
        else:
            clusters.append([event])

    merged = []
    audit = []
    for cluster in clusters:
        start = min(item["start"] for item in cluster)
        end = max(item["end"] for item in cluster)
        state = combine_state({item["state"] for item in cluster})
        sources = sorted({source for item in cluster for source in item["sources"]})
        merged.append(
            {
                "state": state,
                "start_sec": round(start, 4),
                "end_sec": round(end, 4),
                "duration_sec": round(end - start, 4),
                "peak_score": f"{max(item['peak'] for item in cluster):.6f}",
            }
        )
        audit.append(
            {
                "state": state,
                "start_sec": round(start, 4),
                "end_sec": round(end, 4),
                "sources": ";".join(sources),
                "members": len(cluster),
            }
        )

    output_csv = args.output / "dual_scoreboard_lamp_events.csv"
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(merged)
    (args.output / "dual_track_merge_summary.json").write_text(
        json.dumps(
            {
                "flow_events": len(read_events(args.flow_events, "flow")),
                "geometry_events": len(read_events(args.geometry_events, "geometry")),
                "merged_events": len(merged),
                "events": audit,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"merged_events": len(merged), "events": audit}, ensure_ascii=False))


if __name__ == "__main__":
    main()
