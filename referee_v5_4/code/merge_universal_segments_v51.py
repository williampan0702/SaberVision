from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FINAL_FIELDS = [
    "event_id", "state", "start_sec", "end_sec", "duration_sec", "peak_score",
    "confidence_tier", "route", "sources", "segments",
]
REVIEW_FIELDS = FINAL_FIELDS + ["review_reason"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def state_union(a: str, b: str) -> str:
    if a == b:
        return a
    if "double" in {a, b} or {a, b} == {"left", "right"}:
        return "double"
    return a


def merge_events(rows: list[dict[str, str]], tolerance: float) -> list[dict[str, str]]:
    atoms = []
    for row in rows:
        try:
            start = float(row["start_sec"])
            end = float(row["end_sec"])
        except (KeyError, ValueError):
            continue
        atoms.append({**row, "_start": start, "_end": end})
    atoms.sort(key=lambda row: (row["_start"], row["_end"]))
    merged: list[dict[str, str]] = []
    for atom in atoms:
        if merged and atom["_start"] <= float(merged[-1]["end_sec"]) + tolerance:
            current = merged[-1]
            current["state"] = state_union(current["state"], atom["state"])
            current["end_sec"] = f"{max(float(current['end_sec']), atom['_end']):.4f}"
            current["start_sec"] = f"{min(float(current['start_sec']), atom['_start']):.4f}"
            current["duration_sec"] = f"{float(current['end_sec']) - float(current['start_sec']):.4f}"
            current["peak_score"] = f"{max(float(current.get('peak_score') or 0), float(atom.get('peak_score') or 0)):.5f}"
            current["segments"] = ";".join(sorted(set(filter(None, (current.get("segments", "") + ";" + atom.get("segments", "")).split(";")))))
            current["route"] = ";".join(sorted(set(filter(None, (current.get("route", "") + ";" + atom.get("route", "")).split(";")))))
            current["sources"] = ";".join(sorted(set(filter(None, (current.get("sources", "") + ";" + atom.get("sources", "")).split(";")))))
            if "review_reason" in atom:
                current["review_reason"] = ";".join(sorted(set(filter(None, (current.get("review_reason", "") + ";" + atom.get("review_reason", "")).split(";")))))
        else:
            merged.append({key: value for key, value in atom.items() if not key.startswith("_")})
    for index, row in enumerate(merged, 1):
        row["event_id"] = index
    return merged


def overlaps(a: dict[str, str], b: dict[str, str], tolerance: float = 0.5) -> bool:
    return float(a["start_sec"]) <= float(b["end_sec"]) + tolerance and float(a["end_sec"]) >= float(b["start_sec"]) - tolerance


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge overlapping V5.2 segment outputs.")
    parser.add_argument("segments_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--merge-tolerance", type=float, default=0.65)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    final_atoms = []
    review_atoms = []
    route_rows = []
    for segment in sorted(path for path in args.segments_dir.iterdir() if path.is_dir()):
        run_path = segment / "run_summary.json"
        if not run_path.exists():
            continue
        run = json.loads(run_path.read_text(encoding="utf-8-sig"))
        route_rows.append({
            "segment": segment.name,
            "start_sec": run.get("start_sec", ""),
            "end_sec": run.get("end_sec", ""),
            "route": run.get("selected_route", ""),
        })
        for row in read_csv(segment / "final" / "final_events.csv"):
            final_atoms.append({**row, "segments": segment.name})
        for row in read_csv(segment / "final" / "review_events.csv"):
            review_atoms.append({**row, "segments": segment.name})

    final_rows = merge_events(final_atoms, args.merge_tolerance)
    review_rows = merge_events(review_atoms, args.merge_tolerance)
    review_rows = [row for row in review_rows if not any(overlaps(row, final) for final in final_rows)]
    for index, row in enumerate(review_rows, 1):
        row["event_id"] = index

    with (args.output / "final_events.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FINAL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_rows)
    with (args.output / "review_events.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(review_rows)
    with (args.output / "route_timeline.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["segment", "start_sec", "end_sec", "route"])
        writer.writeheader()
        writer.writerows(route_rows)
    summary = {
        "version": "5.4-physical-fusion-segmented",
        "segments": len(route_rows),
        "routes": {route: sum(row["route"] == route for row in route_rows) for route in sorted({row["route"] for row in route_rows})},
        "automatic_events": len(final_rows),
        "review_events": len(review_rows),
        "overlap_policy": "overlapping windows are de-duplicated; automatic events supersede overlapping review candidates",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
