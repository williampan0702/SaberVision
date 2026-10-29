from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SAFE_ROUTES = {
    "triple_panel",
    "dual_scoreboard",
    "primary_v46",
    "primary_flow_v472",
    "primary_flow_v473",
    "structure_assisted",
}

FIELDS = [
    "event_id",
    "state",
    "start_sec",
    "end_sec",
    "duration_sec",
    "peak_score",
    "confidence_tier",
    "route",
    "sources",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def route_quality(route_data: dict) -> tuple[bool, str]:
    route = route_data.get("route", "no_visible_device")
    if route not in SAFE_ROUTES:
        return False, "定位路线未通过自动安全门"
    if route.startswith("primary_flow"):
        quality = route_data.get("primary_flow_quality", {})
        if not quality.get("qualified", False):
            return False, "光流轨迹质量门未通过"
    metrics = route_data.get("metrics", {})
    minimum = int(route_data.get("thresholds", {}).get("minimum_valid_frames", 20))
    if route == "triple_panel" and metrics.get("triple_panel", {}).get("valid_frames", 0) < minimum:
        return False, f"三区域可靠帧不足 {minimum}"
    if route == "dual_scoreboard" and metrics.get("dual_scoreboard", {}).get("valid_frames", 0) < minimum:
        return False, f"双比分屏可靠帧不足 {minimum}"
    if route == "structure_assisted" and metrics.get("structure_assisted", {}).get("valid_frames", 0) < minimum:
        return False, f"结构辅助可靠帧不足 {minimum}"
    return True, "定位路线通过自动安全门"


def confidence(row: dict[str, str], route: str) -> str:
    peak = float(row.get("peak_score") or 0.0)
    duration = float(row.get("duration_sec") or 0.0)
    sources = row.get("sources", "")
    if route == "structure_assisted":
        # This route's box can include persistent scoreboard colours.  Until
        # it has independent centre-device confirmation, keep all events in
        # the review table even when the locator geometry itself is stable.
        return "review"
    if route == "triple_panel" and "three_region" in sources:
        regions = set(filter(None, row.get("supporting_regions", "").split(";")))
        if "physical" in sources or int(row.get("physical_candidates") or 0) > 0 or len(regions) == 3:
            return "high"
    if route == "dual_scoreboard" and duration >= 0.20 and peak >= 0.008:
        # The dual-scoreboard scanner already requires a compact filled lamp
        # block in the physical device band.  Real sabre lamps in recent FIE
        # broadcasts can remain visible for only 5-6 source frames, so the
        # generic 0.018 score floor incorrectly relegated clear short flashes
        # to review (notably the Tunis exchange clips).
        return "medium"
    if route.startswith("primary") and "physical_temporal" in sources and duration >= 0.25 and peak >= 0.006:
        # The v5.4 temporal fallback measures the newly illuminated colour
        # fraction after subtracting the per-clip baseline.  Its score is
        # deliberately smaller than the absolute-colour model score.
        return "medium"
    if peak >= 0.12 or (duration >= 0.35 and peak >= 0.035):
        return "high"
    if duration >= 0.12 and peak >= 0.018:
        return "medium"
    return "review"


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize and safety-gate V5.0 multi-route lamp events.")
    parser.add_argument("route_json", type=Path)
    parser.add_argument("candidate_events", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    route_data = json.loads(args.route_json.read_text(encoding="utf-8-sig"))
    route = route_data.get("route", "no_visible_device")
    qualified, quality_reason = route_quality(route_data)
    candidates = read_rows(args.candidate_events)
    accepted = []
    review = []
    for row in candidates:
        state = (row.get("state") or "none").lower()
        if state not in {"left", "right", "double"}:
            continue
        start = float(row.get("start_sec") or 0.0)
        end = float(row.get("end_sec") or start)
        normalized = {
            "event_id": 0,
            "state": state,
            "start_sec": f"{start:.4f}",
            "end_sec": f"{end:.4f}",
            "duration_sec": f"{max(0.0, end - start):.4f}",
            "peak_score": f"{float(row.get('peak_score') or 0.0):.5f}",
            "confidence_tier": confidence(row, route),
            "route": route,
            "sources": row.get("sources") or route,
        }
        if qualified and normalized["confidence_tier"] != "review":
            accepted.append(normalized)
        else:
            review.append({**normalized, "review_reason": quality_reason if not qualified else "证据强度不足"})
    for index, row in enumerate(accepted, 1):
        row["event_id"] = index
    for index, row in enumerate(review, 1):
        row["event_id"] = index

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "final_events.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(accepted)
    with (args.output / "review_events.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS + ["review_reason"])
        writer.writeheader()
        writer.writerows(review)
    summary = {
        "version": "5.4-physical-fusion",
        "route": route,
        "route_qualified": qualified,
        "route_quality_reason": quality_reason,
        "candidate_events": len(candidates),
        "automatic_events": len(accepted),
        "review_events": len(review),
        "safety_policy": "低质量定位或弱证据只进入 review_events.csv，不进入 final_events.csv",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
