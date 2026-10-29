from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final Tallinn v4.7.3 validation summary.")
    parser.add_argument("blind_route", type=Path)
    parser.add_argument("flow_summary", type=Path)
    parser.add_argument("interpolation_summary", type=Path)
    parser.add_argument("interpolation_quality", type=Path)
    parser.add_argument("lamp_summary", type=Path)
    parser.add_argument("reviewed_summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    blind = load(args.blind_route)
    flow = load(args.flow_summary)
    interpolation = load(args.interpolation_summary)
    quality = load(args.interpolation_quality)
    lamp = load(args.lamp_summary)
    reviewed = load(args.reviewed_summary)
    result = {
        "venue": "Tallinn 2025 U23 European Championships",
        "evaluation_protocol": {
            "blind_v471_frozen": True,
            "blind_route": blind["route"],
            "blind_primary_coverage": blind["metrics"]["primary_v46"]["fraction"],
            "note": "v4.7.2/v4.7.3 changes were made only after the v4.7.1 blind result was frozen.",
        },
        "locator": {
            "route": "primary_flow_v473",
            "qualified": quality["qualified"],
            "flow_only_coverage": flow["coverage"],
            "final_coverage": interpolation["coverage"],
            "original_rows": interpolation["original_rows"],
            "interpolated_rows": interpolation["interpolated_rows"],
            "final_rows": interpolation["final_rows"],
            "accepted_gap_count": len(interpolation["accepted_gaps"]),
            "rejected_gap_count": len(interpolation["rejected_gaps"]),
            "quality_metrics": quality["metrics"],
            "safety_rule": interpolation["settings"],
        },
        "lamp_audit": {
            "strict_events": len(lamp.get("events", [])),
            "reviewed_true_events": reviewed["reviewed_true_events"],
            "automatic_high_confidence": reviewed["automatic_high_confidence"],
            "review_only": reviewed["review_only"],
            "score_changes_total": reviewed["score_changes_total"],
            "score_changes_matched": reviewed["score_changes_matched"],
            "score_changes_unmatched": reviewed["score_changes_unmatched"],
            "warning": reviewed["warning"],
        },
        "status": "new central-device route is integrated; unmatched score windows remain for future lamp-recall training",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
