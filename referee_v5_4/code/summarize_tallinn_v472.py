from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Tallinn blind-test and v4.7.2 optimization summary.")
    parser.add_argument("blind_route", type=Path)
    parser.add_argument("track_summary", type=Path)
    parser.add_argument("quality", type=Path)
    parser.add_argument("lamp_summary", type=Path)
    parser.add_argument("reviewed_summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    blind = load(args.blind_route)
    track = load(args.track_summary)
    quality = load(args.quality)
    lamp = load(args.lamp_summary)
    reviewed = load(args.reviewed_summary)
    result = {
        "venue": "Tallinn 2025 U23 European Championships",
        "evaluation_protocol": {
            "blind_v471_frozen": True,
            "blind_route": blind["route"],
            "blind_primary_coverage": blind["metrics"]["primary_v46"]["fraction"],
            "optimization_version": "v4.7.2",
            "note": "All v4.7.2 changes were made after the v4.7.1 blind outcome was frozen.",
        },
        "locator_v472": {
            "route": "primary_flow_v472",
            "qualified": quality["qualified"],
            "stable_rows": track["stable_rows"],
            "sampled_frames": track["sampled_frames"],
            "full_video_coverage": track["coverage"],
            "source_counts": track["source_counts"],
            "scene_cuts": track.get("scene_cuts"),
            "acquisition_resets": track.get("acquisition_resets"),
            "quality_metrics": quality["metrics"],
            "warning": "Full-video coverage includes timeouts, side angles and ending footage; it is not lamp accuracy.",
        },
        "lamp_audit": {
            "strict_original_events": len(lamp.get("events", [])),
            "reviewed_true_events": reviewed["reviewed_events"],
            "automatic_high_confidence": reviewed["automatic_high_confidence"],
            "review_only": reviewed["review_only"],
            "warning": reviewed["warning"],
        },
        "status": "locator branch validated on this venue; lamp recall remains incomplete without frame-level ground truth",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
