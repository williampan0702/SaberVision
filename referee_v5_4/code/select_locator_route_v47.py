from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def count_and_fraction(summary: dict, count_key: str, total: int) -> tuple[int, float]:
    count = int(summary.get(count_key, 0))
    fraction = summary.get("valid_fraction")
    if fraction is None:
        fraction = count / max(1, total)
    return count, float(fraction)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a conservative v4.7 locator branch from probe statistics.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--sampled-frames", type=int, required=True)
    parser.add_argument("--primary-summary", type=Path)
    parser.add_argument("--triple-summary", type=Path)
    parser.add_argument("--dual-scoreboard-summary", type=Path)
    parser.add_argument("--structure-summary", type=Path)
    parser.add_argument("--triple-threshold", type=float, default=0.12)
    parser.add_argument("--dual-scoreboard-threshold", type=float, default=0.35)
    parser.add_argument("--dual-minimum-primary-frames", type=int, default=3)
    parser.add_argument("--primary-threshold", type=float, default=0.35)
    parser.add_argument("--structure-threshold", type=float, default=0.15)
    parser.add_argument("--review-threshold", type=float, default=0.08)
    parser.add_argument("--minimum-valid-frames", type=int, default=20)
    args = parser.parse_args()

    primary_summary = load(args.primary_summary)
    triple_summary = load(args.triple_summary)
    dual_scoreboard_summary = load(args.dual_scoreboard_summary)
    structure_summary = load(args.structure_summary)
    primary_count, primary_fraction = count_and_fraction(
        primary_summary, "stable_rows", args.sampled_frames
    )
    primary_raw_count = int(primary_summary.get("anchors", 0))
    primary_center_x = float(primary_summary.get("median_center_x_norm", -1.0))
    primary_center_y = float(primary_summary.get("median_center_y_norm", -1.0))
    primary_area = float(primary_summary.get("median_area_fraction", -1.0))
    central_physical_track = (
        0.34 <= primary_center_x <= 0.66
        and 0.42 <= primary_center_y <= 0.82
        and 0.002 <= primary_area <= 0.055
    )
    triple_count, triple_fraction = count_and_fraction(
        triple_summary, "valid_frames", args.sampled_frames
    )
    dual_scoreboard_count, dual_scoreboard_fraction = count_and_fraction(
        dual_scoreboard_summary, "valid_frames", args.sampled_frames
    )
    structure_count, structure_fraction = count_and_fraction(
        structure_summary, "valid_frames", args.sampled_frames
    )

    if triple_count >= args.minimum_valid_frames and triple_fraction >= args.triple_threshold:
        route = "triple_panel"
        reason = "symmetric side panels reliably imply a middle physical scoring device"
    elif (
        primary_count >= args.minimum_valid_frames
        and primary_fraction >= args.primary_threshold
        and central_physical_track
    ):
        route = "primary_v46"
        reason = "a dense learned track is centered on a plausible physical scoring device"
    elif (
        dual_scoreboard_count >= args.minimum_valid_frames
        and dual_scoreboard_fraction >= args.dual_scoreboard_threshold
        and (
            primary_count >= args.dual_minimum_primary_frames
            or (
                args.sampled_frames <= 20
                and dual_scoreboard_fraction >= 0.90
                and primary_raw_count >= args.dual_minimum_primary_frames
            )
        )
    ):
        route = "dual_scoreboard"
        if primary_count >= args.dual_minimum_primary_frames:
            reason = "scoreboard geometry plus stable learned support confirm a dual-scoreboard layout"
        else:
            reason = "short-clip high-coverage scoreboard geometry plus learned detections confirm the device"
    elif primary_count >= args.minimum_valid_frames and primary_fraction >= args.primary_threshold:
        route = "primary_v46"
        reason = "the learned device locator forms a sufficiently dense stable track"
    elif structure_count >= args.minimum_valid_frames and structure_fraction >= args.structure_threshold:
        route = "structure_assisted"
        reason = "primary coverage is limited but dual-confirmed device structure is repeatable"
    elif primary_fraction >= args.review_threshold:
        route = "review_required"
        reason = "some primary track exists, but coverage is too sparse for unattended recognition"
    else:
        route = "no_visible_device"
        reason = "no locator branch forms a reliable continuous physical-device track"

    result = {
        "route": route,
        "reason": reason,
        "sampled_frames": args.sampled_frames,
        "metrics": {
            "triple_panel": {"valid_frames": triple_count, "fraction": triple_fraction},
            "dual_scoreboard": {
                "valid_frames": dual_scoreboard_count,
                "fraction": dual_scoreboard_fraction,
            },
            "primary_v46": {
                "valid_frames": primary_count,
                "fraction": primary_fraction,
                "raw_anchor_frames": primary_raw_count,
                "median_center_x_norm": primary_center_x,
                "median_center_y_norm": primary_center_y,
                "median_area_fraction": primary_area,
                "central_physical_track": central_physical_track,
            },
            "structure_assisted": {"valid_frames": structure_count, "fraction": structure_fraction},
        },
        "thresholds": {
            "triple_panel": args.triple_threshold,
            "dual_scoreboard": args.dual_scoreboard_threshold,
            "dual_minimum_primary_frames": args.dual_minimum_primary_frames,
            "primary_v46": args.primary_threshold,
            "structure_assisted": args.structure_threshold,
            "review_required": args.review_threshold,
            "minimum_valid_frames": args.minimum_valid_frames,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
