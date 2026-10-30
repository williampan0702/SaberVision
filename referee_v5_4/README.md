# SaberVision Score-Box Recognition V5.4

## Overview

V5.4 adds a tight physical score-box detector and fuses it with the two V5.3 locator models. Once a reliable central physical score box is located, the system combines learned lamp recognition with temporal red/green change detection and merges fragments into `LEFT`, `RIGHT`, or `DOUBLE` events.

Long videos are processed in time segments. Scene layout is re-evaluated for every segment, allowing the system to handle camera movement, cuts, and venue changes. Unstable localization is sent to review instead of being forced into an automatic result.

"Universal" refers to multi-layout routing, three-model fusion, and conservative rejection. It does not imply perfect recall at unseen venues.

## Recommended entry points

For a full match or a video containing multiple matches:

```powershell
& ".\run_segmented_universal_v52.ps1" `
  -InputVideo "D:\videos\full_match.mp4" `
  -OutputDir "D:\outputs\full_match_v54" `
  -NoPreview
```

For a pre-cut exchange clip:

```powershell
& ".\run_universal_referee_v52.ps1" `
  -InputVideo "D:\videos\exchange.mp4" `
  -OutputDir "D:\outputs\exchange_v54" `
  -NoPreview
```

These launchers call the required internal scripts automatically.

## Processing pipeline

1. The segmented launcher re-detects the scene every 60 seconds with a 2-second overlap.
2. The physical score-box model, V5.3 main model, and legacy fallback model produce candidate boxes.
3. Routing uses candidate density, location, size, and multi-zone structure; a reliable central physical score box has priority.
4. Stable-box tracks are interpolated and smoothed instead of treating each frame independently.
5. Learned lamp recognition and temporal red/green changes against a historical baseline run inside the box.
6. Short fragments and left/right evidence from one exchange are merged into `LEFT`, `RIGHT`, or `DOUBLE`.
7. Reliable events go to `final_events.csv`; weak, low-coverage, or strongly cut candidates go to `review_events.csv`.

## Validation snapshot

- The physical score-box locator was trained for 25 epochs: validation precision 0.954, recall 0.908, mAP50 0.965, and mAP50-95 0.883.
- On a 74-frame Tunis tight-box holdout, the new model reached IoU >= 0.5 on 74/74 frames with mean IoU 0.975. The previous main model reached 0/74 with mean IoU 0.005.
- Exchange-level integration tests outside the training interval automatically produced final events for exchanges 1, 4, 13, and 22. Exchange 7 was conservatively sent to review because of a strong cut and occlusion.
- In the exchange 1 demo, the central box stays tight around the physical device. The left lamp was detected around 62.80-68.00 s, close to the manual 63.057-68.144 s interval.
- The 60-frame Seoul and Tbilisi samples only checked whether each model produced boxes; they are not accuracy measurements. Three-model fusion remains enabled because the models cover different layouts.

The Tunis training interval and holdout interval come from the same match and venue. The 74/74 result must not be presented as perfect cross-venue accuracy.

## Outputs

- `final/final_events.csv`: automatic events that passed localization and event safety gates
- `final/review_events.csv`: candidates requiring fast human confirmation
- `final/route_timeline.csv`: route selected for each time segment
- `final/summary.json`: routing, automatic-event, and review-event statistics
- `segments/segment_*`: localization tracks, candidates, and quality reports per segment
- `final_boxed_demo.mp4` or `final_route_demo.mp4`: optional preview output

Downstream automation must read only `final_events.csv`. Do not treat `review_events.csv` as ground truth.

## Current limitations

- Very small devices, complete athlete occlusion, severe compression, or cuts immediately around lamp onset can still cause review cases or missed lamps.
- Temporal color compensation requires a short unlit baseline within the same shot; clips lit from the first frame are harder.
- Three-model fusion broadens venue coverage, but extremely distant devices remain a weak case.
- Box coverage and candidate counts are not final accuracy metrics. Formal cross-venue accuracy requires exchange-level lamp labels and timing at additional test venues.
