# SaberVision MVP

SaberVision is an offline prototype that combines fencer detection, fencing pose estimation, coarse blade-direction estimation, score-box lamp recognition, synchronized evidence fusion, and conservative sabre adjudication.

## Repository and deployment model

- The included GitHub Pages workflow can publish the product demo and a sample verdict.
- GitHub Actions runs the deterministic fusion, lamp-onset, decision, and review-workflow tests.
- The repository contains the code and compact model weights needed for local inference.

GitHub Pages cannot execute PyTorch/CUDA or FFmpeg. Full video inference therefore runs locally with the included PowerShell launcher. Supporting video uploads and online inference will require a separate Python compute backend.

## Quick start on Windows

1. Install Python 3.10/3.11 and FFmpeg.
2. For NVIDIA GPU acceleration, install a CUDA-enabled PyTorch build compatible with your system.
3. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Process one exchange clip:

   ```powershell
   & ".\run_sabervision_mvp.ps1" -InputVideo "D:\path\exchange.mp4"
   ```

Add `-Reuse` only when rerunning the same video with the same model files. The
launcher validates cached input and model fingerprints before reusing any
inference output and stops if they do not match. It also parses cached JSON and
CSV outputs, verifies stage-specific fields, and checks videos/pages are
non-empty, so a truncated or wrong-schema stage is regenerated instead of
silently reused. Input videos and model weights include
a content hash in new run manifests. The executable Pose, blade, fusion, MVP,
and score-box sources also share an implementation fingerprint; changing the
pipeline invalidates old inference caches. Manifests are replaced atomically,
and generated video files are probed for a real video stream, dimensions, and
duration before a stage is accepted. External score-box CSVs and every
published non-log output—including Pose data/video and review images/clips—are
content-signed in each newly completed run. Every stage checkpoints its
`running`, `ok`, `failed`, or `reused` state with UTC timestamps, so an active
or interrupted job can be diagnosed directly from the on-disk manifest.

The main outputs are `SaberVision_MVP_demo.mp4`, `decision/verdict.json`, and the local review page at `decision/review_packets/review_index.html`.

Verify the signed outputs of a completed run (this also works after moving the
entire result directory):

```powershell
& ".\verify_mvp_outputs.ps1" -Manifest "D:\path\result\run_manifest.json"
```

Add `-IncludeInputs` for a full reproducibility audit that also verifies the
input video, three model files, and every external score-box CSV. The default
output-only check remains portable when those original inputs are unavailable.

To inspect the exact Python, PyTorch build, CUDA availability, GPU name,
FFmpeg tools, dependencies, and bundled weights before a long run:

```powershell
& ".\check_sabervision_environment.ps1"
```

The launchers first look for `%USERPROFILE%\.conda\envs\torch\python.exe`, then
fall back to `python` on `PATH`; use `-Python` to override that selection. The
check writes `environment_check.json`. A CUDA-enabled PyTorch installation is
tied to the selected Python environment; another environment may download its
own PyTorch even when CUDA works here.

## Run the core tests

The lightweight deterministic tests do not require model inference:

```powershell
python run_core_tests.py
```

The same command runs automatically in GitHub Actions.

## Process-label safety

The offline double-light annotation workspace removes the baked-in official
result banner from bundled contact sheets. Annotators can reveal the official
outcome only after recording the fencing phrase. The workspace records whether
the outcome was visible when a label was created or edited. Exported labels pass
through `code/mvp_v1/validate_process_labels.py`; only internally consistent,
medium/high-confidence human labels made while the outcome was hidden on a
verified redacted sheet are eligible for training.
The current v4 workspace uses an isolated, schema-versioned browser cache and
rejects legacy backup files, preventing an older unlabeled-provenance record
from being silently promoted into training truth.

## Current safety policy

- One clear automatic lamp source is enough; there is no confidence vote or N-of-3 requirement.
- Lamp colours are grouped by onset timing, so a late colour does not rewrite an earlier touch.
- Double-light priority is automatic only when the 0.35 s and 0.75 s lamp-anchored speed margins both exceed 1.70 body-heights/s in the same direction.
- Missing or genuinely conflicting priority evidence remains review-required.
- Non-finite external timestamps, motion values, and blade coordinates are ignored instead of entering the timeline or producing an automatic award.
- Review and demo rendering use sanitized packet indices and ignore malformed overlay coordinates, so bad display-only evidence cannot overwrite files or abort an otherwise reviewable case.

This is an engineering prototype, not an officially certified fencing referee system.
