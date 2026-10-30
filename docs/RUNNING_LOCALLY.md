# Running SaberVision Locally

SaberVision performs full video inference locally because GitHub Pages cannot run Python, PyTorch, CUDA, OpenCV, or FFmpeg.

## Requirements

- Windows with PowerShell
- Python 3.10 or 3.11
- FFmpeg and FFprobe on `PATH`
- Dependencies from `requirements.txt`
- A CUDA-enabled PyTorch installation is optional but recommended for NVIDIA GPUs

## Process one exchange

From the repository root:

```powershell
python -m pip install -r requirements.txt
& ".\run_sabervision_mvp.ps1" -InputVideo "D:\path\exchange.mp4"
```

The main outputs are:

- `SaberVision_MVP_demo.mp4`: annotated demonstration video
- `decision/verdict.json`: complete machine-readable evidence and decision
- `decision/verdict.csv`: compact decision table
- `decision/review_packets/review_index.html`: local review page for unresolved events
- `run_manifest.json`: stage status, timing, fingerprints, and log locations

Check the environment before a long run:

```powershell
& ".\check_sabervision_environment.ps1"
```

Verify the signed outputs after a completed run:

```powershell
& ".\verify_mvp_outputs.ps1" -Manifest "D:\path\result\run_manifest.json"
```

Add `-IncludeInputs` to verify the input video, model files, and external score-box CSV files as well as generated outputs.

This is an engineering prototype, not an officially certified fencing referee system.
