param(
    [Parameter(Mandatory = $true)]
    [string]$InputVideo,
    [Parameter(Mandatory = $true)]
    [string]$OutputDir,
    [double]$Start = 0,
    [Parameter(Mandatory = $true)]
    [double]$End,
    [double]$ProbeFps = 2,
    [string]$Python = "python",
    [string]$LocatorModel = "",
    [string]$FallbackLocatorModel = "",
    [string]$LegacyLocatorModel = ""
)

$ErrorActionPreference = "Stop"
$scriptDir = $PSScriptRoot
$packageRoot = Split-Path $scriptDir -Parent
if (-not $LocatorModel) {
    $LocatorModel = Join-Path $packageRoot "models\score_device_v54_physical.pt"
}
if (-not (Test-Path -LiteralPath $LocatorModel)) {
    throw "Score-box locator model not found: $LocatorModel"
}

$anchorDir = Join-Path $OutputDir "01_anchors"
$fallbackAnchorDir = Join-Path $OutputDir "01_anchors_fallback"
$mergedAnchorDir = Join-Path $OutputDir "01_anchors_merged"
$legacyAnchorDir = Join-Path $OutputDir "01_anchors_legacy"
$allMergedAnchorDir = Join-Path $OutputDir "01_anchors_all_merged"
$primaryDir = Join-Path $OutputDir "02_primary_track"
$primaryFlowDir = Join-Path $OutputDir "02_primary_flow_v472"
$primaryInterpolationDir = Join-Path $OutputDir "02_primary_flow_v473"
$tripleDir = Join-Path $OutputDir "03_triple_panel"
$dualScoreboardDir = Join-Path $OutputDir "04_dual_scoreboard"
$dualScoreboardFlowDir = Join-Path $OutputDir "04_dual_scoreboard_flow"
$structureDir = Join-Path $OutputDir "04_structure_candidates"
$structureTrackDir = Join-Path $OutputDir "05_structure_track"
$routePath = Join-Path $OutputDir "locator_route.json"
New-Item -ItemType Directory -Force -Path $anchorDir, $primaryDir, $tripleDir, $dualScoreboardDir, $structureDir, $structureTrackDir | Out-Null

& $Python (Join-Path $scriptDir "scan_device_anchors_v46.py") `
    $InputVideo $anchorDir --model $LocatorModel --fps $ProbeFps --start $Start --end $End
if ($LASTEXITCODE -ne 0) { throw "Candidate scan failed" }

$anchors = Join-Path $anchorDir "device_anchors.csv"
if ($FallbackLocatorModel -and (Test-Path -LiteralPath $FallbackLocatorModel)) {
    New-Item -ItemType Directory -Force -Path $fallbackAnchorDir, $mergedAnchorDir | Out-Null
    & $Python (Join-Path $scriptDir "scan_device_anchors_v46.py") `
        $InputVideo $fallbackAnchorDir --model $FallbackLocatorModel --fps $ProbeFps --start $Start --end $End
    if ($LASTEXITCODE -ne 0) { throw "Fallback locator candidate scan failed" }
    & $Python (Join-Path $scriptDir "merge_device_anchors_v51.py") `
        $anchors (Join-Path $fallbackAnchorDir "device_anchors.csv") $mergedAnchorDir
    if ($LASTEXITCODE -ne 0) { throw "Two-locator candidate fusion failed" }
    $anchors = Join-Path $mergedAnchorDir "device_anchors.csv"
}
if ($LegacyLocatorModel -and (Test-Path -LiteralPath $LegacyLocatorModel)) {
    New-Item -ItemType Directory -Force -Path $legacyAnchorDir, $allMergedAnchorDir | Out-Null
    & $Python (Join-Path $scriptDir "scan_device_anchors_v46.py") `
        $InputVideo $legacyAnchorDir --model $LegacyLocatorModel --fps $ProbeFps --start $Start --end $End
    if ($LASTEXITCODE -ne 0) { throw "Legacy venue locator candidate scan failed" }
    & $Python (Join-Path $scriptDir "merge_device_anchors_v51.py") `
        $anchors (Join-Path $legacyAnchorDir "device_anchors.csv") $allMergedAnchorDir
    if ($LASTEXITCODE -ne 0) { throw "Three-locator candidate fusion failed" }
    $anchors = Join-Path $allMergedAnchorDir "device_anchors.csv"
}
& $Python (Join-Path $scriptDir "track_device_anchors_v46.py") `
    $InputVideo $anchors $primaryDir --required-hits 3
if ($LASTEXITCODE -ne 0) { throw "Standard track probing failed" }

& $Python (Join-Path $scriptDir "track_triple_panel_layout_v47.py") `
    $InputVideo $anchors $tripleDir --start $Start --end $End --fps $ProbeFps
if ($LASTEXITCODE -ne 0) { throw "Three-screen layout probing failed" }

& $Python (Join-Path $scriptDir "infer_center_from_dual_scoreboards_v47.py") `
    $InputVideo $dualScoreboardDir --start $Start --end $End --fps $ProbeFps
if ($LASTEXITCODE -ne 0) { throw "Dual-score-screen geometry probing failed" }

& $Python (Join-Path $scriptDir "detect_scoreboard_structure_v47.py") `
    $InputVideo $structureDir --start $Start --end $End --fps $ProbeFps
if ($LASTEXITCODE -ne 0) { throw "Color-structure probing failed" }

& $Python (Join-Path $scriptDir "fuse_structure_track_v47.py") `
    $InputVideo (Join-Path $structureDir "structure_boxes.csv") $structureTrackDir `
    --fps $ProbeFps --anchors $anchors
if ($LASTEXITCODE -ne 0) { throw "Structure cross-check failed" }

$tripleSummary = Get-Content -LiteralPath (Join-Path $tripleDir "summary.json") -Raw | ConvertFrom-Json
$sampledFrames = [int]$tripleSummary.sampled_frames
$minimumValidFrames = [math]::Min(20, [math]::Max(4, [math]::Ceiling($sampledFrames * 0.10)))
& $Python (Join-Path $scriptDir "select_locator_route_v47.py") `
    $routePath --sampled-frames $sampledFrames `
    --primary-summary (Join-Path $primaryDir "summary.json") `
    --triple-summary (Join-Path $tripleDir "summary.json") `
    --dual-scoreboard-summary (Join-Path $dualScoreboardDir "summary.json") `
    --structure-summary (Join-Path $structureTrackDir "summary.json") `
    --minimum-valid-frames $minimumValidFrames
if ($LASTEXITCODE -ne 0) { throw "Automatic route selection failed" }

$routeData = Get-Content -LiteralPath $routePath -Raw | ConvertFrom-Json
if ($routeData.route -eq "dual_scoreboard") {
    New-Item -ItemType Directory -Force -Path $dualScoreboardFlowDir | Out-Null
    & $Python (Join-Path $scriptDir "track_device_anchors_v47.py") `
        $InputVideo (Join-Path $dualScoreboardDir "dual_scoreboard_track.csv") $dualScoreboardFlowDir `
        --start $Start --end $End --fps 10 --required-hits 2 --alpha 0.22 `
        --flow-max-gap 7 --hold-max-gap 0.6 --cut-threshold 0.72 --feature-factor 1.8
    if ($LASTEXITCODE -ne 0) { throw "Dual-score-screen optical-flow refinement failed" }
    $routeData | Add-Member -NotePropertyName selected_track -NotePropertyValue (Join-Path $dualScoreboardFlowDir "stable_device_track.csv") -Force
    $routeData | Add-Member -NotePropertyName selected_track_summary -NotePropertyValue (Join-Path $dualScoreboardFlowDir "summary.json") -Force
    $routeData | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $routePath -Encoding utf8
}
elseif ($routeData.route -in @("primary_v46", "review_required")) {
    New-Item -ItemType Directory -Force -Path $primaryFlowDir | Out-Null
    & $Python (Join-Path $scriptDir "track_device_anchors_v47.py") `
        $InputVideo $anchors $primaryFlowDir `
        --start $Start --end $End --fps 10 --required-hits 2 --alpha 0.25 `
        --flow-max-gap 7 --hold-max-gap 0.8 --cut-threshold 0.60 --feature-factor 1.6
    if ($LASTEXITCODE -ne 0) { throw "Central score-box optical-flow refinement failed" }

    $flowQualityPath = Join-Path $primaryFlowDir "quality_gate.json"
    & $Python (Join-Path $scriptDir "validate_primary_flow_v472.py") `
        $InputVideo (Join-Path $primaryFlowDir "stable_device_track.csv") $flowQualityPath `
        --start $Start --end $End
    if ($LASTEXITCODE -ne 0) { throw "Central score-box optical-flow quality check failed" }
    $flowQuality = Get-Content -LiteralPath $flowQualityPath -Raw | ConvertFrom-Json
    $routeData | Add-Member -NotePropertyName primary_flow_quality -NotePropertyValue $flowQuality -Force
    if ($flowQuality.qualified) {
        $routeData | Add-Member -NotePropertyName probe_route -NotePropertyValue $routeData.route -Force
        $routeData.route = "primary_flow_v472"
        $routeData.reason = "learned central-device anchors plus cut-aware optical flow passed the conservative quality gate"
        $routeData | Add-Member -NotePropertyName selected_track -NotePropertyValue (Join-Path $primaryFlowDir "stable_device_track.csv") -Force
        $routeData | Add-Member -NotePropertyName selected_track_summary -NotePropertyValue (Join-Path $primaryFlowDir "summary.json") -Force

        New-Item -ItemType Directory -Force -Path $primaryInterpolationDir | Out-Null
        & $Python (Join-Path $scriptDir "interpolate_device_track_gaps_v472.py") `
            $InputVideo (Join-Path $primaryFlowDir "stable_device_track.csv") $primaryInterpolationDir `
            --start $Start --end $End --fps 10 --maximum-gap 25 `
            --maximum-center-distance 0.04 --maximum-area-ratio 1.9
        if ($LASTEXITCODE -ne 0) { throw "Central score-box conservative gap filling failed" }

        $interpolationQualityPath = Join-Path $primaryInterpolationDir "quality_gate.json"
        & $Python (Join-Path $scriptDir "validate_primary_flow_v472.py") `
            $InputVideo (Join-Path $primaryInterpolationDir "stable_device_track_interpolated.csv") `
            $interpolationQualityPath --start $Start --end $End
        if ($LASTEXITCODE -ne 0) { throw "Central score-box filled-track quality check failed" }
        $interpolationQuality = Get-Content -LiteralPath $interpolationQualityPath -Raw | ConvertFrom-Json
        $interpolationSummary = Get-Content -LiteralPath (Join-Path $primaryInterpolationDir "summary.json") -Raw | ConvertFrom-Json
        $routeData | Add-Member -NotePropertyName primary_interpolation_quality -NotePropertyValue $interpolationQuality -Force
        $routeData | Add-Member -NotePropertyName primary_interpolation_summary -NotePropertyValue $interpolationSummary -Force
        if (
            $interpolationQuality.qualified -and
            [int]$interpolationSummary.interpolated_rows -gt 0 -and
            [double]$interpolationSummary.coverage -gt [double]$flowQuality.metrics.coverage
        ) {
            $routeData.route = "primary_flow_v473"
            $routeData.reason = "qualified cut-aware optical flow plus conservative same-shot gap interpolation"
            $routeData.selected_track = Join-Path $primaryInterpolationDir "stable_device_track_interpolated.csv"
            $routeData.selected_track_summary = Join-Path $primaryInterpolationDir "summary.json"
        }
    }
    $routeData | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $routePath -Encoding utf8
}

Write-Host "Automatic route probing completed: $routePath"
