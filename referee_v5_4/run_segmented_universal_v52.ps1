param(
    [Parameter(Mandatory = $true)]
    [string]$InputVideo,
    [Parameter(Mandatory = $true)]
    [string]$OutputDir,
    [double]$Start = 0,
    [double]$End = -1,
    [double]$SegmentSeconds = 60,
    [double]$OverlapSeconds = 2,
    [string]$Python = "python",
    [switch]$NoPreview
)

$ErrorActionPreference = "Stop"
if ($SegmentSeconds -le 5) { throw "SegmentSeconds must be greater than 5 seconds" }
if ($OverlapSeconds -lt 0 -or $OverlapSeconds -ge $SegmentSeconds / 2) {
    throw "OverlapSeconds must be non-negative and less than half the segment length"
}
$packageRoot = $PSScriptRoot
$codeDir = Join-Path $packageRoot "code"
$videoInfoPath = Join-Path $OutputDir "video_info.json"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
& $Python (Join-Path $codeDir "probe_video_info_v50.py") $InputVideo $videoInfoPath
if ($LASTEXITCODE -ne 0) { throw "Unable to read video metadata" }
$videoInfo = Get-Content -LiteralPath $videoInfoPath -Raw | ConvertFrom-Json
if ($End -lt 0) { $End = [double]$videoInfo.duration_sec }
if ($End -le $Start) { throw "End must be greater than Start" }

$segmentsDir = Join-Path $OutputDir "segments"
New-Item -ItemType Directory -Force -Path $segmentsDir | Out-Null
$cursor = $Start
$index = 0
while ($cursor -lt $End - 0.001) {
    $segmentStart = [math]::Max($Start, $cursor - $(if ($index -gt 0) { $OverlapSeconds } else { 0 }))
    $segmentEnd = [math]::Min($End, $cursor + $SegmentSeconds + $OverlapSeconds)
    $segmentDir = Join-Path $segmentsDir ("segment_{0:D4}_{1:F1}_{2:F1}" -f $index, $segmentStart, $segmentEnd)
    $runnerArgs = @{
        InputVideo = $InputVideo
        OutputDir = $segmentDir
        Start = $segmentStart
        End = $segmentEnd
        Python = $Python
    }
    if ($NoPreview) { $runnerArgs["NoPreview"] = $true }
    & (Join-Path $packageRoot "run_universal_referee_v52.ps1") @runnerArgs
    if ($LASTEXITCODE -ne 0) { throw "Segment $index failed" }
    $cursor += $SegmentSeconds
    $index += 1
}

$finalDir = Join-Path $OutputDir "final"
& $Python (Join-Path $codeDir "merge_universal_segments_v51.py") $segmentsDir $finalDir
if ($LASTEXITCODE -ne 0) { throw "Unable to merge segment results" }
if (-not $NoPreview) {
    & $Python (Join-Path $codeDir "render_segmented_demo_v51.py") `
        $InputVideo (Join-Path $finalDir "route_timeline.csv") `
        (Join-Path $finalDir "final_events.csv") `
        (Join-Path $finalDir "review_events.csv") `
        (Join-Path $OutputDir "final_route_demo.mp4") `
        --start $Start --end $End
    if ($LASTEXITCODE -ne 0) { throw "Unable to render the final segmented demo" }
}
Write-Host "V5.4 segmented physical score-box fusion completed: $OutputDir"
