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
& (Join-Path $PSScriptRoot "run_segmented_universal_v52.ps1") @PSBoundParameters
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
