param(
    [Parameter(Mandatory = $true)]
    [string]$InputVideo,
    [Parameter(Mandatory = $true)]
    [string]$OutputDir,
    [double]$Start = 0,
    [double]$End = -1,
    [double]$ProbeFps = 2,
    [double]$TrackFps = 5,
    [double]$AnalysisFps = 15,
    [string]$Python = "python",
    [string]$LocatorModel = "",
    [string]$FallbackLocatorModel = "",
    [string]$LegacyLocatorModel = "",
    [string]$LampModel = "",
    [string]$FallbackLampModel = "",
    [switch]$NoPreview
)

$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "run_universal_referee_v52.ps1") @PSBoundParameters
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
