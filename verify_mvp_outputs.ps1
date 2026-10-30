param(
    [Parameter(Mandatory = $true)]
    [string]$Manifest,
    [string]$Python = "",
    [string]$OutputJson = "",
    [switch]$IncludeInputs
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$verifier = Join-Path $projectRoot "code\mvp_v1\verify_run_manifest.py"
$manifestPath = [System.IO.Path]::GetFullPath($Manifest)

if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Run manifest not found: $manifestPath"
}
if (-not (Test-Path -LiteralPath $verifier -PathType Leaf)) {
    throw "Output verifier not found: $verifier"
}
if ([string]::IsNullOrWhiteSpace($Python)) {
    $preferredPython = Join-Path $env:USERPROFILE ".conda\envs\torch\python.exe"
    if (Test-Path -LiteralPath $preferredPython -PathType Leaf) {
        $Python = $preferredPython
    }
    else {
        $command = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $command) {
            throw "Python was not found. Use -Python to specify python.exe."
        }
        $Python = $command.Source
    }
}
if ([string]::IsNullOrWhiteSpace($OutputJson)) {
    $OutputJson = Join-Path (Split-Path -Parent $manifestPath) "verification.json"
}
else {
    $OutputJson = [System.IO.Path]::GetFullPath($OutputJson)
}

$arguments = @($verifier, "--manifest", $manifestPath, "--output-json", $OutputJson)
if ($IncludeInputs) {
    $arguments += "--include-inputs"
}
& $Python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Output verification failed. Details were written to: $OutputJson"
}
Write-Host "Output verification passed: $OutputJson" -ForegroundColor Green
