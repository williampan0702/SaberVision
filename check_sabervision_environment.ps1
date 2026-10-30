param(
    [string]$Python = "",
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = "1"
$packageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$checker = Join-Path $packageRoot "code\mvp_v1\check_environment.py"

if ([string]::IsNullOrWhiteSpace($Python)) {
    $preferredPython = Join-Path $env:USERPROFILE ".conda\envs\torch\python.exe"
    if (Test-Path -LiteralPath $preferredPython -PathType Leaf) {
        $Python = $preferredPython
    } else {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($null -ne $pythonCommand) {
            $Python = $pythonCommand.Source
        }
    }
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python was not found. Use -Python to specify the python.exe to inspect."
}
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $packageRoot "environment_check.json"
}

& $Python $checker --package-root $packageRoot --output $Output
$exitCode = $LASTEXITCODE
Write-Host ""
Write-Host "Environment report: $Output"
if ($exitCode -eq 0) {
    Write-Host "The SaberVision runtime environment is complete." -ForegroundColor Green
} else {
    Write-Host "The runtime environment is incomplete. Resolve the missing items listed in the JSON report." -ForegroundColor Yellow
}
exit $exitCode
