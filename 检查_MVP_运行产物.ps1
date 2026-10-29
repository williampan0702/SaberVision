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
    throw "找不到运行清单：$manifestPath"
}
if (-not (Test-Path -LiteralPath $verifier -PathType Leaf)) {
    throw "找不到产物校验器：$verifier"
}
if ([string]::IsNullOrWhiteSpace($Python)) {
    $preferredPython = Join-Path $env:USERPROFILE ".conda\envs\torch\python.exe"
    if (Test-Path -LiteralPath $preferredPython -PathType Leaf) {
        $Python = $preferredPython
    }
    else {
        $command = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $command) {
            throw "找不到 Python。请用 -Python 指定 python.exe。"
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
    throw "产物校验失败；详情已写入：$OutputJson"
}
Write-Host "产物校验通过：$OutputJson" -ForegroundColor Green
