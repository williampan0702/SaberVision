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
    throw "找不到 Python。请用 -Python 指定要检查的 python.exe。"
}
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $packageRoot "environment_check.json"
}

& $Python $checker --package-root $packageRoot --output $Output
$exitCode = $LASTEXITCODE
Write-Host ""
Write-Host "体检报告：$Output"
if ($exitCode -eq 0) {
    Write-Host "SaberVision 运行环境完整。" -ForegroundColor Green
} else {
    Write-Host "运行环境不完整；请按上方 JSON 中的缺失项处理。" -ForegroundColor Yellow
}
exit $exitCode
