param(
    [Parameter(Mandatory = $true)]
    [string]$InputVideo,

    [string]$OutputDir = "",

    [string]$Python = "",

    [switch]$Reuse
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = "1"
$packageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$runner = Join-Path $packageRoot "code\mvp_v1\sabervision_mvp.py"

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

if (-not (Test-Path -LiteralPath $InputVideo -PathType Leaf)) {
    throw "找不到输入视频：$InputVideo"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "找不到 Python。请用 -Python 指定已安装 PyTorch 的 python.exe。"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($InputVideo)
    $OutputDir = Join-Path $packageRoot ("results\" + $stem)
}

$arguments = @(
    $runner,
    "--video", $InputVideo,
    "--output-dir", $OutputDir,
    "--python", $Python,
    "--fencer-model", (Join-Path $packageRoot "models\fencer_v4.pt"),
    "--pose-model", (Join-Path $packageRoot "models\fencing_pose_v1.pt"),
    "--fallback-person-model", (Join-Path $packageRoot "models\yolo11n-pose.pt"),
    "--referee-runner", (Join-Path $packageRoot "referee_v5_4\run_universal_referee_v54.ps1")
)
if ($Reuse) {
    $arguments += "--reuse"
}

& $Python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "MVP 运行失败。请查看 $OutputDir\run_manifest.json 和 logs 文件夹。"
}

Write-Host ""
Write-Host "SaberVision MVP 已完成" -ForegroundColor Green
Write-Host "演示视频：$OutputDir\SaberVision_MVP_demo.mp4"
Write-Host "判罚结果：$OutputDir\decision\verdict.json"
$reviewSummary = Join-Path $OutputDir "decision\review_packet_summary.json"
if (Test-Path -LiteralPath $reviewSummary -PathType Leaf) {
    $review = Get-Content -LiteralPath $reviewSummary -Raw | ConvertFrom-Json
    if ([int]$review.review_events -gt 0) {
        Write-Host "待复核事件：$($review.review_events) 个" -ForegroundColor Yellow
        Write-Host "复核动作图：$OutputDir\decision\review_packets"
        Write-Host "复核浏览页：$($review.review_index)"
    } else {
        Write-Host "待复核事件：0 个"
    }
}
