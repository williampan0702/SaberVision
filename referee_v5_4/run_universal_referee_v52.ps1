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
$packageRoot = $PSScriptRoot
$codeDir = Join-Path $packageRoot "code"
$modelDir = Join-Path $packageRoot "models"
if (-not (Test-Path -LiteralPath $codeDir)) {
    # Development-tree fallback: this script is next to its Python dependencies.
    $codeDir = $PSScriptRoot
    $modelDir = Join-Path (Split-Path $PSScriptRoot -Parent) "models"
}
if (-not $LocatorModel) { $LocatorModel = Join-Path $modelDir "score_device_v54_physical.pt" }
if (-not $FallbackLocatorModel) { $FallbackLocatorModel = Join-Path $modelDir "score_device_v46_best.pt" }
if (-not $LegacyLocatorModel) { $LegacyLocatorModel = Join-Path $modelDir "score_device_v46b_legacy.pt" }
if (-not $LampModel) { $LampModel = Join-Path $modelDir "physical_lamp_v474_epoch0.pt" }
if (-not $FallbackLampModel) { $FallbackLampModel = Join-Path $modelDir "physical_lamp_fallback_v45b.pt" }
foreach ($required in @($InputVideo, $LocatorModel, $FallbackLocatorModel, $LegacyLocatorModel, $LampModel, $FallbackLampModel)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "找不到必需文件：$required" }
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$videoInfoPath = Join-Path $OutputDir "video_info.json"
& $Python (Join-Path $codeDir "probe_video_info_v50.py") $InputVideo $videoInfoPath
if ($LASTEXITCODE -ne 0) { throw "读取视频信息失败" }
$videoInfo = Get-Content -LiteralPath $videoInfoPath -Raw | ConvertFrom-Json
if ($End -lt 0) { $End = [double]$videoInfo.duration_sec }
if ($End -le $Start) { throw "End 必须大于 Start" }

$probeDir = Join-Path $OutputDir "01_locator_probe"
& (Join-Path $codeDir "run_v47_auto_probe.ps1") `
    -InputVideo $InputVideo -OutputDir $probeDir -Start $Start -End $End `
    -ProbeFps $ProbeFps -Python $Python -LocatorModel $LocatorModel `
    -FallbackLocatorModel $FallbackLocatorModel -LegacyLocatorModel $LegacyLocatorModel
if ($LASTEXITCODE -ne 0) { throw "自动定位路线探测失败" }

$routePath = Join-Path $probeDir "locator_route.json"
$routeData = Get-Content -LiteralPath $routePath -Raw | ConvertFrom-Json
$route = [string]$routeData.route
$candidateEvents = $null
$selectedTrack = $null

if ($route -eq "triple_panel") {
    $strictTrack = Join-Path $probeDir "03_triple_panel\triple_panel_track.csv"
    $anchors = Join-Path $probeDir "01_anchors\device_anchors.csv"
    if ($ProbeFps -lt $TrackFps) {
        # Sparse 2 fps is enough to choose the route, but three synchronized
        # boxes need a denser 5 fps trajectory for 15 fps colour analysis.
        $denseAnchorDir = Join-Path $OutputDir "02_triple_dense_anchors"
        $denseTrackDir = Join-Path $OutputDir "02_triple_dense_track"
        & $Python (Join-Path $codeDir "scan_device_anchors_v46.py") `
            $InputVideo $denseAnchorDir --model $LocatorModel --fps $TrackFps --start $Start --end $End
        if ($LASTEXITCODE -ne 0) { throw "三区域密集候选扫描失败" }
        $anchors = Join-Path $denseAnchorDir "device_anchors.csv"
        & $Python (Join-Path $codeDir "track_triple_panel_layout_v47.py") `
            $InputVideo $anchors $denseTrackDir --start $Start --end $End --fps $TrackFps --no-preview
        if ($LASTEXITCODE -ne 0) { throw "三区域密集轨迹失败" }
        $strictTrack = Join-Path $denseTrackDir "triple_panel_track.csv"
    }
    $interpolatedDir = Join-Path $OutputDir "02_triple_interpolated"
    & $Python (Join-Path $codeDir "interpolate_three_region_track_v48.py") `
        $strictTrack $interpolatedDir --fps $AnalysisFps --maximum-gap 1.2 `
        --width ([int]$videoInfo.width) --height ([int]$videoInfo.height)
    if ($LASTEXITCODE -ne 0) { throw "三区域保守插值失败" }

    $trackDir = Join-Path $OutputDir "03_triple_track_v50"
    & $Python (Join-Path $codeDir "augment_three_region_track_v49.py") `
        (Join-Path $interpolatedDir "triple_panel_track_interpolated.csv") $anchors $trackDir `
        --width ([int]$videoInfo.width) --height ([int]$videoInfo.height)
    if ($LASTEXITCODE -ne 0) { throw "三区域当前帧安全恢复失败" }
    $selectedTrack = Join-Path $trackDir "triple_panel_track_v49.csv"

    $physicalDir = Join-Path $OutputDir "04_physical_lamp"
    $previewFlag = @()
    if ($NoPreview) { $previewFlag = @("--no-preview") }
    & $Python (Join-Path $codeDir "scan_device_gated_lamps_v474.py") `
        $InputVideo $selectedTrack $physicalDir --lamp-model $LampModel `
        --start $Start --end $End --analysis-fps $AnalysisFps @previewFlag
    if ($LASTEXITCODE -ne 0) { throw "实体裁判器灯扫描失败" }

    $rawDir = Join-Path $OutputDir "05_three_region_raw"
    & $Python (Join-Path $codeDir "audit_three_region_signals_v49.py") `
        $InputVideo $selectedTrack $rawDir --start $Start --end $End `
        --analysis-fps $AnalysisFps @previewFlag
    if ($LASTEXITCODE -ne 0) { throw "三区域颜色审计失败" }

    $regionalDir = Join-Path $OutputDir "06_three_region_rethreshold"
    & $Python (Join-Path $codeDir "rethreshold_three_region_frames_v49.py") `
        (Join-Path $rawDir "three_region_frames.csv") $regionalDir `
        --side-delta 0.055 --center-delta 0.035 `
        --side-margin 0.015 --center-left-margin 0.009 --center-right-margin 0.015
    if ($LASTEXITCODE -ne 0) { throw "三区域自适应基线重判失败" }

    $fusedDir = Join-Path $OutputDir "07_fused_candidates"
    & $Python (Join-Path $codeDir "fuse_three_region_events_v49.py") `
        (Join-Path $physicalDir "device_gated_lamp_events.csv") `
        (Join-Path $regionalDir "three_region_events.csv") $fusedDir
    if ($LASTEXITCODE -ne 0) { throw "三区域事件融合失败" }
    $candidateEvents = Join-Path $fusedDir "final_events.csv"
}
elseif ($route -eq "dual_scoreboard") {
    $selectedTrack = [string]$routeData.selected_track
    if (-not $selectedTrack) { $selectedTrack = Join-Path $probeDir "04_dual_scoreboard\dual_scoreboard_track.csv" }
    # Keep the optical-flow track for the stable user-facing box, but measure
    # lamp pixels from the current-frame scoreboard geometry.  Flow smoothing
    # can drift upward while a fencer crosses the centre and clip the external
    # lamp blocks even though the preview box still looks plausible.
    $lampTrack = Join-Path $probeDir "04_dual_scoreboard\dual_scoreboard_track.csv"
    $lampDir = Join-Path $OutputDir "04_dual_scoreboard_lamp"
    $flowLampDir = Join-Path $lampDir "flow_track"
    $geometryLampDir = Join-Path $lampDir "current_geometry"
    & $Python (Join-Path $codeDir "scan_dual_scoreboard_lamps_v47.py") `
        $InputVideo $selectedTrack $flowLampDir --start $Start --end $End --analysis-fps $AnalysisFps
    if ($LASTEXITCODE -ne 0) { throw "双比分屏光流框亮灯扫描失败" }
    & $Python (Join-Path $codeDir "scan_dual_scoreboard_lamps_v47.py") `
        $InputVideo $lampTrack $geometryLampDir --start $Start --end $End --analysis-fps $AnalysisFps
    if ($LASTEXITCODE -ne 0) { throw "双比分屏当前几何框亮灯扫描失败" }
    & $Python (Join-Path $codeDir "merge_dual_track_events_v53.py") `
        (Join-Path $flowLampDir "dual_scoreboard_lamp_events.csv") `
        (Join-Path $geometryLampDir "dual_scoreboard_lamp_events.csv") $lampDir
    if ($LASTEXITCODE -ne 0) { throw "双比分屏双轨事件融合失败" }
    $candidateEvents = Join-Path $lampDir "dual_scoreboard_lamp_events.csv"
}
elseif ($route -eq "structure_assisted") {
    # The structure box includes scoreboard graphics in some broadcasts.  Use
    # the conservative pre-recall scanner here; the high-recall v4.7.4 lamp
    # model is reserved for learned/flow-confirmed physical-device tracks.
    $selectedTrack = Join-Path $probeDir "05_structure_track\fused_device_track.csv"
    $lampDir = Join-Path $OutputDir "04_structure_gated_lamp"
    & $Python (Join-Path $codeDir "scan_device_gated_lamps_v47.py") `
        $InputVideo $selectedTrack $lampDir --lamp-model $LampModel `
        --start $Start --end $End --analysis-fps $AnalysisFps
    if ($LASTEXITCODE -ne 0) { throw "结构辅助路线保守亮灯扫描失败" }
    $candidateEvents = Join-Path $lampDir "device_gated_lamp_events.csv"
}
elseif ($route -in @("primary_v46", "primary_flow_v472", "primary_flow_v473")) {
    $selectedTrack = [string]$routeData.selected_track
    if (-not $selectedTrack -and $route -eq "structure_assisted") {
        $selectedTrack = Join-Path $probeDir "05_structure_track\fused_device_track.csv"
    }
    if (-not $selectedTrack) {
        $selectedTrack = Join-Path $probeDir "02_primary_track\stable_device_track.csv"
    }
    $lampDir = Join-Path $OutputDir "04_device_gated_lamp"
    $previewFlag = @()
    if ($NoPreview) { $previewFlag = @("--no-preview") }
    & $Python (Join-Path $codeDir "scan_device_gated_lamps_v474.py") `
        $InputVideo $selectedTrack $lampDir --lamp-model $LampModel `
        --start $Start --end $End --analysis-fps $AnalysisFps @previewFlag
    if ($LASTEXITCODE -ne 0) { throw "中央裁判器亮灯扫描失败" }
    $temporalLampDir = Join-Path $OutputDir "04_physical_temporal_lamp"
    & $Python (Join-Path $codeDir "scan_physical_lamp_temporal_v54.py") `
        $InputVideo $selectedTrack $temporalLampDir `
        --start $Start --end $End --analysis-fps $AnalysisFps
    if ($LASTEXITCODE -ne 0) { throw "中央裁判器时序颜色补判失败" }
    $fusedLampDir = Join-Path $OutputDir "04_device_lamp_fused_v54"
    & $Python (Join-Path $codeDir "merge_primary_fallback_events_v47.py") `
        (Join-Path $lampDir "device_gated_lamp_events.csv") `
        (Join-Path $temporalLampDir "physical_temporal_lamp_events.csv") $fusedLampDir
    if ($LASTEXITCODE -ne 0) { throw "中央裁判器双方法事件融合失败" }
    $candidateEvents = Join-Path $fusedLampDir "fused_events.csv"
}
else {
    # No reliable device track: mine only strong full-frame colour pairs and
    # let the final route gate force every candidate into review_events.csv.
    # This recovers possible Cairo-style edge lamps without reintroducing the
    # Hong Kong green-vest/background false positives into automatic output.
    $fallbackDir = Join-Path $OutputDir "04_fullframe_review_fallback"
    $fallbackPreviewFlag = @()
    if ($NoPreview) { $fallbackPreviewFlag = @("--no-preview") }
    & $Python (Join-Path $codeDir "scan_edge_lamp_fallback_v47.py") `
        $InputVideo $fallbackDir --model $FallbackLampModel `
        --start $Start --end $End --analysis-fps $AnalysisFps `
        --weak-threshold 0.005 --strong-threshold 0.05 @fallbackPreviewFlag
    if ($LASTEXITCODE -ne 0) { throw "低覆盖全画面复核补漏失败" }
    $candidateEvents = Join-Path $fallbackDir "edge_lamp_fallback_events.csv"
}

$finalDir = Join-Path $OutputDir "final"
& $Python (Join-Path $codeDir "finalize_universal_events_v50.py") `
    $routePath $candidateEvents $finalDir
if ($LASTEXITCODE -ne 0) { throw "最终事件安全门失败" }

if (-not $NoPreview -and $selectedTrack -and (Test-Path -LiteralPath $selectedTrack)) {
    & $Python (Join-Path $codeDir "render_final_boxed_demo_v52.py") `
        $InputVideo $selectedTrack `
        (Join-Path $finalDir "final_events.csv") `
        (Join-Path $finalDir "review_events.csv") `
        (Join-Path $OutputDir "final_boxed_demo.mp4") `
        --start $Start --end $End --fps $AnalysisFps --route $route
    if ($LASTEXITCODE -ne 0) { throw "最终带框演示生成失败" }
}

$runSummary = [ordered]@{
    version = "5.4-physical-fusion"
    input = (Resolve-Path -LiteralPath $InputVideo).Path
    start_sec = $Start
    end_sec = $End
    selected_route = $route
    route_file = $routePath
    selected_track = $selectedTrack
    final_events = (Join-Path $finalDir "final_events.csv")
    review_events = (Join-Path $finalDir "review_events.csv")
}
$runSummary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $OutputDir "run_summary.json") -Encoding utf8
Write-Host "V5.4 实体裁判器融合版完成：$OutputDir；路线=$route"
