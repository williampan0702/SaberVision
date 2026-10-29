# SaberVision 裁判器识别 V5.4 实体裁判器融合版

## 这版完成了什么

V5.4 的核心变化是：不再只依赖宽松的大框或两侧比分屏推算中央位置，而是新增了一个专门寻找实体裁判器的紧框模型，并与 V5.3 的两个旧定位模型融合。定位到可信的中央实体裁判器后，系统同时运行学习型亮灯识别和新的时序红/绿变化检测，再把碎片合并成一条 `LEFT`、`RIGHT` 或 `DOUBLE` 事件。

整场视频仍按时间分段，每段都会重新判断场景布局，因此可以处理镜头移动、切镜和同一视频中场馆布局变化。定位不稳定时，系统会输出到复核表，而不会硬猜成自动结果。

“通用版”表示它具备多布局路由、三模型融合和安全拒绝机制；不表示在从未见过的场馆能够达到 100% 召回率。

## 推荐的一键入口

整场或包含多场比赛的视频：

```powershell
& ".\run_segmented_universal_v54.ps1" `
  -InputVideo "D:\视频\整场比赛.mp4" `
  -OutputDir "D:\输出\整场比赛_v54" `
  -Python (Join-Path $env:USERPROFILE ".conda\envs\torch\python.exe") `
  -NoPreview
```

已经切好的单剑片段：

```powershell
& ".\run_universal_referee_v54.ps1" `
  -InputVideo "D:\视频\一剑片段.mp4" `
  -OutputDir "D:\输出\一剑片段_v54" `
  -Python (Join-Path $env:USERPROFILE ".conda\envs\torch\python.exe")
```

这两个就是用户入口，会自动调用内部所需代码；平时不需要逐个运行 `code` 目录里的脚本。

## 当前处理流程

1. 整场入口默认每 60 秒重新探测一次场景，并保留 2 秒重叠。
2. 新实体裁判器模型、V5.3 主模型和旧版补漏模型共同产生候选框。
3. 根据候选密度、位置、大小及多判罚区结构选择路线；可信的中央实体裁判器优先。
4. 对稳定框进行轨迹插值和防抖，不把每帧独立检测框直接当最终框。
5. 在框内同时运行模型亮灯识别与相对历史基线的红/绿时序变化检测。
6. 合并同一剑的短碎片和左右证据，生成 `LEFT`、`RIGHT` 或 `DOUBLE`。
7. 可靠事件写入 `final_events.csv`；低覆盖、强切镜或证据不足的候选写入 `review_events.csv`。

## V5.4 实测结果

- 新实体裁判器定位模型训练 25 个 epoch：验证集 Precision 0.954、Recall 0.908、mAP50 0.965、mAP50-95 0.883。
- Tunis 固定紧框留出集共 74 个有效帧：新模型 74/74 达到 IoU≥0.5，平均 IoU 0.975；旧主模型为 0/74、平均 IoU 0.005。
- 未参与该模型训练区间的单剑集成测试：第 1、4、13、22 剑均自动形成一条最终事件；第 7 剑因强切镜和遮挡进入复核，没有自动乱报。
- 第 1 剑演示中，中央框保持紧贴实体裁判器，左灯约在 62.80–68.00 秒识别，和人工记录 63.057–68.144 秒接近。
- Seoul/Tbilisi 的 60 帧抽样只做了“模型是否出框”的补充检查，不是准确率。新模型分别为 16/60、3/60，旧模型分别为 20/60、6/60，因此最终保留三模型融合，未用新模型完全替换旧模型。

Tunis 训练区间和固定留出集虽然时间段不同，但来自同一场比赛/场馆；74/74 不能当成跨场馆 100% 准确率。详细记录见 `validation/V54_LOCATOR_AND_INTEGRATED_SUMMARY.json`。

## 输出文件

- `final/final_events.csv`：通过定位和事件安全门的自动结果。
- `final/review_events.csv`：需要人工快速确认的低覆盖或弱证据候选。
- `final/route_timeline.csv`：整场分段时各时间窗选择的路线。
- `final/summary.json`：路线、自动事件和复核事件统计。
- `segments/segment_*`：各时间窗的定位轨迹、候选和质量报告。
- 未使用 `-NoPreview` 时会生成 `final_boxed_demo.mp4` 或 `final_route_demo.mp4`。

下游自动化只能读取 `final_events.csv`；不要把 `review_events.csv` 当作真值。

## 演示和审计

- `demos/tunis_exchange01_v54_physical.mp4`：V5.4 实体裁判器紧框、当前灯和最终锁存结果。
- `demos/mixed_venue_route_demo.mp4`：不同场馆连续切换时的路由示例。
- `demos/seoul_v52_boxed_latched.mp4`：三区域场馆示例。
- `demos/riyadh_v52_boxed_latched.mp4`：双比分屏场馆示例。
- `validation/V54_LOCATOR_AND_INTEGRATED_SUMMARY.json`：V5.4 定位和独立单剑集成结果。
- `validation/V53_EXPANDED_VALIDATION_SUMMARY.json`：上一版 21 例扩展固定测试，作为历史回归记录保留。

## 当前边界

- 裁判器很小、被运动员完全遮挡、画面严重压缩或在亮灯前后立刻切镜时，仍可能进入复核或漏灯。
- 时序颜色补偿依赖同一镜头中有一小段未亮基线；整段从第一帧就持续亮灯的片段更难判断。
- 三模型融合提升场馆覆盖面，但 Tbilisi 一类超远裁判器仍是弱项。
- 覆盖率、出框率和候选数量都不等于最终准确率；要获得正式跨场馆准确率，仍需对更多测试场馆逐剑标注真实灯型和时间。
