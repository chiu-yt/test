# 硕士研究课题开题与执行报告

**课题名称：** 基于 LiDAR 几何物理验证的多模态 3D 目标检测测试时自适应研究  
**(LiDAR Geometry Verification for Multimodal 3D Object Detection Test-Time Adaptation)**

**研究周期：** 3-6 个月  
**目标受众：** 硕士学位论文 / SCI 二区期刊为主，兼顾向 IROS/ICRA/T-IV 级别工作靠拢  
**核心约束：** 严格聚焦 source-free / off-the-shelf BEVFusion 的 Test-Time Adaptation，不重新预训练检测器，不引入需要 source training 的新网络结构。

---

## 一、研究背景与最新问题修正

本项目最初围绕“跨模态冲突感知”展开：在 fog 条件下，BEVFusion dual-fusion 结果明显弱于单模态结果，因此推测 LiDAR 与 Camera 的几何冲突导致 fused pseudo labels 不可靠。

full-val source-only benchmark 已得到以下关键事实：

- `clean_fullval`: `NDS=0.6736`, `mAP=0.6421`
- `fog_lidar_s3_fullval`: `NDS=0.6647`, `mAP=0.6286`
- `fog_image_s3_fullval`: `NDS=0.6655`, `mAP=0.6278`
- `fog_image_lidar_s3_fullval`: `NDS=0.6563`, `mAP=0.6134`
- `fog_lidar_s5_fullval`: `NDS=0.6320`, `mAP=0.5775`
- `fog_image_s5_fullval`: `NDS=0.6516`, `mAP=0.6025`
- `fog_image_lidar_s5_fullval`: `NDS=0.6017`, `mAP=0.5213`

随后完成的 `fog_s3_conflict_probe` 进一步修正了原假设：

- `B_L` / `B_C` 中心距离与 fused GT center error 的整体相关性仅为 `-0.034`；
- `pedestrian` 为 `-0.014`；
- `traffic_cone` 为 `-0.010`；
- 在 `score >= 0.3` 的高置信 pseudo-label 区间，`pedestrian` 和 `traffic_cone` 的 Camera-only 独立支持非常稀疏。

随后的 depth entropy probes 进一步表明：Camera/LSS 分支不仅不能稳定补充强 fog 下的 pseudo labels，甚至其 depth entropy 也不足以稳定地区分 TP/FP。frame-level clean vs `fog s3` entropy 几乎不变；object-level 上 `traffic_cone` 只有弱的 FP 高熵趋势，而 `pedestrian` 反而呈现 TP entropy 更高的反例。

因此，当前更准确的科学问题不是“如何度量 LiDAR 与 Camera 的对称几何冲突”，也不是“如何直接用 Camera uncertainty 过滤伪标签”，而是：

> 在强 fog 等恶劣天气下，多模态检测发生非对称退化：source-free BEVFusion TTA 如何在不依赖 fused score、Camera entropy 或对称 LiDAR-Camera agreement 的情况下，直接用 LiDAR 物理几何验证并复用低置信 fused pseudo labels？

---

## 二、核心创新结构

### 大创新：LiDAR 几何物理验证的非对称伪标签可靠性建模

不再要求 `B_L` 与 `B_C` 必须达成几何一致，也不再把 LSS entropy 作为主诊断信号，而是将 LiDAR 点云的物理几何签名作为 pseudo-label 的可靠性锚点。

核心物理假设：

- 真实目标产生 **surface reflection**，点云应在 box 内形成具有表面支撑和垂直延展的几何分布；
- 雾气误检来自 **volumetric scattering**，点云更可能是稀疏、悬浮、体内散布的离群簇；
- 因此 `point_count`、`point_density`、`z_span`、`z_var` 比 Camera uncertainty 更适合作为强 fog 下的伪标签真伪验证信号。

当前主线的可靠性形式改为 **Physical-over-Semantic recall mining**：在严重模态退化下，不再把 fused classification score 视为低分候选的唯一准入标准，而是用 LiDAR 物理几何对被低语义分数抑制的候选进行召回挖掘。

为了避免方法被写成脆弱的 hard-threshold trick，第一版论文表述应采用连续的 **Geometric Confidence Calibration**：

```text
G_density(i) = sigmoid(k * (rho_i - rho_th))
G_z(i)       = sigmoid(m * (z_span_i - z_th))
R_geo(i)     = G_density(i) * G_z(i)              # optional: * G_count(i)
W_i          = max(Score_i, R_geo(i))
```

- `Score_i`: fused detector score；
- `N_pts`: pseudo box 内 LiDAR 点数；
- `rho_pts = N_pts / (length * width * height)`: 点密度；
- `z_span` / `z_var`: box 内点云垂直跨度与方差；
- `R_geo(i)`: 基于 sigmoid 映射的软几何可靠性分数；
- `H_depth(i)`: 仅作为后续 combined ablation 的辅助项，不作为当前主线。

落地方式也随最新诊断调整：geometry verifier 不再依赖 `memory_ensemble_utils.save_pseudo_label_batch()` 之后的 ignored boxes，而应直接读取 raw `pred_dicts` 中 target-class low-score window，先做低分候选去重（candidate dedup / TTA-NMS），再做 LiDAR geometry verification，最后以保守的 reweight 为主、promote 为辅地写入伪标签。

写作边界：可以把该机制称为“利用跨模态物理先验缓解 self-training confirmation bias”，但在 `fog s3` promoted-count、TP/FP proxy 与短 TTA 增益尚未闭环前，不应宣称低分 raw proposals 中“隐藏大量 TP”已经被证明。

### 小创新 1：Selective Adaptation Boundary

继续保留并系统化当前已验证的 F1 策略：冻结 `image_backbone + neck`，避免 fog 下脏视觉梯度破坏图像先验。

### 小创新 2：Class-aware Geometry-aware Denoising

继续保留 `F1 + D1.3` 稳定基线：只对 `pedestrian=0.24` 与 `traffic_cone=0.34` 做 targeted denoising。下一步不是继续调 entropy 或 geometry 门限，而是在 raw proposal 阶段对这两个高噪类别建立 class-aware reliability window：`pedestrian` 先验证 direct geometry promote，`traffic_cone` 先诊断低于 `NEG_THRESH` 的极低分候选是否有可用几何证据。

该小创新在论文中应从 “Class-aware depth-aware denoising” 升级为 “Class-aware raw reliability window”：不同高噪类别使用不同的 raw score 探测窗口与几何校准强度，避免把全类别统一低阈值写成简单调参。

---

## 三、当前不采用为主线的方案

以下方向具有论文吸引力，但不适合作为当前 3-6 个月 source-free TTA 主线：

1. **GaussianLSS / 学习深度方差**  
   需要修改 view transform 或训练策略，不是现成 BEVFusion 的免参 TTA。

2. **Evidential Learning / EDL Head**  
   需要改检测头输出与训练损失，通常依赖 source-side supervised training。

3. **Dual-Expert / PanDA-style Refinement**  
   需要额外专家、gating 或语义修正训练，工程周期和 proof burden 都过高。

这些方法可放在 Related Work 或 Future Work，不应写成当前主方法。

### 3.1 文献定位与表述边界

外部检索支持下面这个谨慎定位：

- LSS / BEVDepth / BEVFusion 系列本身已经计算离散深度 posterior，并用其完成 image feature lifting；
- 单目 3D 检测中存在 depth uncertainty weighting 的先例；
- TTA / 半监督场景中，entropy-based pseudo-label filtering 是常见可靠性思想；
- 但目前没有直接证据表明 BEVFusion 标准做法已经用“depth-bin entropy”过滤 3D pseudo labels。

因此论文中应写成：

> We derive a zero-cost uncertainty score from the existing LSS categorical depth posterior.

不要写成：

> BEVFusion commonly uses depth entropy for pseudo-label filtering.

这样既能保持学术合理性，也避免被审稿人质疑“把已有方法说成自己的”或“引用依据不足”。

---

## 四、工程落地路径

### 4.1 现有代码落点

- `pcdet/models/view_transforms/depth_lss.py`
  - `get_cam_feats()` 已产生 `depth_prob`；
  - 当前保存 `depth_conf_map = depth_prob.max(dim=2)[0]`；
  - 下一步可新增 `depth_entropy_map = -sum(p log p) / log(D)`。

- `pcdet/tta_methods/mos.py`
  - `save_pseudo_label_batch()` 已有伪标签过滤入口；
  - `_apply_depth_uncertainty_filter()` 保留为 `F1 + D1.3` 基线的一部分；
  - 已确认 `score_relax_verify` 依赖 ignored boxes 的接入点拿不到 relaxed candidates；
  - 下一步应新增 `raw_geometry_direct_promote`：在调用 `memory_ensemble_utils.save_pseudo_label_batch()` 前，从 raw `pred_dicts` 中抽取 target-class low-score candidates，做 LiDAR geometry verification 后直接 promote/reweight。

- `pcdet/ops/roiaware_pool3d/roiaware_pool3d_utils.py`
  - 可复用 `points_in_boxes_cpu/gpu` 或 roiaware pooling 提取 pseudo box 内点；
  - 计算 `N_pts`、`rho_pts`、`z_span`、`z_var`。

- `pcdet/models/dense_heads/transfusion_head.py`
  - 若后续需要真正 per-instance loss weighting，可在 `get_targets_single()` / `loss()` 中通过 `label_weights` 与 `bbox_weights` 接入；第一版优先做过滤/软 gate，不急于改 head。

### 4.2 推荐实现顺序

1. **LiDAR geometry probe**
   - 在 `fog s3 full-val` 上导出 fused pseudo boxes 内点云几何统计；
   - 对比 TP/FP proxy 的 `N_pts`、`rho_pts`、`z_span`、`z_var` 分布；
   - 优先分析 `pedestrian` 与 `traffic_cone`。

2. **Raw-proposal geometry verification**
   - `F1 + D1.3`
   - `F1 + raw candidate dedup only`
   - `F1 + raw_geometry_direct_reweight`
   - `F1 + raw_geometry_direct_promote`（只作为对照 ablation）
   - `F1 + raw_geometry_direct_reweight + entropy auxiliary`（只作为后续 ablation）

   `fog_s3_geometry_probe` 显示当前 `D1.3` 高分框本身已经具有约 `95%` 量级的 2m TP proxy，因此第一版 verifier 不应主要做高分 hard filtering。最新 raw-window 诊断进一步显示：`pedestrian` 有少量 raw relaxed candidates，但它们没有进入 ignored boxes；`traffic_cone` 当前几乎全低于 `NEG_THRESH`。因此下一步应直接在 raw proposal 阶段验证低置信候选，而不是继续调 ignored-box geometry gate。

3. **短 TTA 验证**
   - 只有当 pseudo-label precision / retained count / FP 控制改善后，才跑 `fog s3` TTA；
   - `fog s3` 成立后，再用 `fog s5` 做强退化确认。

   初版 `score relaxation + geometry verifier` 的 `fog s3` 短 TTA 已显示 best 仍停在初始化附近，说明单纯进入自训练没有带来稳定增益。下一步评价重点应从最终 NDS 暂时前移到 raw proposal 诊断：`raw_relaxed_window`、`geometry_checked`、`direct_promoted`、`direct_reweighted`，以及 promoted boxes 的 `point_density / z_span` 与 TP/FP proxy 分布。

   最新 `fog_s3_raw_window_diag` 显示：`pedestrian_raw_total=419`、`raw_after_neg=16`、`raw_relaxed_window=11`、`raw_above_score=5`，但 `ignored_relaxed=0`、`promoted=0`；`traffic_cone_raw_total=12` 且全部低于 `NEG_THRESH`。后续 `raw_direct_promote` soft 试验进一步表明，虽然 `geometry_checked` 和 `promoted` 已经非零，但 `Pred[pedestrian]` 会异常膨胀到 50 万量级，总体 `NDS/mAP` 仍基本持平。因此下一步的主方法不应是继续 aggressive promote，而应转向 **candidate dedup + raw_direct_reweight**：先压缩低分重复框，再做软几何置信度校准。

---

## 五、实验设计

### 主 benchmark

- `fog s3 full-val`: 主开发 benchmark；
- `fog s5 full-val`: 强退化确认；
- `fog s1`: sanity check，不作为主结论。

### 后续扩展

- `rain s3` 与 `sunlight s3`: 方法成型后扩展；
- night 602 帧: 真实 adverse scene 补充；
- KITTI-C: 跨数据集补充，不在第一阶段铺开。

### 关键指标

- NDS / mAP；
- `pedestrian` 与 `traffic_cone` 的 per-class AP；
- pseudo-label precision / retained count；
- raw proposal window 中可验证候选数、dedup 后候选数、direct promoted/reweighted 数、TP proxy、retained count；
- 训练期 `geom_filter/*` / `raw_geometry/*` 统计：`raw_relaxed_window`、`raw_nms_kept`、`geometry_checked`、`direct_promoted`、`direct_reweighted`、reweighted/promoted 框几何均值；
- `pedestrian` / `traffic_cone` 的 `N_pts`、`rho_pts`、`z_span`、`z_var` 的 TP/FP 分布差异；
- TTA 早期 iter 是否优于最终 epoch。

---

## 六、论文叙事

论文主线应从以下逻辑展开：

1. full-val fog benchmark 证明 dual-fusion 在强 fog 下可能弱于单模态；
2. forward-only probe 证伪了简单的 `B_L/B_C` 对称几何冲突指标；
3. 进一步揭示强 fog 下的非对称退化：Camera-only proposal 支持稀疏，fused pseudo labels 更依赖 LiDAR；
4. 提出不改模型架构的 parameter-free reliability：LiDAR geometry verification，用 surface reflection vs volumetric scattering 的物理差异过滤天气幻觉伪标签；
5. 在 `F1 + D1.3` 稳定基线之上做伪标签质量提升与 TTA 增益验证。

当前最重要的写作约束：

- 不要再把 aggregation 写成中心；
- 不要把 `B_L/B_C` 中心距离写成已成立的可靠性分数；
- 不要承诺 GaussianLSS / EDL / Dual-Expert 是当前主方法；
- 强调“off-the-shelf BEVFusion + parameter-free physical geometry verification”的实际价值。

---

## 七、近期行动清单

1. 在 `depth_lss.py` 中保留 `depth_conf_map`，新增 `depth_entropy_map`；
2. 写 clean vs `fog s3` 的 depth entropy 统计脚本；
3. 若 entropy 现象成立，扩展 `mos.py` 的伪标签过滤；
4. 再加入 point-density anchor；
5. 先做 pseudo-label quality ablation，再进入 TTA；
6. `fog s3` 成立后，再确认 `fog s5`。

*原则：先验证现象，再写训练逻辑；先改善伪标签质量，再追求 TTA 指标。*
