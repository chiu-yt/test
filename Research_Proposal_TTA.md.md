# 硕士研究课题开题与执行报告

**课题名称：** 基于非对称可靠性建模的多模态 3D 目标检测测试时自适应研究  
**(Asymmetric Reliability Modeling for Multimodal 3D Object Detection Test-Time Adaptation)**

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

因此，当前更准确的科学问题不是“如何度量 LiDAR 与 Camera 的对称几何冲突”，而是：

> 在强 fog 下，多模态检测发生非对称退化：fused pseudo labels 主要由 LiDAR 几何支撑，而 Camera/LSS 深度分支表现为高不确定性。TTA 应如何在不改预训练模型架构的条件下，利用 LiDAR 支持度与深度不确定性估计伪标签可靠性？

---

## 二、核心创新结构

### 大创新：LiDAR 锚定的非对称伪标签可靠性建模

不再要求 `B_L` 与 `B_C` 必须达成几何一致，而是将 LiDAR 点云支持度作为 pseudo-label 的物理锚点，同时用 LSS depth entropy 惩罚 Camera/depth 不确定性。

第一版可靠性形式：

```text
W_i = Score_i * R_lidar(N_i) * exp(-lambda * H_depth(i))
```

- `Score_i`: fused detector score；
- `R_lidar(N_i)`: pseudo box 内真实点数得到的 LiDAR 支持度；
- `H_depth(i)`: 现有 LSS depth probability 的归一化熵；
- `lambda`: 第一版固定或少量离散验证，避免过度调参。

### 小创新 1：Selective Adaptation Boundary

继续保留并系统化当前已验证的 F1 策略：冻结 `image_backbone + neck`，避免 fog 下脏视觉梯度破坏图像先验。

### 小创新 2：Class-aware Depth-aware Denoising

继续保留 `F1 + D1.3` 主线：只对 `pedestrian=0.24` 与 `traffic_cone=0.34` 做深度相关过滤，并将其作为非对称可靠性建模的基础版本。

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
  - `_apply_depth_uncertainty_filter()` 可扩展为 entropy-aware；
  - `_apply_multimodal_conflict_filter()` 应改名或重写为 asymmetric reliability filter，避免继续表达对称冲突假设。

- `pcdet/ops/roiaware_pool3d/roiaware_pool3d_utils.py`
  - 可复用 `points_in_boxes_cpu/gpu` 计算 pseudo box 内点数。

- `pcdet/models/dense_heads/transfusion_head.py`
  - 若后续需要真正 per-instance loss weighting，可在 `get_targets_single()` / `loss()` 中通过 `label_weights` 与 `bbox_weights` 接入；第一版优先做过滤/软 gate，不急于改 head。

### 4.2 推荐实现顺序

1. **Depth entropy 现象验证**
   - 在 clean 与 `fog s3` 上导出/统计 depth entropy；
   - 先确认 fog 下熵是否显著升高，尤其是 `pedestrian / traffic_cone` 附近。

2. **Forward-only pseudo-label quality ablation**
   - `F1 + D1.3`
   - `F1 + D1.3 + depth entropy only`
   - `F1 + D1.3 + LiDAR point-density only`
   - `F1 + D1.3 + depth entropy + LiDAR density`

3. **短 TTA 验证**
   - 只有当 pseudo-label precision / retained count / FP 控制改善后，才跑 `fog s3` TTA；
   - `fog s3` 成立后，再用 `fog s5` 做强退化确认。

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
- 被 entropy / point-density 筛掉的低质量框比例；
- TTA 早期 iter 是否优于最终 epoch。

---

## 六、论文叙事

论文主线应从以下逻辑展开：

1. full-val fog benchmark 证明 dual-fusion 在强 fog 下可能弱于单模态；
2. forward-only probe 证伪了简单的 `B_L/B_C` 对称几何冲突指标；
3. 进一步揭示强 fog 下的非对称退化：Camera-only proposal 支持稀疏，fused pseudo labels 更依赖 LiDAR；
4. 提出不改模型架构的 parameter-free reliability：LiDAR point-density anchor + LSS depth entropy；
5. 在 `F1 + D1.3` 稳定基线之上做伪标签质量提升与 TTA 增益验证。

当前最重要的写作约束：

- 不要再把 aggregation 写成中心；
- 不要把 `B_L/B_C` 中心距离写成已成立的可靠性分数；
- 不要承诺 GaussianLSS / EDL / Dual-Expert 是当前主方法；
- 强调“off-the-shelf BEVFusion + parameter-free uncertainty mining”的实际价值。

---

## 七、近期行动清单

1. 在 `depth_lss.py` 中保留 `depth_conf_map`，新增 `depth_entropy_map`；
2. 写 clean vs `fog s3` 的 depth entropy 统计脚本；
3. 若 entropy 现象成立，扩展 `mos.py` 的伪标签过滤；
4. 再加入 point-density anchor；
5. 先做 pseudo-label quality ablation，再进入 TTA；
6. `fog s3` 成立后，再确认 `fog s5`。

*原则：先验证现象，再写训练逻辑；先改善伪标签质量，再追求 TTA 指标。*
