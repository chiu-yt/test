# BEVFusion 恶劣条件 TTA 论文与实验路线图

## 1. 研究定位

当前工作的更优主线，不建议继续收缩为“夜间部署专用 TTA”，而应提升为：

> **面向恶劣条件 domain shift 的多模态 3D 目标检测 Test-Time Adaptation（TTA）**

核心目标不是证明“夜间有效”，而是证明：

1. 多模态 3D 检测在恶劣天气/光照条件下存在明显 domain shift；
2. 伪标签噪声是 TTA 失效的主要瓶颈；
3. 一个面向噪声与模态冲突的 TTA 框架，能够稳定提升 BEVFusion 在 adverse conditions 下的检测精度。

---

## 2. 主 benchmark 选择

### 2.1 主 benchmark

**nuScenes-C**

原因：

- 与当前 BEVFusion/OpenPCDet 代码链最一致；
- 多模态设置自然；
- 工程迁移成本最低；
- 更适合作为主实验表，保证尽快出结果。

### 2.2 次 benchmark

**KITTI-C**

作用：

- 用于补充“不是只在 nuScenes-C 上有效”；
- 放在第二阶段，等 nuScenes-C 主结果成型后再补；
- 不建议在第一阶段同时铺开，避免实验线过长。

### 2.3 补充真实场景实验

**当前 602 帧 night subset**

作用：

- 作为真实低照 target stream 的 supplementary case study；
- 用来说明方法不仅在合成 corruption 上有效，也能迁移到真实 adverse condition；
- 不作为全文主 benchmark。

---

## 3. 建议优先选择的 corruption 条件

为了快速形成论文主线，建议先做以下三类：

1. **Fog（雾）**
2. **Rain（雨）**
3. **Sunlight / glare（强光）**

选择理由：

- `fog`：同时影响相机与 LiDAR，最适合做多模态 TTA；
- `rain`：是最常见、最容易被接受的 adverse weather 场景之一；
- `sunlight`：更偏图像链路退化，能体现多模态方法相对单模态的意义。

### 暂不作为主线的条件

- `night`：保留为补充实验；
- `snow`：可做扩展，但优先级低于 fog/rain；
- `shadow / sand-dust / smoke`：先不展开，避免 benchmark 与论文故事过散。

---

## 4. 论文主故事建议

推荐主标题方向：

> **Test-Time Adaptation for Multimodal 3D Object Detection under Adverse Weather and Illumination Shifts**

更具体的论文叙事应围绕以下三点展开：

### 4.1 问题定义

- 多模态 3D 检测器在 adverse condition 下性能明显下降；
- 简单 self-training 容易被 noisy pseudo labels 拖垮；
- 多模态场景下，伪标签质量问题比“是否做 aggregation”更关键。

### 4.2 主要观察

- 高噪类别是 TTA 主瓶颈；
- 历史 checkpoint 聚合未必真正提供增益，尤其在权重接近均匀时；
- 稳定训练语义与防 NaN 机制，对在线适应至关重要。

### 4.3 方法落点

方法重点应落在：

- **pseudo-label denoising / reliability modeling**
- **modality conflict-aware weighting**
- 而不是继续强调 checkpoint aggregation 本身

---

## 5. 当前代码主线如何保留

当前已验证有价值的部分，可以继续保留：

1. `F1`：冻结 `image_backbone + neck`
2. 深度相关过滤思路
3. NaN / 非有限值防护
4. best-iter 自动评估

这些应被视为：

- 稳定主线的工程基础；
- 后续 adverse-condition 实验的 default setting；
- 而不是论文的全部 novelty。

---

## 6. 第一阶段实验矩阵（建议优先完成）

### 6.1 Baseline 组

在 `nuScenes-C` 上至少准备以下对照：

1. **Source-only BEVFusion**
2. **Vanilla TTA / self-training baseline**
3. **当前稳定版 MM-MOS（含 fix_nan）**

### 6.2 Condition 组

先只做：

- Fog
- Rain
- Sunlight

### 6.3 表格组织建议

主表 1：不同条件下的总体指标

- mAP
- NDS
- 相对 source-only 的提升

主表 2：per-class 结果（至少列关键类）

- car
- truck
- pedestrian
- traffic_cone
- bicycle
- barrier

### 6.4 第一阶段目标

先证明：

> 方法在 3 个主要 adverse shifts 上都能比 source-only 和普通 TTA 更稳。

只要这个主结论成立，论文主线就站住了。

---

## 7. 第二阶段 ablation 设计

重点做少而关键的 ablation，不要把表做得过大。

### 7.1 推荐 ablation

1. **freeze vs no-freeze**
2. **depth filter on/off**
3. **conflict-aware filter / weighting on/off**
4. **fix_nan on/off**（作为稳定性分析，不一定单独成主表）
5. **aggregation on/off**

### 7.2 最关键的归因实验

优先做：

1. `fix_nan_only`
2. `fix_nan + conflict-aware`
3. `fix_nan + no_aggregation`

这个组合能快速回答：

- 收益到底来自稳定性修复，还是来自新的过滤/权重策略；
- aggregation 在多模态 TTA 里是否真的值得保留。

---

## 8. 第三阶段补充实验

在主实验已经成立后，再补：

### 8.1 KITTI-C

目的：

- 证明方法不是 nuScenes-C 特例；
- 提升论文说服力。

### 8.2 Night subset

目的：

- 作为真实场景补充验证；
- 强调方法不仅能处理 synthetic corruption，也对真实低照 shift 有帮助。

注意：

- `night` 不要升格为主 benchmark；
- 应以 supplementary / secondary experiment 的地位呈现。

---

## 9. 不建议作为主线的方向

### 9.1 完整混合数据集 TTA

不建议作为主实验主线，原因：

- target domain 不纯；
- 会冲淡 domain shift；
- 难以清楚回答“模型到底在适配什么”。

### 9.2 夜间专用 TTA 论文

不建议作为唯一主线，原因：

- benchmark 规模偏小；
- 故事偏窄；
- 更容易被理解成场景特化 engineering work。

### 9.3 过于泛化的“通用所有 domain TTA”

当前不建议，原因：

- proof burden 太高；
- 实验量太大；
- 不利于快速投稿。

---

## 10. 适合快速发 SCI 二区的策略

### 10.1 推荐路线

**主线：**

- adverse weather / illumination TTA
- 主 benchmark 用 nuScenes-C
- 条件选 fog/rain/sunlight

**副线：**

- KITTI-C 做泛化补充
- night subset 做真实场景补充

### 10.2 论文 novelty 不要押在什么地方

不建议把 novelty 主要押在：

- checkpoint bank 更复杂；
- 更细的阈值调参；
- iter-only/epoch-only 的 bank 细节。

### 10.3 论文 novelty 更适合押在什么地方

更建议押在：

- adverse-condition 下 pseudo-label noise 的诊断
- multimodal conflict-aware reliability modeling
- noise-robust online adaptation

---

## 11. 推荐的近期执行顺序

### 第 1 周

1. 确认 `nuScenes-C` 数据可用
2. 先跑 `source-only` 与 `fix_nan stable baseline`
3. 只测 fog/rain/sunlight 三个条件

### 第 2 周

1. 做核心 ablation：freeze / conflict / aggregation
2. 固定一版最稳配置
3. 输出主表草稿

### 第 3 周

1. 补 `KITTI-C` 或 `night subset`
2. 补 failure case 分析
3. 开始写论文初稿

---

## 12. nuScenes-C benchmark 实施策略

### 12.1 主实现路径选择

在当前项目阶段，推荐采用下面的分工：

- **`3D_Corruptions_AD`：主实现路径**
- **`Robo3D`：benchmark 规范与数据组织参考**

原因如下：

1. `3D_Corruptions_AD` 明确面向 `OpenPCDet / MMDetection3D` 的 corruption 接入；
2. 对当前 `BEVFusion + TTA` 项目来说，更适合直接复用 corruption 函数，快速接到现有测试/自适应流程里；
3. `Robo3D` 更适合作为 benchmark 目录结构、severity 命名、数据组织方式的参考，而不是第一阶段的主落地代码入口。

### 12.2 第一阶段采用在线 corruption，而不是离线生成完整 nuScenes-C

第一阶段不建议直接先离线生成一整套 `nuScenes-C`，而建议：

> **保持原始 nuScenes 数据不变，在 BEVFusion / OpenPCDet 的测试与 TTA 流程中在线注入 corruption。**

原因：

- 当前服务器空间有限；
- 仍处在方法探索阶段，尚未确定最终保留哪些 corruption；
- 在线 corruption 更适合 TTA 场景，因为 corruption 会直接影响 pseudo-label 生成与在线适配过程；
- 能够更快得到第一批可用结果。

### 12.3 corruption 接入原则

在当前项目中，corruption 的注入顺序应为：

```text
原始 nuScenes sample
-> 加载点云 / 多相机图像
-> 注入 corruption
-> 模型预测
-> pseudo label 生成
-> TTA 更新
-> evaluation
```

这保证 TTA 适配的是真正的 corrupted target domain，而不是只在 evaluation 端做后处理。

### 12.4 第一阶段优先接入的 corruption

只做以下三类：

1. `fog`
2. `rain`
3. `sunlight / glare`

暂不优先：

- `snow`
- `temporal misalignment`
- `spatial misalignment`
- 各类 object-level corruption

原则是先把 **weather / illumination 主线** 跑通，再扩展更复杂 corruption。

---

## 13. 第一阶段执行步骤（建议严格按顺序推进）

### 13.1 准备阶段

1. **清理服务器空间**
   - 保留主 baseline checkpoint；
   - 保留当前稳定版 TTA run；
   - 删除明显失败、重复、无论文价值的旧 output。

2. **准备参考仓库，但不要替换主项目**
   - 单独拉取 `3D_Corruptions_AD`；
   - 单独保存 `Robo3D` 文档作为 benchmark 参考；
   - 不要直接把它们覆盖进当前 OpenPCDet 项目。

3. **固定当前项目默认起点**
   - 使用当前稳定版 `BEVFusion + fix_nan + best_iter auto eval`；
   - 固定一版基础 cfg，作为后续所有 adverse-condition 实验的统一起点。

### 13.2 第一批最小落地目标

第一阶段只追求一个最小闭环：

> **让 BEVFusion 在 fog / rain / sunlight 三个 corruption 下，先跑通 source-only，再跑通稳定版 TTA。**

不要在这个阶段同时追求：

- 离线全量 nuScenes-C 生成；
- KITTI-C；
- 一大批额外 corruption；
- 复杂 ablation。

### 13.3 工程接入顺序

按下面顺序推进：

1. **先只接一个 corruption：fog**
   - 验证多模态数据流是否被正确污染；
   - 验证 evaluation 能正常运行；
   - 验证指标会比 clean 数据下降。

2. **然后补 `rain` 与 `sunlight`**
   - 保持实现方式一致；
   - 保持统一 severity 规范。

3. **先做 source-only，再做 TTA**
   - 先证明 benchmark 正常；
   - 再证明 TTA 在 corrupted domain 下有增益。

### 13.4 第一阶段推荐实验顺序

建议严格按这个顺序：

1. `source-only + fog`
2. `source-only + rain`
3. `source-only + sunlight`
4. `stable TTA + fog`
5. `stable TTA + rain`
6. `stable TTA + sunlight`

### 13.5 第一阶段必须记录的内容

每个实验至少保存：

- `cfg_file`
- `extra_tag`
- corruption 名称
- severity
- best iter / best epoch
- NDS
- mAP
- per-class AP（至少保留 `pedestrian / traffic_cone / bicycle / barrier / car / truck`）
- 输出目录
- 关键日志片段

### 13.6 第一阶段完成标准

满足以下条件即可进入第二阶段：

1. `fog / rain / sunlight` 三类 corruption 全部跑通；
2. source-only 与 stable TTA 都有结果；
3. 至少拿到一版可汇总的主表；
4. 已经确认哪一类 corruption 最值得继续深入。

---

## 14. 最新 full-val fog benchmark 结论（已完成）

### 14.1 已有结果摘要

基于 `clean_fullval` 与 `fog s1/s3/s5 fullval` 的 source-only 实验，当前已经得到以下关键事实：

- `clean_fullval`: `NDS=0.6736`, `mAP=0.6421`
- `fog_lidar_s1_fullval`: `NDS=0.6728`, `mAP=0.6409`
- `fog_image_s1_fullval`: `NDS=0.6715`, `mAP=0.6393`
- `fog_image_lidar_s1_fullval`: `NDS=0.6705`, `mAP=0.6381`

- `fog_lidar_s3_fullval`: `NDS=0.6647`, `mAP=0.6286`
- `fog_image_s3_fullval`: `NDS=0.6655`, `mAP=0.6278`
- `fog_image_lidar_s3_fullval`: `NDS=0.6563`, `mAP=0.6134`

- `fog_lidar_s5_fullval`: `NDS=0.6320`, `mAP=0.5775`
- `fog_image_s5_fullval`: `NDS=0.6516`, `mAP=0.6025`
- `fog_image_lidar_s5_fullval`: `NDS=0.6017`, `mAP=0.5213`

### 14.2 已验证的直接结论

这些结果已经足够支持下面几个判断：

1. **`fog s1` 太弱，不适合作为后续 TTA 主 benchmark**
   - 相比 clean，`s1` 只带来极小退化；
   - 这种设置更适合作为 sanity check，而不是主表核心。

2. **`fog s3` 是当前最合适的主开发 benchmark**
   - 退化已经足够明显；
   - 但仍未极端到让实验完全不稳定；
   - 最适合后续方法开发与 ablation。

3. **`fog s5` 适合作为强退化确认 benchmark**
   - 在 `s5` 下，性能退化明显；
   - 更适合作为方法泛化与强鲁棒性验证，而不是第一开发集。

4. **强 fog 下，多模态融合会比单模态更脆弱，但问题更接近非对称退化而非对称冲突**
   - `s3` 下 dual 明显差于 lidar-only 与 image-only；
   - `s5` 下 dual 的退化最严重；
   - 后续 `fog_s3_conflict_probe` 显示，`B_L` 与 `B_C` 的中心距离和 fused GT center error 几乎无正相关；
   - 更合理的解释是：强 fog 下 Camera-only proposal 支持稀疏，高置信 fused pseudo labels 主要由 LiDAR 几何支撑，Camera 分支更多体现为深度/语义不确定性。

### 14.3 这组 benchmark 的真正意义

这一步最重要的贡献，不是单纯补齐了 benchmark，而是帮助当前项目确认了：

> **后续论文与方法创新，应该聚焦“LiDAR 锚定的非对称伪标签可靠性建模”，而不是继续做更细的阈值调参与 aggregation 微调。**

也就是说，这些 fog benchmark 已经把主问题从“是否存在 adverse shift”推进到了：

- adverse shift 下，哪一类 corruption 能作为主 benchmark；
- 多模态融合是否总是有利；
- 未来最值得投入的方法方向是什么。

---

## 15. 下一阶段方法主线（基于 forward-only probe 后的修正）

### 15.1 当前最值得做的大方向

下一阶段不建议继续优先做：

- 更细的 fog 阈值调节；
- 更复杂的 aggregation 细节；
- 把 `s1` 再扩展成更多弱 corruption 对照。

当前最值得做的方法主线应修正为：

> **LiDAR-anchored asymmetric pseudo-label reliability modeling for multimodal 3D TTA**

也就是：

- 不默认信任 fused pseudo labels；
- 不再把 `B_L` 与 `B_C` 的几何距离作为核心可靠性信号；
- 将 LiDAR 点云支持度作为几何锚点；
- 将 LSS 深度分布熵作为 Camera / depth 不确定性惩罚；
- 在不改预训练 BEVFusion 架构的前提下，对伪标签做过滤、降权或可靠性重分配。

### 15.2 一大创新 + 两个小创新的建议结构

#### 大创新

**LiDAR 锚定的非对称 pseudo-label 可靠性建模**

主张：

- 强 fog 下，fusion 结果可能比单模态更不可靠；
- 但当前 probe 不支持“LiDAR/Camera 对称几何冲突距离”作为主信号；
- 因此 TTA 不应无条件依赖 fused pseudo labels，而应使用 LiDAR 点云密度与 LSS depth entropy 估计其可靠性；
- 该方向保持 source-free / off-the-shelf BEVFusion 约束，不需要 GaussianLSS、EDL head 或 dual-expert 预训练。

#### 小创新 1

**Selective adaptation boundary**

继续保留并系统化当前已验证有效的冻结策略：

- 冻结 `image_backbone + neck`
- 只让更靠后的融合与检测部分做适配

这一点适合作为 supporting contribution，而不是主 novelty。

#### 小创新 2

**Class-aware depth-aware denoising**

继续保留并整理当前已验证有效的：

- `pedestrian`
- `traffic_cone`

高噪类深度过滤思路，将其作为类级别的 targeted denoising 支撑模块。

### 15.3 最优先的下一个方法实验

最推荐的下一实验不是直接扩到更多 corruption，而是：

1. 用 `fog s3 full-val` 作为主开发 benchmark；
2. 以 `F1 + D1.3` 为 base；
3. 先做 forward-only 伪标签质量验证，而不是直接跑完整 TTA；
4. 比较四个伪标签可靠性版本：
   - `F1 + D1.3`
   - `F1 + D1.3 + depth entropy only`
   - `F1 + D1.3 + LiDAR point-density only`
   - `F1 + D1.3 + depth entropy + LiDAR density`
5. 若伪标签 precision / retained-count / 高噪类误报控制在 `fog s3` 上改善，再跑短 TTA；
6. 若 `s3` 上成立，再在 `s5` 上做强退化确认。

推荐的第一版可靠性形式：

```text
W_i = Score_i * R_lidar(N_i) * exp(-lambda * H_depth(i))
```

其中：

- `R_lidar(N_i)` 来自 pseudo box 内点数，第一版应作为软权重或分段 gate，避免远距离/小目标被硬过滤；
- `H_depth(i)` 来自现有 LSS depth probability 的归一化熵，优先从 box center 或小窗口取值；
- `lambda` 第一版固定为 `1` 或少量离散值，不要过度调参以免削弱 parameter-free 叙事。

文献表述边界：LSS / BEVDepth / BEVFusion 已有离散 depth posterior，单目 3D 检测与 TTA 中也有 uncertainty / entropy weighting 先例；但目前不要声称 BEVFusion 标准做法已经用 depth-bin entropy 过滤 3D pseudo labels。更稳妥的表述是：从已有 depth posterior 中派生一个 zero-cost uncertainty score。

### 15.4 当前不建议优先投入的方向

在下一阶段，不建议优先把精力投入到：

- 继续做 `fog s1` TTA；
- 再次把 aggregation 拉回论文主线；
- 大量微调单个伪标签阈值；
- 把 `GaussianLSS`、EDL 或 dual-expert/PanDA 作为当前主方法；
- 过早同时展开 `rain + sunlight + night + KITTI-C`。

更合理的策略是：

- 先用 `fog s3` 建立方法主结论；
- 再用 `fog s5` 验证强退化；
- 成立之后再扩展到 `rain / sunlight / night / KITTI-C`。

---

## 16. 当前最推荐的主结论

如果后续实验顺利，论文最终最适合落在下面这类结论：

> 在多模态 3D 目标检测中，adverse weather / illumination shift 下的 TTA 主要受 noisy pseudo labels 限制；
> 与继续增强 checkpoint aggregation 相比，面向 LiDAR 几何支持与 LSS 深度不确定性的非对称伪标签可靠性机制更有希望稳定提升性能。

这个结论比“夜间有效”更通用，比“通用所有 domain”更稳，也更适合快速形成一篇完整论文。
