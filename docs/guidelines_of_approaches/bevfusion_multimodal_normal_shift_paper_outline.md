# BEVFusion 正常场景多模态自适应论文骨架

## 1. 论文定位

推荐第一篇论文定位为：

> **面向正常场景部署偏移的单车多模态 3D 检测轻量自适应**

工作边界：

- 单车（single-agent）
- 多模态（LiDAR + Camera）
- 基于现成 `BEVFusion`
- source-free
- 不引入新预训练
- 不依赖协同/V2X代码库

## 2. 题目候选

### 候选 A

**Cross-Modal Reliability Calibration for Source-Free Adaptation of BEVFusion**

### 候选 B

**Lightweight Multimodal Adaptation for BEVFusion under Normal-Scene Deployment Shift**

### 候选 C

**Parameter-Efficient Multimodal Adaptation for BEVFusion via Reliability-Guided Calibration**

## 3. 一句话贡献

> 我们在不改动 BEVFusion 主干架构的前提下，利用 LiDAR 与 Camera 在 BEV 空间中的内部可靠性结构，对 fused prediction / pseudo labels 进行轻量校准，从而在正常场景部署偏移下实现稳定的 source-free 多模态自适应。

## 4. 创新点骨架

### 主创新

**Cross-Modal Reliability Calibration**

- 目标不是重新设计融合器，而是对已有融合结果进行可靠性重估；
- 融合 `LiDAR BEV`、`Image BEV`、depth posterior、单模态与融合预测的一致性信息；
- 重点放在 `reweight / calibration`，避免 aggressive promote。

### 辅创新 1

**Selective Adaptation Boundary**

- 延续并系统化 `freeze + fix_nan` 的稳定策略；
- 强调 adaptation 只作用于最需要更新的局部组件或伪标签路径。

### 辅创新 2

**Deployment-Style Normal-Scene Shift Benchmarking**

- 不把第一阶段写成重型跨仓库 cross-dataset；
- 在现有 `nuScenes + BEVFusion` 上构造可控 normal-scene target shifts；
- 强调方法先在工程闭环、解释性和稳定性上站住。

## 5. 论文结构建议

### 5.1 Introduction

核心逻辑：

1. 多模态 3D 检测在真实部署中会遭遇轻度但持续的分布偏移；
2. 现有多模态 adaptation 要么过重、要么不稳定、要么依赖复杂训练系统；
3. 对 BEVFusion 这类大模型来说，轻量、source-free、可解释的方法更有实际价值；
4. 提出 cross-modal reliability calibration。

### 5.2 Related Work

建议分四块：

1. multimodal 3D detection / BEVFusion
2. test-time adaptation / source-free adaptation
3. parameter-efficient adaptation
4. multimodal reliability / uncertainty calibration

### 5.3 Method

建议结构：

1. BEVFusion baseline and deployment shift setting
2. cross-modal reliability formulation
3. calibration / reweight pipeline
4. selective adaptation boundary
5. optional lightweight adapter extension

### 5.4 Experiments

主表：

1. 三种 normal-scene shift 上的总体结果
2. 与 `source-only` / `vanilla` / `freeze baseline` / `method` 对比

ablation：

1. reliability cues 拆分
2. freeze vs no-freeze
3. calibration vs adapter
4. early-iter / stability analysis

### 5.5 Discussion

可以自然衔接 archived adverse-weather 线：

- 正常场景下方法成立；
- 强 adverse-weather 会进入非对称模态塌缩，更适合作为后续研究；
- 这样既保留研究深度，也不把第一篇拖进高风险问题。

## 6. 实验矩阵

### 第一阶段 baseline

1. `B0`: source-only
2. `B1`: vanilla adaptation
3. `B2`: `fix_nan + F1 freeze`

### 第一阶段主方法

1. `M1`: `B2 + cross-modal reliability calibration`
2. `M1a`: `M1 + depth entropy cue`
3. `M1b`: `M1 + branch-disagreement cue`

### 第二阶段备选

1. `M2`: `B2 + fusion adapter`
2. `M2a`: `M2 + calibration`

## 7. 结果判断标准

以下条件满足时即可停止扩展、转入写作：

1. 至少 2 个 normal-scene shift 上优于 `B0/B1/B2`；
2. 增益不是单个偶然 iter；
3. 统计量没有爆炸；
4. 方法解释可以清楚写成“跨模态可靠性校准”。

## 8. 与恶劣天气研究的关系

恶劣天气线在这篇论文中的角色：

1. 不是主实验；
2. 可以作为 supplementary discussion；
3. 可以作为毕业论文扩展章节；
4. 保留为第二篇工作的直接起点。

推荐写法：

> Our archived adverse-weather exploration suggests that multimodal adaptation under severe corruption is dominated by asymmetric modality collapse and pathological pseudo-label noise. We therefore first target the more tractable normal-scene deployment-shift setting, where lightweight reliability calibration can be studied in a cleaner and more controlled regime.
