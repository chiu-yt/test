# BEVFusion 正常场景多模态自适应路线图

## 1. 当前主线定位

当前第一篇论文主线固定为：

> **单车多模态 BEVFusion 在正常场景部署偏移下的轻量自适应**

关键词：

- `single-agent`
- `multimodal`
- `BEVFusion-native`
- `source-free`
- `lightweight adaptation`

该主线明确不依赖：

- 新代码库；
- 协同/V2X 数据链；
- source-side 重新预训练；
- 大型 teacher-student UDA 系统。

## 2. 现实约束

### 2.1 当前仓库的真实承载能力

- 现成成熟的多模态主链是 `nuScenes + BEVFusion`；
- `KITTI` 虽有图像-点云多模态配置，但不是同一条 `BEVFusion` 路线；
- 当前仓库没有现成 `DAIR-V2X-C`、`V2V4Real`、`collaborative perception` 的训练与评测脚手架。

因此，第一阶段不应把题目写成依赖新数据栈的协同/UDA 项目。

### 2.2 当前不建议的方向

1. **协同/V2X/S2C-UDA 继承路线**
   - 没有代码时，工程成本接近重写；
   - 与当前单车 BEVFusion 主链结构差异过大。

2. **恶劣天气在线 TTA 作为第一篇主线**
   - 风险高，机制尚未闭环；
   - 已转为保留研究资产。

3. **重型 teacher-student UDA 系统**
   - 调试成本高；
   - 容易再次进入“工程大、结果弱”的状态。

## 3. 核心科学问题

新的核心问题是：

> 在 BEVFusion 主体保持不变的前提下，如何利用 LiDAR 与 Camera 在正常场景下的内部几何与置信度结构，对 fused pseudo labels 或 fused BEV features 进行轻量校准，从而实现稳定的 source-free adaptation？

这里的重点不是“彻底重建融合结构”，而是：

- 让多模态内部信息更可信；
- 让 adaptation 更稳定；
- 让工程代价足够小，适合快速形成可发表结果。

## 4. 优先方法顺序

### 4.1 首选：Cross-Modal Reliability Calibration

推荐作为第一版主方法。

思路：

- 复用当前 `LiDAR BEV`、`Image BEV`、`depth posterior / entropy`、单模态/融合预测；
- 构造 fused pseudo-label 的跨模态可靠性分数；
- 以 `reweight / calibration` 为主，而不是激进 promote。

可行原因：

- 改动集中在现有 `BEVFusion + MOS/TTA` 周边；
- 与当前 adverse-weather 线积累的 probe/diagnostic 工具兼容；
- 论文叙事容易控制在“轻量、可解释、BEVFusion-native”。

### 4.2 次选：Parameter-Efficient Fusion Adapter

思路：

- 冻结 `image_backbone`、`backbone_3d`、`neck`；
- 在 `FUSER` 后或 `BACKBONE_2D` 前插入小型 adapter；
- 只更新极少量参数。

优点：

- 工程可控；
- 更符合“车载算力受限”的实际背景；
- 适合作为 reliability calibration 不显著时的第二条备选线。

### 4.3 第三选择：Backprop-Free Checkpoint / Model Merging

思路：

- 利用 BEV 融合特征或预测签名对历史 checkpoint 进行组合；
- 不走 full backprop adaptation；
- 优先作为稳态替代路线，而不是当前第一优先。

## 5. 第一阶段 benchmark 设计

### 5.1 第一阶段不写成跨仓库 cross-dataset

第一阶段优先使用当前 `nuScenes + BEVFusion` 管线，构造**正常场景部署偏移**：

- 轻度相机风格偏移；
- 轻度相机尺度/裁剪偏移；
- 点云稀疏化或 beam dropout；
- 轻度模态不对齐；
- 轻度 calibration-like perturbation。

原因：

- 与当前仓库最匹配；
- 能保证实验链快速闭环；
- 先证明方法在单车多模态正常 shift 下成立，再决定是否扩大到真正 cross-dataset。

### 5.1.1 推荐的第一版 target shift 组合

第一阶段建议把 normal-scene target shift 写成 **deployment-style multimodal mismatch**，而不是重新定义数据集。

推荐按以下优先级分三组：

1. **Camera style / appearance shift**
   - brightness / contrast / color temperature perturbation
   - mild blur / JPEG-like degradation
   - 目的：模拟不同相机 ISP、曝光、成像风格

2. **Camera geometry / resize shift**
   - image resize range perturbation
   - crop / principal-point-like offset
   - 目的：模拟不同安装、不同预处理与轻度标定误差

3. **LiDAR sparsity / modality mismatch shift**
   - random beam dropout
   - point thinning / sweep sparsification
   - 目的：模拟不同雷达线数、采样密度与同步差异

### 5.1.2 第一阶段不建议直接做的 shift

以下设置不适合作为起跑版：

1. **强 fog / rain / night**
   - 会再次把问题拉回病态噪声与非对称塌缩；
   - 不利于验证新主线是否本身有效。

2. **完整跨仓库 cross-dataset**
   - 当前仓库没有现成 BEVFusion-on-KITTI/Waymo 的同构主链；
   - 容易把第一阶段变成数据与配置重构工程。

3. **复杂多因子联合 shift**
   - 同时叠多种 shift 会模糊归因；
   - 第一阶段应优先保证解释性。

### 5.2 Baseline 组

至少准备：

1. `source-only BEVFusion`
2. `vanilla self-training / TTA`
3. `fix_nan + freeze baseline`

### 5.2.1 推荐 baseline 命名

为了后续实验记录和论文表格统一，建议从第一轮开始固定命名：

1. `B0`: source-only
2. `B1`: source-free vanilla adaptation
3. `B2`: `fix_nan + F1 freeze`
4. `M1`: `B2 + cross-modal reliability calibration`
5. `M2`: `B2 + fusion adapter`

### 5.3 第一阶段目标

只需要先证明：

> 轻量多模态 adaptation 方法能够在正常场景 target shift 下，比 source-only 和 naive adaptation 更稳且更可解释。

### 5.4 第一阶段成功标准

满足以下 4 条中的 3 条，就可以认为主线成立：

1. 在至少 2 种 normal-scene shift 下稳定优于 `B0` 和 `B1`；
2. 最优点不是单个偶然 iter，而是一个短区间内可重复；
3. 伪标签统计或可靠性统计没有出现明显爆炸；
4. 方法解释可以清楚落到“跨模态可靠性校准”而不是纯阈值技巧。

## 6. 两周起跑计划

### Day 1-3

- 固定 `nuScenes + BEVFusion` 主链；
- 定义第一版正常场景 target shift；
- 跑通 `source-only`、`vanilla TTA`、`fix_nan + freeze`。

推荐先跑：

1. `camera_style_shift_s1`
2. `camera_resize_shift_s1`
3. `lidar_sparsity_shift_s1`

### Day 4-7

- 落地第一版 `cross-modal reliability calibration`；
- 加最小统计和可解释日志；
- 不引入复杂新结构。

### Day 8-10

- 做最小 ablation：
  - baseline
  - + calibration
  - + one auxiliary cue

推荐 auxiliary cue：

1. depth entropy only
2. LiDAR / Camera branch score disagreement
3. BEV feature cosine agreement

### Day 11-14

- 若结果稳定提升，锁题并开始写论文；
- 若结果无实质增益，切到 `fusion adapter` 路线；
- 不回头扩展到 V2X 或高风险 adverse-weather 主线。

## 7. 与 archived adverse-weather 线的关系

此前 fog/night 方向的资产全部保留：

- `fix_nan`
- freeze 策略
- best-iter 自动评估
- conflict/depth/geometry probes
- 伪标签质量分析工具

这些资产现在的角色是：

1. 为新主线提供工程基础；
2. 为第二篇 adverse-weather 研究保留直接入口；
3. 在毕业论文中作为“高难扩展 / future work”章节支撑研究深度。

## 8. 当前一句话原则

> **先在当前 BEVFusion 单车多模态主链上，用最小结构改动做出稳定、可解释、可发表的正常场景 adaptation 结果；恶劣天气与协同感知都保留，但不再抢占第一篇论文主线。**

## 9. 近期具体执行清单

1. 为 `nuScenes + BEVFusion` 定义三种 normal-scene shift 配置：`camera_style`、`camera_resize`、`lidar_sparsity`。
2. 在现有 `bevfusion_mos.yaml` 基础上复制出正常场景 adaptation 配置，而不是继续复用 adverse-weather 命名。
3. 跑 `B0/B1/B2` 三条 baseline，记录统一实验表。
4. 先在 `mos.py` 或紧邻模块落地 `M1: cross-modal reliability calibration`。
5. 若 `M1` 信号弱，再进入 `M2: fusion adapter`，不直接跳大型 teacher-student。

## 10. 第一批可运行命令

以下命令默认在 `tools/` 目录执行。

### 10.1 B0: source-only + camera_style_shift_s1

```bash
python test.py \
  --cfg_file cfgs/nuscenes_models/bevfusion_mos_normal_shift.yaml \
  --ckpt ../output/nuscenes_models/bevfusion/multimodal_baseline_4gpu/ckpt/checkpoint_epoch_10.pth \
  --batch_size 4 \
  --eval_tag normal_shift_b0_camera_style_s1 \
  --set \
  DATA_CONFIG.CORRUPTION.ENABLED True \
  DATA_CONFIG.CORRUPTION.APPLY_IN "['test']" \
  DATA_CONFIG.CORRUPTION.IMAGE_STYLE.ENABLED True \
  DATA_CONFIG.CORRUPTION.IMAGE_STYLE.SEVERITY 1 \
  DATA_CONFIG.CORRUPTION.IMAGE_GEOMETRY.ENABLED False \
  DATA_CONFIG.CORRUPTION.LIDAR_SPARSITY.ENABLED False \
  TTA.ENABLED False
```

### 10.2 B0: source-only + camera_resize_shift_s1

```bash
python test.py \
  --cfg_file cfgs/nuscenes_models/bevfusion_mos_normal_shift.yaml \
  --ckpt ../output/nuscenes_models/bevfusion/multimodal_baseline_4gpu/ckpt/checkpoint_epoch_10.pth \
  --batch_size 4 \
  --eval_tag normal_shift_b0_camera_resize_s1 \
  --set \
  DATA_CONFIG.CORRUPTION.ENABLED True \
  DATA_CONFIG.CORRUPTION.APPLY_IN "['test']" \
  DATA_CONFIG.CORRUPTION.IMAGE_STYLE.ENABLED False \
  DATA_CONFIG.CORRUPTION.IMAGE_GEOMETRY.ENABLED True \
  DATA_CONFIG.CORRUPTION.IMAGE_GEOMETRY.SEVERITY 1 \
  DATA_CONFIG.CORRUPTION.LIDAR_SPARSITY.ENABLED False \
  TTA.ENABLED False
```

### 10.3 B0: source-only + lidar_sparsity_shift_s1

```bash
python test.py \
  --cfg_file cfgs/nuscenes_models/bevfusion_mos_normal_shift.yaml \
  --ckpt ../output/nuscenes_models/bevfusion/multimodal_baseline_4gpu/ckpt/checkpoint_epoch_10.pth \
  --batch_size 4 \
  --eval_tag normal_shift_b0_lidar_sparsity_s1 \
  --set \
  DATA_CONFIG.CORRUPTION.ENABLED True \
  DATA_CONFIG.CORRUPTION.APPLY_IN "['test']" \
  DATA_CONFIG.CORRUPTION.IMAGE_STYLE.ENABLED False \
  DATA_CONFIG.CORRUPTION.IMAGE_GEOMETRY.ENABLED False \
  DATA_CONFIG.CORRUPTION.LIDAR_SPARSITY.ENABLED True \
  DATA_CONFIG.CORRUPTION.LIDAR_SPARSITY.SEVERITY 1 \
  TTA.ENABLED False
```

### 10.4 B1/B2 起跑命令模板

当 `B0` 三组跑通后，再切换到 adaptation：

```bash
python train.py \
  --cfg_file cfgs/nuscenes_models/bevfusion_mos_normal_shift.yaml \
  --ckpt ../output/nuscenes_models/bevfusion/multimodal_baseline_4gpu/ckpt/checkpoint_epoch_10.pth \
  --batch_size 2 \
  --epochs 1 \
  --extra_tag normal_shift_b2_camera_style_s1 \
  --set \
  DATA_CONFIG.CORRUPTION.ENABLED True \
  DATA_CONFIG.CORRUPTION.APPLY_IN "['train','test']" \
  DATA_CONFIG.CORRUPTION.IMAGE_STYLE.ENABLED True \
  DATA_CONFIG.CORRUPTION.IMAGE_STYLE.SEVERITY 1 \
  DATA_CONFIG.CORRUPTION.IMAGE_GEOMETRY.ENABLED False \
  DATA_CONFIG.CORRUPTION.LIDAR_SPARSITY.ENABLED False \
  TTA.ENABLED True
```

说明：

- `B1`：关闭 `fix_nan + freeze` 之外的额外主线策略，仅保留 vanilla adaptation；
- `B2`：保留当前 `fix_nan + F1 freeze` 稳定化设置；
- 第一轮只建议用 `severity=1`，先确认方法在轻度 deployment shift 下有正信号。
