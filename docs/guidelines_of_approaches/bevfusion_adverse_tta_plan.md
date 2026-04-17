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

## 14. 当前最推荐的主结论

如果后续实验顺利，论文最终最适合落在下面这类结论：

> 在多模态 3D 目标检测中，adverse weather / illumination shift 下的 TTA 主要受 noisy pseudo labels 限制；
> 与继续增强 checkpoint aggregation 相比，面向模态冲突与伪标签可靠性的适应机制更能稳定提升性能。

这个结论比“夜间有效”更通用，比“通用所有 domain”更稳，也更适合快速形成一篇完整论文。
