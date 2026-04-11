# OpenPCDet / BEVFusion Night TTA 实验记录

## 0. Auto-maintained Experiment Ledger

<!-- AUTO-EXPERIMENT-LOG:BEGIN -->
- No auto-recorded experiment runs yet. Run `tools/train.py` or `tools/test.py` to populate this section.
<!-- AUTO-EXPERIMENT-LOG:END -->

## 1. 研究背景与问题定义

本阶段工作的出发点，是参考 MOS 在**单激光雷达**场景下表现较好的 test-time adaptation（TTA）/ self-training 思路，尝试将其迁移到 **OpenPCDet + BEVFusion** 的**多模态夜间场景**中，希望在 night domain 上继续提升检测性能，重点指标为 **NDS** 和 **mAP**。

初始判断很直接：

* MOS 在单 LiDAR 任务上对 domain shift 处理得较好；
* BEVFusion 在夜间场景下虽然已有较强 baseline，但仍存在明显域偏移；
* 如果能把 MOS 的 memory ensemble、伪标签自训练、在线模型聚合等机制改造成**多模态可用版本**，理论上有机会进一步超过 baseline。

但实际实验表明，这件事比预期更难：

* **BEVFusion 的 night baseline 本身已经接近 41% NDS**；
* 直接把 MOS 迁移到多模态后，实验长期**难以稳定超过基准**；
* 真正的瓶颈并不只是“有没有加 TTA”，而是：

  1. 夜间域偏移下伪标签噪声是否被放大；
  2. 多模态结构下 aggregation / self-training 机制是否仍然成立；
  3. 图像链路在 night domain 中是否比 LiDAR 链路更脆弱。

这也是后面整条实验线逐步从“直接套 MOS”演化为“伪标签质量控制 → 冻结策略 → 深度感知过滤 → aggregation 恢复定位”的原因。 fileciteturn2file0

---

## 2. 整体实验路线演化

整个项目并不是一开始就大改模型，而是按“**先稳定基线，再逐步定位主矛盾**”的思路推进：

1. 先确认当前最稳的可复现实验基线；
2. 借助 D1 / D2 类别分布、per-class AP 等统计判断问题主要来自哪里；
3. 优先做轻量改动，先管控伪标签质量，而不是直接大改 detector 主体；
4. 如果轻量伪标签策略无效，再转向结构级别的方法；
5. 当主链结果仍恢复不到历史最好时，再进一步做代码语义回归和 aggregation 恢复定位。

这条路线背后的核心思想是：

* **先控制噪声，再谈适配能力**；
* **先保证训练语义正确，再谈方法创新**；
* **先找到稳定可交付版本，再决定是否进入下一代方法**。 fileciteturn2file0

---

## 3. 问题诊断：night TTA 的核心矛盾是什么

从多轮实验与类别统计中，逐步形成了一个非常明确的诊断：

### 3.1 主类不是主要矛盾

在当前 BEVFusion 主体下，`car`、`truck` 等主类本身已经比较强，说明 backbone / detector 主体并没有整体失效。

### 3.2 真正困难来自高噪伪标签类别

真正的问题集中在伪标签质量，尤其是几个夜间高噪类别：

* `pedestrian`
* `traffic_cone`
* `construction_vehicle`

其中，`pedestrian` 和 `traffic_cone` 在预测分布里长期占比偏高，并且和 GT 分布不匹配，说明这些类别的伪标签里存在明显噪声堆积。如果过滤太弱，噪声会持续污染训练；如果过滤太强，又会误伤有效监督信号。 fileciteturn2file0

### 3.3 因此主问题不是“有没有 TTA”，而是“伪标签能不能被可信地利用”

这也是为什么最早的一整轮实验都集中在：

* 伪标签限流
* 按类过滤
* 自适应 cap
* 深度不确定性过滤

而不是直接大规模重写检测头或 backbone。

---

## 4. 第一阶段：围绕伪标签质量控制的实验

### 4.1 A8：稳定最优参考基线

A8 是这条线最早建立起来的**稳定最优参考版本**。它的特点不是做了很多激进过滤，而是维持了一个相对稳妥的 MOS + memory ensemble 自训练框架：

* 保持 MOS 主体逻辑；
* 使用较稳的 self-training 权重；
* `NUM_PROPOSALS = 200`；
* `TAR.LOSS_WEIGHT = 0.3`；
* 不额外加入过于 aggressive 的伪标签限流逻辑。

A8 的结果是：

* **NDS = 0.4073**

这使得 A8 成为此后所有实验的对照基线。换句话说，后面任何改动，如果不能稳定超过 A8，就不应被视为更优解。 fileciteturn2file0

### 4.2 A11：全类 Top-K 伪标签过滤

A11 的动机来自一个直观观察：既然 `pedestrian` / `traffic_cone` 预测量明显偏高，那么是不是可以对伪标签按类别做 Top-K，只保留每类最靠前的一部分框，从而抑制噪声？

对应实现为：

* 在 `pcdet/tta_methods/mos.py` 中新增 `_apply_class_topk_filter(...)`；
* 在 `save_pseudo_label_batch(...)` 后增加前置过滤；
* 在 YAML 中加入 `SELF_TRAIN.PS_FILTER.ENABLED`、`DEFAULT_TOPK_PER_CLASS`、`CLASS_TOPK` 等配置。

A11 的实验结果：

* **NDS = 0.4059**

比 A8 更差。

这说明：

* “全类统一限流”虽然能抑制一部分噪声，但也会误伤有效伪标签；
* 主类和中等置信的有用框也可能被裁掉；
* 最终并没有转化成整体收益。

A11 的意义不在于结果变好，而在于它明确验证了：**粗粒度的统一过滤不是最优方向**。 fileciteturn2file0

### 4.3 A12：只对高噪类做自适应限流

既然 A11 的问题是“全类一起限流太粗暴”，A12 就把策略收缩到更精细的版本：

* 只对高噪类做约束；
* 主要针对 `pedestrian` 和 `traffic_cone`；
* 对高置信框适度放宽，尽量保住真正有价值的监督信号。

实现方式为：

* 在 `pcdet/tta_methods/mos.py` 中加入 `_apply_adaptive_noisy_class_cap(...)`；
* 用 `TARGET_CLASSES`、`BASE_TOPK`、`HIGH_SCORE_TOPK`、`HIGH_SCORE_THRESH` 控制 noisy class 的保留策略；
* 同时关闭 A11 的全类 `PS_FILTER`。

A12 的结果：

* **NDS = 0.4072**

几乎与 A8 打平，只差 0.0001。

这说明：

* 只限制高噪类、而不动主类，是明显更合理的方向；
* 这种方法比 A11 更细，副作用也更小；
* 但在当前配置下，还不足以稳定超过 A8。

A12 的一个重要价值是：它第一次让我们看到“定向约束 noisy class”确实比“全类统一过滤”更接近正确答案。 fileciteturn2file0

### 4.4 A13：对 A12 的轻微放宽

A13 是在 A12 基础上的一轮小幅参数放松，思路是：

* 既然 A12 已经很接近 A8；
* 那么稍微放宽 high-noise 类的保留条件，也许能多保住一些真正的好框。

具体上：

* `BASE_TOPK: 20 -> 24`
* `HIGH_SCORE_THRESH: 0.45 -> 0.40`
* `HIGH_SCORE_TOPK: 35` 保持不变

结果却是：

* **NDS = 0.4058**

明显退化。

A13 的意义很重要：它说明 A12 已经接近一个局部最优，再放松 noisy class 的保留策略，会重新把更多噪声引回来，最终拖累整体性能。也就是说，靠继续调这类阈值，很难再从 0.407 左右明显往上推。 fileciteturn2file0

---

## 5. 第一阶段结论：真正瓶颈在伪标签噪声，而不是 detector 主体

经过 A8 → A11 → A12 → A13 这一轮，最终形成了几条很稳定的结论：

1. **瓶颈主要来自伪标签噪声，不是 backbone 本身失效**；
2. `pedestrian` 和 `traffic_cone` 是最主要的噪声源；
3. **全类统一过滤不是好办法**；
4. **只针对高噪类做精细控制更合理**；
5. 但在这一阶段里，A8 仍然是更稳的最终交付版本。

因此，当时的收敛策略是：

* 保留实验代码，方便后续继续研究；
* 但训练/评测默认行为回到 A8 等价配置；
* 也就是在 YAML 中关闭 `PS_FILTER.ENABLED` 和 `ADAPTIVE_CAP.ENABLED`。

这代表项目的第一阶段已经完成：**我们知道问题在哪，也知道哪些轻量过滤方法不够强**。 fileciteturn2file0

---

## 6. 第二阶段：结构级改进 —— 冻结策略（Freezing Strategies）

既然单靠伪标签裁剪已经接近瓶颈，下一步就进入结构级思路：

> 在 night TTA 里，最脆弱的可能不是头部，而是图像特征提取链路。

因此提出冻结策略：

* 尽量保护图像 backbone 不被夜间噪声破坏；
* 允许模型只在更靠后的融合层和检测层做适配；
* 避免“全模型一起乱动”导致图像语义进一步漂移。

### 6.1 工程实现

冻结逻辑被加在 `tools/train.py` 中，关键原则是：

1. build model
2. load checkpoint
3. apply freeze strategy
4. rebuild optimizer
5. build scheduler

同时，为了保证冻结真正生效，还做了两个关键工程处理：

* 冻结模块必须强制 `eval()`，避免后续 `model.train()` 把它们重新拉回训练态；
* 冻结后必须重建 optimizer，只保留 `requires_grad=True` 的参数。

### 6.2 F1：冻结 `image_backbone + neck`

第一轮正式生效的冻结策略是：

* **F1 = 冻结 `image_backbone + neck`**

结果：

* **NDS = 0.4077**

这是第一次稳定超过 A8 的结构改动。

日志也验证了冻结不是“名义上打开”，而是真正生效：

* `image_backbone` 冻结约 27.5M 参数；
* `neck` 冻结约 1.59M 参数；
* 最终可训练参数约 `10.69M / 40.80M = 26.20%`；
* optimizer 已按冻结后参数重新构建。

这说明 F1 的提升有明确机制基础：它通过保护图像特征，允许融合与检测层去适应夜间域，而不是让全模型一起被 noisy pseudo label 拖着跑。 fileciteturn2file0

### 6.3 F1.1 与 F2：冻结过少或过多都不理想

后续也做了冻结范围的扩展和对比：

* **F1.1**：只冻结 `image_backbone`

  * 结果：**NDS = 0.4058**
  * 说明只冻 backbone、不冻 neck，不足以稳定住图像链路。

* **F2**：冻结 `image_backbone + neck + vtransform`

  * 结果：**NDS = 0.4065**
  * 说明继续冻到 `vtransform` 又过头了，反而限制了多模态对 night domain 的适配能力。

因此，冻结策略的结论很清楚：

* **F1 是有效的**；
* **F1.1 冻得不够，F2 冻得太过**；
* **`image_backbone + neck` 是当前最合理的冻结边界**。

---

## 7. 第三阶段：基于深度估计的不确定性过滤（Depth-aware Filtering）

在 F1 生效后，实验继续往前推进：

> 既然 BEVFusion 自身就有深度估计，那么能不能利用深度分支的不确定性来进一步清洗伪标签？

这一思路的核心不是只看分类分数，而是判断：

* 某个伪标签是否落在深度预测高度不确定的区域；
* 如果是，则即使它分类分数不低，也可以认为它不够可信。

### 7.1 工程实现

这个方向的最小闭环实现包括两处改动：

1. **`pcdet/models/view_transforms/depth_lss.py`**

   * 暴露 `batch_dict['depth_conf_map'] = depth_prob.max(dim=2)[0]`；
   * 使用深度 softmax 的 `max_prob` 作为深度置信度图。

2. **`pcdet/tta_methods/mos.py`**

   * 新增 `_apply_depth_uncertainty_filter(...)`；
   * 只对 `pedestrian / traffic_cone` 生效；
   * 采用“框中心 + 多相机 max depth_conf + 阈值过滤”的最小实现；
   * 插入在 `save_pseudo_label_batch(...)` 中，位于 memory ensemble 之前。 fileciteturn2file0

### 7.2 D1 ~ D1.3 的实验结果

在 F1 基础上，逐步推进深度过滤：

* **D1**：在 F1 上加入 depth-aware filtering（`pedestrian / traffic_cone`，中心点，多相机 `max`，`max_prob`）

  * **NDS = 0.4074**
  * 方向正确，但没超过 F1。

* **D1.1**：统一把深度阈值提到 `0.30`

  * **NDS = 0.4067**
  * 说明全局统一提阈值过于粗暴，会误伤有效伪标签。

* **D1.2**：按类设置阈值

  * `pedestrian = 0.24`
  * `traffic_cone = 0.32`
  * **NDS = 0.4076**
  * 基本追平 F1。

* **D1.3**：进一步把 `traffic_cone` 提到 `0.34`

  * **NDS = 0.4077**
  * 与 F1 在 NDS 上打平；
  * 但 **mAP 更高**（`0.3532` vs `0.3517`）。

### 7.3 这一阶段的最终结论

这一轮的结论非常明确：

* depth-aware filtering 的方向是对的；
* 但不能用统一阈值；
* 按类阈值控制比全局阈值更合理；
* 最终最优综合方案不是“只 freeze”也不是“只 depth filter”，而是：

> **F1 + D1.3**

也就是：

* 冻结 `image_backbone + neck`
* 深度过滤只针对 `pedestrian / traffic_cone`
* 阈值设为：

  * `pedestrian = 0.24`
  * `traffic_cone = 0.34`

这个组合达到了：

* **NDS = 0.4077**
* **mAP = 0.3532**

在当时是最优综合解。它说明：**伪标签质量控制与结构稳定性结合，确实可以把 BEVFusion 的 night TTA 推到比 A8 更高的位置。** fileciteturn2file0

---

## 8. 第四阶段：进入恢复线 —— 为什么仍然回不到历史最好结果

虽然历史上已经得到过 **F1 + D1.3 = 0.4077** 这条强线，但后续在代码恢复和分支回归时，发现当前实现始终回不到这个结果，于是实验目标从“继续调参”变成了：

> 弄清楚为什么当前分支恢复不到历史最好 0.4077。

这一阶段不再是新方法搜索，而是**代码语义回归与主链修复**。

### 8.1 已定位并修复的主因

重点修复集中在 `pcdet/tta_methods/mos.py`：

1. **`_inject_pseudo_labels(...)` 的伪标签解析错误**

   * 9 维 raw pseudo 语义被错读；
   * ignore 负标签和 padding 0 被误裁成正类；
   * 已统一修成正确的 10 维输出，并与 `TransFusionHead` 的 `valid_idx` 语义对齐。

2. **训练语义缺失**

   * `loss.mean()` 之前没补齐；
   * `SELF_TRAIN.TAR.LOSS_WEIGHT` 之前没有真正乘上；
   * `update_global_step()` 被绕过；
   * 这些都已经补回。

3. **side-channel 与增强同步问题**

   * `proto_pseudo_scores` 已加入并和 TTA augmentation 对齐；
   * `pcdet/utils/tta_utils.py` 也已支持 10 维 `gt_boxes` 和 side-channel 同步。

这些修复说明：之前并不是“方法本身彻底无效”，而是代码语义已经偏离了原始有效版本。

### 8.2 修复后的恢复结果

修复之后，主线结果有明显回升，但仍低于历史最优：

* 完整主链修复后：**NDS = 0.3714**

这说明：

* 伪标签注入和训练语义问题确实是重要回归来源；
* 但它们并不是唯一原因；
* 后续还需要进一步检查 aggregation 这条线。

---

## 9. 第五阶段：aggregation 恢复与多模态适配定位

在主链修复后，另一个高嫌疑模块就是 MOS 的 aggregation。因为 MOS 的原设计来自**单 LiDAR**场景，而当前目标是 **BEVFusion 多模态**，二者在模型表征、特征来源、时序保存方式上并不完全一致。

因此，这一阶段的核心问题变成：

* 问题到底出在 ckpt bank 的保存/读取时序？
* 还是 aggregation 的数学逻辑本身不适配多模态？

### 9.1 对照实验结果

为了回答这个问题，做了几组关键定位实验：

* **no-aggregation**：

  * 结果：**NDS = 0.3552**
  * 说明 aggregation 不是完全无效的，关掉后更差。

* **agg_path_fix_v2**：

  * 修复当前 run ckpt 目录、iter ckpt 保存、aggregation 运行 bug；
  * 结果：**NDS = 0.3688**
  * 说明 aggregation 确实能带来一些正收益。

* **P2-min**：

  * 严格对齐 MOS-main 的 `samples_seen + checkpoint_iter_* only` 语义；
  * 结果：**NDS = 0.3579**
  * 这是负向验证，说明“更像 MOS-main”并不等于“更适合 BEVFusion”。

* **P2-full**：

  * 在 P2-min 基础上，把 aggregation 数学逻辑改成多模态版；
  * 引入融合特征向量、prediction signature、box consistency；
  * 结果：**NDS = 0.3604**
  * 比 P2-min 好，但仍不如 `agg_path_fix_v2 = 0.3688`。

### 9.2 这一阶段的核心认识

这些结果给出一个非常关键的判断：

1. aggregation **本身不是错误方向**，因为 `0.3688 > 0.3552`；
2. 但 **严格照搬 MOS-main 的单 LiDAR timing / iter-only bank 是负向的**；
3. 多模态 aggregation 数学改写本身有一定价值，但不是决定性收益来源；
4. 对多模态场景来说，MOS 的流程骨架可以借，但其**单 LiDAR aggregation 细节不能完全照搬**。

因此，下一步提出了 **Hybrid H1**：

* 保留 P2-full 的多模态 aggregation 数学逻辑；
* 但撤回 P2-min 里过于严格的 iter-only bank 约束；
* 改成只读当前 run，但允许 `checkpoint_iter_* + checkpoint_epoch_*` 混合 bank。

这一阶段的意义在于：它把“为什么恢复不到 0.4077”这个问题，从泛泛的猜测，压缩成了**清晰的工程定位问题**。

---

## 10. 第六阶段：自动评估 best iter 与 early-iter 现象

在后续实验推进中，又出现了一个很重要的现象：

> 某些最优结果并不出现在训练后期，而是出现在更早的 iter checkpoint。

因此又做了一个工程增强：

* 在 `tools/test.py` 中支持按 `iter` 或 `epoch` 扫描 checkpoint；
* 在 `tools/train.py` 中训练结束后自动调用 `repeat_eval_ckpt(...)`；
* 再自动汇总生成 best ckpt summary。

这样做的目的是：

* 避免每次手动找最优 iter；
* 系统化观察 early-iter 是否就是当前设置下真正的最优点。

在这条支线里，S1 的 early-iter 实验曾出现过：

* **best ckpt = `checkpoint_iter_10`**
* **NDS = 0.4116**
* **mAP = 0.3580**

这说明两个重要事实：

1. 在某些配置下，TTA 确实有能力**在单次最优点上超过约 41% 的 baseline**；
2. 但这类收益具有明显的 early-iter 特征，说明当前训练后期可能仍然存在过适配、噪声累积或 bank 策略不稳的问题。

所以，如果从“有没有出现过超过 baseline 的结果”来看，答案已经不是绝对否定；但如果从“是否已经得到稳定、可复现、训练全程都稳的最终方案”来看，仍然还有不少工作没收敛。

---

## 11. 全部阶段的总结果回顾

如果把整个实验过程串起来，可以概括为下面这条主线：

### 11.1 伪标签控制阶段

* A8：**0.4073**（稳定强基线）
* A11：**0.4059**（全类 Top-K，负向）
* A12：**0.4072**（只限高噪类，接近打平）
* A13：**0.4058**（继续放宽，负向）

### 11.2 结构与深度过滤阶段

* F1：**0.4077**（冻结 `image_backbone + neck`，首次超过 A8）
* F1.1：**0.4058**（只冻 backbone，退化）
* F2：**0.4065**（冻到 `vtransform`，过强）
* D1：**0.4074**
* D1.1：**0.4067**
* D1.2：**0.4076**
* D1.3：**0.4077**（与 F1 打平，但 mAP 更高）

### 11.3 恢复与 aggregation 定位阶段

* 主链修复后：**0.3714**
* no-aggregation：**0.3552**
* agg_path_fix_v2：**0.3688**
* P2-min：**0.3579**
* P2-full：**0.3604**
* H1：待验证

### 11.4 自动评估与 early-iter 支线

* S1 最优 early iter：

  * `checkpoint_iter_10`
  * **NDS = 0.4116**
  * **mAP = 0.3580**

这几组结果合在一起，实际上说明了项目已经取得了三种不同层面的成果：

* **方法层面**：冻结 + 深度过滤确实有效；
* **工程层面**：伪标签注入、loss 权重、global step、ckpt 读取等关键语义已经被补齐；
* **流程层面**：训练后自动评估 best iter 的能力已落地。

---

## 12. 当前阶段的统一结论

把整段实验放在一起看，目前最合理的统一结论是：

### 12.1 MOS 思想对多模态是有启发的，但不能直接照搬

MOS 在单 LiDAR 上有效，不代表直接迁移到 BEVFusion 多模态也会自然成立。特别是：

* aggregation 数学逻辑；
* ckpt bank 的 timing / iter-only 机制；
* 多模态特征在 night domain 下的脆弱性；

这些都要求重新设计，而不是简单复制。

### 12.2 当前最稳定、最可信的改进来自两条路线

目前已经被验证有效的两条路线是：

1. **冻结策略（Freezing Strategies）**
2. **基于深度估计的不确定性过滤（Depth-aware Filtering）**

其中最优综合方案是：

> **F1 + D1.3**

即：

* 冻结 `image_backbone + neck`
* 对 `pedestrian / traffic_cone` 施加按类深度阈值过滤
* `pedestrian = 0.24`
* `traffic_cone = 0.34`

该方案达到：

* **NDS = 0.4077**
* **mAP = 0.3532**

这是当前最清晰、最有机制解释、也最值得作为主方法结论保留的一条线。 fileciteturn2file0

### 12.3 如果继续沿 aggregation 线走，H1 是最合理的下一步

当前 aggregation 线最合理的下一步不是继续死抠 MOS-main 的单 LiDAR 对齐，而是：

* 保留 P2-full 的多模态 aggregation 数学逻辑；
* 恢复当前 run 的 `iter + epoch` 混合 bank；
* 验证 Hybrid H1 能否超过 `0.3688`。

如果 H1 仍然不能超过 `0.3688`，那就说明 aggregation 恢复这条线已经接近阶段性上限，应考虑减少投入。

### 12.4 再往后，更值得进入“下一代方法”而不是继续微调

如果后续目标仍然是继续往上推，最合理的下一阶段方法不是再在 D1 阈值上做小修小补，而是进入新的范式，例如：

* **Prototype Alignment**

而且建议从轻量版 P1 开始：

* 以 **F1 + D1.3** 为 baseline；
* 不动现有 D1 主逻辑；
* 仅新增 prototype alignment loss；
* 优先只对高噪类 / 长尾类做：

  * `pedestrian`
  * `traffic_cone`
  * 后续再考虑 `construction_vehicle`

---

## 13. 最终总结

这段实验最重要的收获，不只是“哪个数字更高”，而是逐步建立起了对 **BEVFusion Night TTA** 的完整认识：

1. **直接把单 LiDAR MOS 搬到多模态，不会自动成功**；
2. **night TTA 的核心矛盾是伪标签噪声，尤其是 `pedestrian / traffic_cone`**；
3. **粗糙的全类过滤无效，按类精细控制更合理**；
4. **冻结图像链路是有效的结构级改动**；
5. **深度不确定性过滤是适合 BEVFusion 的多模态信息利用方式**；
6. **训练语义和 aggregation 读取语义的正确性，和方法本身同样重要**；
7. **自动找 best iter 后，已经观察到 early-iter 可以超过 baseline，但还需要把这种收益变成稳定结论。**

因此，到目前为止，这条工作线可以概括为：

> 你是从 MOS 文献出发，希望把单 LiDAR 上有效的 TTA/self-training 机制迁移到多模态 BEVFusion；
> 过程中发现 night 场景下真正的主矛盾是高噪伪标签与多模态不匹配；
> 最终验证出 **F1 + D1.3** 是当前最稳的主线改进，且 aggregation 恢复与下一代 prototype 方法仍有继续探索空间。

