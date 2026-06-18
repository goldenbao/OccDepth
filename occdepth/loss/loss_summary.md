# OccDepth Loss 方案总结

## 总览

OccDepth 的损失函数由多个互补的 loss 组成，定义在 `occdepth/models/OccDepth.py` 的 `step()` 方法中（第 820-975 行）。每个 loss 通过 YAML 配置中的 bool 开关独立启用，最终加权求和。

```
loss = CE_ssc_loss
     + sem_scal_loss
     + geo_scal_loss
     + fp_loss (frustum KL)
     + relation_loss (CRP multi-label BCE)
     + depth_loss (flosp_depth GT 或 igev_rr student 蒸馏)
     + [cascade_cls 时的 occ CE loss]
     + [occluded_cls 时的 occluded CE loss]
```

---

## 1. CE_ssc_loss — 语义交叉熵

**文件：** `occdepth/loss/ssc_loss.py`

**公式：**
```
CE_ssc_loss = CrossEntropyLoss(pred, target, weight=class_weights, ignore_index=255)
```

**说明：**
- 标准的交叉熵，作用于 3D 体素网格的逐体素语义分类
- 使用 `class_weights` 缓解类别不平衡（高频类别权重低，稀有类别权重高）
- `ignore_index=255` 忽略未知体素（不参与 loss 计算）

**class_weights 来源（`train.py` 第 77-78 行）：**
- **Sweeper：** `[0.05, 1, 1, ..., 1]`（24 类，free 类权重 0.05 降低空体素影响）
- **KITTI：** `1 / log(class_frequencies + 0.001)`（基于频率的逆权重）
- **NYU：** `[0.05, 1, 1, ..., 1]`（12 类）
- **TartanAir：** 同 NYU

**配置开关：** `CE_ssc_loss: true/false`

---

## 1b. Focal Loss — 聚焦损失

**文件：** `occdepth/loss/ssc_loss.py`

**公式：**
```
FL(p_t) = -α_t * (1-p_t)^γ * log(p_t)
```

**说明：**
- 在 CE 基础上乘以 `(1-p_t)^γ`，自动降权易分类样本（p_t 接近 1 的样本），让模型更关注难分类/稀有类
- `class_weights` 作为 α_t 复用
- `γ=0` 时退化为普通 CE；`γ=2` 是标准设定
- 适用于长尾分布中稀有类难以学习的问题

**配置开关：** `ssc_loss_type: "focal"`, `focal_gamma: 2.0`

---

## 1c. Dice Loss — Dice 损失

**文件：** `occdepth/loss/ssc_loss.py`

**公式：**
```
Dice = (2 * |A∩B| + smooth) / (|A|+|B| + smooth)
DiceLoss = 1 - weighted_mean(Dice)
```

**说明：**
- 逐类计算 Dice 系数（预测概率 vs one-hot 标签），用 class_weights 加权平均
- 直接优化 IoU，天然处理类间不平衡（因为是比率形式，不受绝对体素数影响）
- `smooth` 参数防止除零（默认 1.0）
- 适用于分割/占据预测中类别极度不平衡的场景

**配置开关：** `ssc_loss_type: "dice"`, `dice_smooth: 1.0`

---

## 1d. 组合模式

支持 CE 与 Dice 或 Focal 与 Dice 组合：

| ssc_loss_type | CE | Focal | Dice |
|---|---|---|---|
| "ce" | ✓ | | |
| "focal" | | ✓ | |
| "dice" | | | ✓ |
| "ce+dice" | ✓ | | ✓ |
| "focal+dice" | | ✓ | ✓ |

组合模式下两个 loss 的 logits/梯度直接相加，无需额外权重。

**配置开关：** `ssc_loss_type: "ce+dice"` 或 `"focal+dice"`

---

## 2. Cascade Occ CE Loss — 级联占据二分类

**文件：** `occdepth/loss/ssc_loss.py`（复用 CE_ssc_loss）

**触发条件：** `cascade_cls: true` 时启用

**公式：**
```
target_occ = (target != 0) & (target != 255)  # 占据=1, 空闲=0
loss_occ = CrossEntropyLoss(occ_pred, target_occ, weight=class_weights_occ)
```

**说明：**
- 先做二分类占据预测，再做细粒度语义分类
- `class_weights_occ` 通常为 `[0.05, 2]`（空闲类低权重，占据类高权重）

**配置开关：** `cascade_cls: true/false`

---

## 3. Occluded Cls CE Loss — 遮挡分类

**文件：** `occdepth/loss/ssc_loss.py`（复用 CE_ssc_loss）

**触发条件：** `occluded_cls: true` 且 batch 中包含 `occluded` 标签

**公式：**
```
loss_occluded = CrossEntropyLoss(occluded_pred, target_occluded, weight=[1, 1])
```

**说明：**
- 二分类：每个体素是否被遮挡
- 权重均等 `[1, 1]`，不偏向任何一类

**配置开关：** `occluded_cls: true/false`

---

## 4. Geo_scal_loss — 几何缩放损失

**文件：** `occdepth/loss/ssc_loss.py` 第 16-40 行

**目标：** 提升占据/空闲二分类的几何质量

**公式：**
```
empty_probs   = softmax(pred)[:, class_0]      # 空闲概率
nonempty_probs = 1 - empty_probs                # 占据概率
nonempty_target = (target != 0)                 # 占据 GT

precision = (nonempty_target * nonempty_probs).sum() / nonempty_probs.sum()
recall    = (nonempty_target * nonempty_probs).sum() / nonempty_target.sum()
specificity = ((1-nonempty_target) * empty_probs).sum() / (1-nonempty_target).sum()

loss = BCE(precision, 1) + BCE(recall, 1) + BCE(specificity, 1)
```

**说明：**
- 将每个样本的 precision / recall / specificity 作为标量推到 1.0
- 本质是最大化 IoU 的代理 loss
- 忽略 `target=255` 的体素

**配置开关：** `geo_scal_loss: true/false`

---

## 5. Sem_scal_loss — 语义缩放损失

**文件：** `occdepth/loss/ssc_loss.py` 第 43-87 行

**目标：** 提升每个语义类别的 precision / recall / specificity

**公式（对每个类别 i 循环）：**
```
completion_target = (target == i)  # 属于该类=1, 否则=0

precision   = sum(p * completion_target) / sum(p)
recall      = sum(p * completion_target) / sum(completion_target)
specificity = sum((1-p) * (1-completion_target)) / sum(1-completion_target)

loss_i = BCE(precision, 1) + BCE(recall, 1) + BCE(specificity, 1)
```

**说明：**
- 对所有类别 i 求平均：`loss = mean(loss_i)`
- 支持步长衰减（`sem_step_decay_loss: true` 时，从 1.0 线性衰减到 0.1）
- 步长衰减的意义：早期用 sem_scal_loss 引导收敛，后期减少干扰让 CE 主导

**配置开关：** `sem_scal_loss: true/false`, `sem_step_decay_loss: true/false`

---

## 6. FP_loss — Frustum KL 散度损失

**文件：** `occdepth/loss/ssc_loss.py` 第 6-13 行（KL_sep）

**目标：** 让每个 Frustum（视锥体）内的类别分布与 GT 分布一致

**公式：**
```
for each frustum f:
    prob = pred_prob * frustum_mask          # 该 frustum 内的预测概率
    cum_prob = prob.sum() / total_prob       # 各类别的归一化比例
    target_proportion = batch_cnt[f] / total_cnt  # GT 类别比例
    
    frustum_loss += KL_div(cum_prob, target_proportion)
```

**说明：**
- 每个 frustum 是从相机出发的一个锥体区域
- KL 散度约束 frustum 内预测的类别比例与 GT 一致
- 仅非空的 frustum 参与 loss 计算
- 仅在训练时启用（`step_type != "test"`），推理时不需要 frustum mask

**配置开关：** `fp_loss: true/false`

---

## 7. Relation_loss — CRP 多标签 BCE 损失

**文件：** `occdepth/loss/CRP_loss.py`

**目标：** 约束 Context Relation Prior 模块的 relation 预测

**公式：**
```
pos_weight = max(1.0, clamp(count_neg / count_pos, max=2000))
loss = BCEWithLogitsLoss(pred_logits, CP_mega_matrices, pos_weight=pos_weight)
```

**说明：**
- CRP 模块预测类别间的共现关系（4 种关系 × N 对 × M 个 mega-voxel）
- 正负样本极度不平衡，使用 `pos_weight` 调节（负样本数 / 正样本数，上限 2000）
- 这是多标签分类任务，一个 pair 可以有多个关系同时成立

**配置开关：** `relation_loss: true/false`（需要 `context_prior: true` 才会有 CRP 输出）

---

## 8. Depth_loss — 深度分类损失

**文件：** `occdepth/loss/depth_loss.py`

**目标：** 监督深度分布预测

### 8a. FlospDepth + GT Depth 模式

**触发条件：** `trans_2d_to_3d="flosp_depth"` 且 `use_stereo_depth_gt / use_lidar_depth_gt / use_depth_gt=true`

**流程：**
1. GT depth 双线性插值下采样到 1/8 分辨率
2. 窗口内取最小值（避免空洞）
3. 映射到 80-bin LID 编码 + one-hot
4. 与深度预测的 80-bin 分布算 BCE

```
depth_labels = downsample + one_hot(gt_depth → 80 bins)
fg_mask = max(depth_labels) > 0  # 前景区域
loss = BCE(depth_pred[fg_mask], depth_labels[fg_mask]).sum() / max(1, fg_mask.sum())
```

**配置开关：** `depth_loss_weight: 0.1`（loss 乘以此系数加权）

### 8b. IGEV-RR Student 蒸馏模式

**触发条件：** `trans_2d_to_3d="igev_rr_depth"` 且 `use_igev_student=true`

**流程：**
1. 冻结的 IGEV-RR teacher 生成稠密深度伪标签 `depth_teacher`
2. FlospDepth student 预测深度分布 `depth_pred_student`
3. 使用与 8a 完全相同的 `DepthClsLoss.get_depth_loss()` 计算 BCE

```
loss_depth = DepthClsLoss(depth_teacher, depth_pred_student) * depth_loss_weight
```

**说明：**
- 不需要 GT depth，teacher 提供伪标签
- student 输出双视角深度，取 `depth_pred[:, 0:1]`（左目）匹配 teacher 单目输出
- 融合推理：`x3ds_depth_fused = alpha * teacher + (1-alpha) * student`

**配置开关：** `use_igev_student: true`, `igev_student_fuse_alpha: 0.7`, `depth_loss_weight: 0.0`

---

## Loss 启用配置示例

### flosp_depth 模式（有 GT depth）

```yaml
CE_ssc_loss: true
sem_scal_loss: true
geo_scal_loss: true
fp_loss: true
relation_loss: true
depth_loss_weight: 0.1
use_stereo_depth_gt: true    # 启用 flosp_depth 的 depth loss
```

### igev_rr_depth 模式（无 GT depth）

```yaml
CE_ssc_loss: true
sem_scal_loss: true
geo_scal_loss: true
fp_loss: true
relation_loss: true
depth_loss_weight: 0.0       # 不需要 GT depth loss
use_stereo_depth_gt: false
use_igev_student: true       # 启用 teacher-student 蒸馏 depth loss
igev_student_fuse_alpha: 0.7
```

---

## 新增 Config 配置项

```yaml
# 主 SSC loss 类型和 class weight 模式
ssc_loss_type: "ce"           # "ce" | "focal" | "dice" | "ce+dice" | "focal+dice"
class_weight_mode: "uniform"   # "uniform" | "frequency"
focal_gamma: 2.0
dice_smooth: 1.0
```

- `class_weight_mode="frequency"`：对 sweeper 使用 `1/log(freq+ε)` 频率逆权重（KITTI 方案），替代硬编码的 uniform 权重
- 两个模式默认值均为向后兼容，不改动配置则行为完全不变

## Loss 组合总览

| Loss | 作用对象 | 类型 | 权重来源 |
|------|---------|------|---------|
| CE_ssc_loss | 语义分类 | 加权 CE | class_weights（数据集相关） |
| FocalLoss | 语义分类 | 加权 Focal CE | class_weights + (1-p_t)^γ |
| DiceLoss | 语义分类 | Dice (IoU proxy) | class_weights 逐类加权 |
| Occ CE Loss | 占据二分类 | 加权 CE | class_weights_occ = [0.05, 2] |
| Occluded CE | 遮挡二分类 | CE | [1, 1] 均等 |
| geo_scal_loss | 占据几何质量 | BCE on PRS | 无，直接推到 1.0 |
| sem_scal_loss | 语义 precision/recall | BCE on PRS | 无，逐类推到 1.0 |
| fp_loss | Frustum 内分布 | KL 散度 | 无，GT 分布比例 |
| relation_loss | CRP 共现关系 | 加权 BCE | pos_weight = neg/pos count |
| depth_loss (GT) | 深度分布 | BCE | depth_loss_weight |
| depth_loss (student) | 深度分布蒸馏 | BCE | depth_loss_weight |
