# Ablation 实验方案

## 基线（Baseline）

| # | 配置 | 说明 |
|---|------|------|
| B0 | `ssc_loss_type: "ce"`, `class_weight_mode: "uniform"` | 原始方案，所有非空类权重 = 1.0 |

## 实验组

### 实验 1: 频率加权

| # | ssc_loss_type | class_weight_mode | focal_gamma | dice_smooth | 预期效果 |
|---|---------------|-------------------|-------------|-------------|---------|
| A1 | ce | **frequency** | 2.0 | 1.0 | floor/wall 权重降低，wire/shoe/pet 权重升高 → **稀有类 IoU↑** |
| A2 | ce+dice | **frequency** | 2.0 | 1.0 | 频率权重 + Dice 双重抗不平衡 → **综合最优** |

### 实验 2: Loss 类型

| # | ssc_loss_type | class_weight_mode | focal_gamma | dice_smooth | 预期效果 |
|---|---------------|-------------------|-------------|-------------|---------|
| B1 | ce | uniform | 2.0 | 1.0 | 纯 CE 基线（同 B0） |
| B2 | **focal** | uniform | 2.0 | 1.0 | 降权易分类样本，稀有类关注度↑ |
| B3 | **dice** | uniform | 2.0 | 1.0 | 直接优化 IoU，各类更均衡 |
| B4 | **ce+dice** | uniform | 2.0 | 1.0 | CE 全局梯度 + Dice 逐类约束 |
| B5 | **focal+dice** | uniform | 2.0 | 1.0 | Focal 降权易分类 + Dice 逐类优化 |

### 实验 3: Focal γ 消融

| # | ssc_loss_type | class_weight_mode | focal_gamma | 预期效果 |
|---|---------------|-------------------|-------------|---------|
| C1 | focal | uniform | **0.5** | 弱聚焦，接近 CE |
| C2 | focal | uniform | **2.0** | 标准设定（推荐） |
| C3 | focal | uniform | **3.0** | 强聚焦，极稀有类收益但易分类可能下降 |
| C4 | focal | uniform | **5.0** | 极端聚焦，可能训练不稳定 |

### 实验 4: Dice smooth 消融

| # | ssc_loss_type | class_weight_mode | dice_smooth | 预期效果 |
|---|---------------|-------------------|-------------|---------|
| D1 | dice | uniform | **0.1** | 小 smooth，梯度更尖锐，但对零体素类敏感 |
| D2 | dice | uniform | **1.0** | 标准设定（推荐） |
| D3 | dice | uniform | **10.0** | 大 smooth，训练稳定但梯度弱 |

## 推荐执行顺序

```
B0 (基线确认)
  → A1 (频率权重, 最小改动看收益)
    → A2 (频率 + Dice)
      → B4 (ce+dice, uniform) 对比 A2 看频率权重单独贡献
```

如果 A1 有明显提升，说明频率权重有效，可以做更细的 A2。
如果 B4 比 A1 好，说明 Dice 贡献更大。

## 指标关注点

| 指标 | 关注原因 |
|------|---------|
| **mIoU** | 整体指标，必须不降 |
| **class IoU 标准差** | 各类别之间的方差——降低说明更均衡 |
| **稀有类 IoU（wire/shoe/pet/plant/building_blocks）** | 长尾改进的核心指标 |
| **常见类 IoU（floor/wall）** | 确保不因权重调整而下降 |
| **val/IoU（占据 IoU）** | 占据检测是否受影响 |

## 实验记录模板

每次实验后记录：

```yaml
# 实验编号: A1
# 启动时间: 2026-06-15
# 配置差异: class_weight_mode: frequency
# mIoU: xx.xx (+x.xx vs B0)
# 稀有类 mIoU: xx.xx (+x.xx vs B0)
# 注意:
```
