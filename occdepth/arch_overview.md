# OccDepth 架构概览

> 基于代码分析的深度感知 3D 语义场景补全（SSC）框架架构文档。

---

## 整体数据流

```
RGB图像 (bs, n_views, 3, H, W)
        │
        ▼
┌──────────────────┐
│    UNet2D        │  EfficientNet-B3 编码器 + 轻量解码器
│ (unet2d.py:199)  │  输出多尺度 2D 特征 {1_1, 1_2, 1_4, 1_8, 1_16}
└────────┬─────────┘
         │
         ├──────────────────────┐
         ▼                      ▼
┌──────────────────┐  ┌──────────────────────┐
│  SFA (×4 scales) │  │    FlospDepth         │
│    (SFA.py:5)    │  │ (flosp_depth.py:325)  │
│  几何投影 2D→3D  │  │  深度感知特征提升      │
│  预计算像素索引   │  │  预测深度分布 + 采样   │
│  无学习参数       │  │  到体素空间            │
└────────┬─────────┘  └──────────┬───────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
    融合: x3ds = SFA_features × FlospDepth_features × 100
                     │
                     ▼
┌─────────────────────────┐
│    UNet3D               │  3D 编码器-解码器
│ (unet3d_*.py)           │  Process + Downsample (×2)
│                         │  [可选 CRP 上下文关系模块]
│                         │  Upsample + 跳跃连接
│                         │  SegmentationHead (ASPP + Conv)
└───────────┬─────────────┘
             │
             ▼
     ssc_logit: (bs, n_classes, X, Y, Z)
             │
             ▼
┌─────────────────────────┐
│   Loss Functions        │  CE_ssc_loss + sem_scal_loss
│  (occdepth/loss/)       │  + geo_scal_loss + depth_loss
│                         │  + CRP_loss + frustum_loss (KL)
└─────────────────────────┘
```

---

## 1. 主控模块 — `OccDepth`

**文件**: `occdepth/models/OccDepth.py:31`

整个管线的 `pl.LightningModule`，在 `forward()` 中串联所有子模块：

```python
# OccDepth.py:371-408 核心 forward 流程
def forward(self, batch):
    # 步骤1: UNet2D 提取多尺度 2D 特征
    x_rgb, n_views = self.process_rgbs(img, batch, n_views)

    # 步骤2: SFA + 可选 FlospDepth 投影到 3D 体素空间
    x3ds, depth_pred = self._forward_2d_to_3d(batch, x_rgb, img, bs, vox_origin)

    # 步骤3: UNet3D 解码语义占用
    net_out = self.net_3d_decoder({"x3d": x3ds})
```

### 两种 2D→3D 模式

由配置项 `trans_2d_to_3d` 控制（`OccDepth.py:173-230`）：

- **`"flosp"`**: 纯几何投影，SFA 用预计算像素索引查表，无学习参数
- **`"flosp_depth"`**: 深度感知投影，SFA 几何投影 + FlospDepth 学习的深度注意力加权

### 深度特征融合公式 (`OccDepth.py:363-366`)

```python
# Fuse geometric projection (SFA) with depth-aware attention (FlospDepth).
# x3ds_depth provides a soft per-voxel weight based on learned depth distribution,
# and the factor 100 amplifies the depth signal to match the scale of SFA features.
x3ds = x3ds * x3ds_depth * 100
```

### 训练损失 (`step`, `OccDepth.py:412`)

累加最多 6 种损失：
1. **`CE_ssc_loss`** — 带类别权重的交叉熵，忽略标签 255
2. **`sem_scal_loss`** — 每类 precision/recall/specificity BCE
3. **`geo_scal_loss`** — 二值（空/非空）几何尺度 BCE
4. **`fp_loss`** — 视锥（frustum）KL 散度
5. **`relation_loss`** — CRP 上下文关系多标签 BCE
6. **`depth_loss`** — 深度 bin 分类 BCE

---

## 2. 2D 骨干 — `UNet2D`

**文件**: `occdepth/models/unet2d.py:199`

EfficientNet-B3 编码器 + 轻量解码器，输出 5 个尺度的特征图：

| 尺度 Key | 空间尺寸（KITTI 示例） | 通道数 |
|----------|----------------------|--------|
| `1_1` | 370 × 1220 | 24 |
| `1_2` | 185 × 610 | 24 |
| `1_4` | 92 × 305 | 24 |
| `1_8` | 46 × 152 | 24 |
| `1_16` | 23 × 76 | 24 |

### 架构要点

- 编码器（`Encoder`, `unet2d.py:183`）封装 `tf_efficientnet_b3_ns`，移除全局池化和分类头，以 `blocks` 模块为单位挂钩收集中间特征
- 解码器（`DecoderBN`, `unet2d.py:49`）提取特定层级特征 `[0, 4, 5, 6, 8, 11]`，用 `UpSampleBN` 逐级上采样 + 跳跃连接拼接
- 输出通道数由配置 `feature_2d_oc` 控制（默认 24）

---

## 3. 2D→3D 投影

### 3.1 几何投影 — `SFA`

**文件**: `occdepth/models/SFA.py:5`

**无参数模块**，纯粹用预计算像素索引查表。

#### forward 核心逻辑

```python
# SFA.py:18-113
def forward(self, x2d, projected_pix, fov_mask):
    # 1. 逐视图: 对每个体素中心，在多个投影 pattern 上 gather 2D 特征并平均
    for view in range(n_views):
        src = x2d[view].view(c, -1)
        # 零向量填充 out-of-FOV 体素
        src = torch.cat([src, zeros_vec], 1)
        # 按 projected_pix 坐标 gather 特征 → 多 pattern 平均
        sub_src_feature = gather(src, img_indices[:, :, 0])
        for pattern in patterns:
            sub_src_feature += gather(src, img_indices[:, :, pattern])
        sub_src_feature /= sub_weights

    # 2. 多视图融合: 余弦相似度加权
    for idx_i in range(n_views):
        for idx_j in range(idx_i + 1, n_views):
            weight_ij = fov_mask_i * fov_mask_j        # 共视体素
            weight_diff = fov_mask_i - fov_mask_j       # 单视图体素
            cos_weight = cosine_similarity(feat_i, feat_j) * weight_ij
            # 加权求和: 共视体素由相似度加权，单视图体素权重提升
            sum_weight += cos_weight_i * feat_i + cos_weight_j * feat_j

    # 3. reshape 到 3D 体素网格 (c, X/scale, Y/scale, Z/scale)
```

### 3.2 深度感知投影 — `FlospDepth`

**文件**: `occdepth/models/flosp_depth/flosp_depth.py:325`

学习深度分布，用来加权 SFA 的特征。

#### 子模块

| 子模块 | 类名 | 位置 | 作用 |
|--------|------|------|------|
| DepthNet | `DepthNet` | `flosp_depth.py:202` | 从 2D 特征预测多通道深度概率分布，使用 ASPP-like 空洞卷积 + SE 层融入相机内参条件 |
| PCFE | `PCFE` | `flosp_depth.py:261` | Pixel Cloud Feature Extraction，残差卷积块 |

#### forward 核心逻辑

```python
# flosp_depth.py:325 forward
def forward(self, img_feat, cam_k, T_velo_2_cam, ...):
    # 1. DepthNet 预测深度分布
    depth_feature = self._forward_depth_net(img_feat, intrins_mat, ...)
    depth = depth_feature.softmax(1)    # (bs*n_cams, D_bins, h, w)

    # 2. 对每张相机图，用 frustum grid 采样深度加权特征到体素
    for cam in range(n_cams):
        grid = self.grid_generator(...)     # 生成视锥采样网格
        voxel = sampler(depth[cam] * feat[cam], grid)  # 双线性采样

    # 3. 多相机融合 (mean/sum)
    agg = sum(features) / sum(masks)   # mean 模式
```

---

## 4. 3D 语义补全 — `UNet3D`

三个数据集特定实现，结构类似但体素分辨率不同：

| 文件 | 输入尺寸 → 输出尺寸 | 场景 |
|------|-------------------|------|
| `unet3d_kitti.py:14` | 128×128×16 → 256×256×32 | 自动驾驶 (KITTI) |
| `unet3d_nyu.py:16` | 60×36×60 → 240×144×240 | 室内 (NYU) |
| `unet3d_sweeper.py:13` | 40×40×24 → 80×80×48 | 自定义 |

### KITTI 版结构 (`unet3d_kitti.py:14`)

```
x3d_l1 = input["x3d"]          # (bs, 24, 128, 128, 16)  project_scale=2
    │
    ├── Process(dilation=[1,2,3]) + Downsample(stride=2)
    ▼
x3d_l2                          # (bs, 48, 64, 64, 8)
    │
    ├── Process(dilation=[1,2,3]) + Downsample(stride=2)
    ▼
x3d_l3                          # (bs, 96, 32, 32, 4)
    │
    ├── [可选] CPMegaVoxels (CRP 上下文关系模块)
    ▼
x3d_up_l2 = Upsample(x3d_l3) + x3d_l2    # 跳跃连接
x3d_up_l1 = Upsample(x3d_up_l2) + x3d_l1
x3d_up_lfull = Upsample(x3d_up_l1)       # 恢复到全分辨率
    │
    ├── SegmentationHead (Conv + ASPP + Conv)
    ▼
ssc_logit: (bs, 20, 256, 256, 32)        # 20 类语义占用
```

### 共享 3D 构建模块 (`occdepth/models/modules.py`)

| 组件 | 位置 | 作用 |
|------|------|------|
| `Process` | `modules.py:258` | 3 连续 Bottleneck3D，扩张率 [1,2,3]，逐步扩大感受野 |
| `Downsample` | `modules.py:320` | Bottleneck3D + stride=2 + AvgPool+Conv 跳跃连接，通道翻倍 |
| `Upsample` | `modules.py:278` | ConvTranspose3d stride=2 |
| `Convblock3d` | `modules.py:299` | ConvTranspose3d stride=1 |
| `SegmentationHead` | `modules.py:51` | Conv3d → ASPP → Conv3d 分类头 |
| `SegmentationHeadCascadeCLS` | `modules.py:109` | 先二值占用分类 → 拼接 → 语义分类（级联模式） |
| `ASPP` | `modules.py:6` | 多尺度 3D 空洞卷积，残差连接 |

---

## 5. 配置体系

基于 Hydra YAML，单文件控制全部参数。以 `occdepth/config/semantic_kitti/flospdepth_2080ti.yaml` 为例：

```yaml
dataset: "kitti"
full_scene_size: [256, 256, 32]      # 3D 场景尺寸 (X, Y, Z)
project_scale: 2                      # 3D 投影下采样因子
feature: 24                           # 3D 网络通道数
feature_2d_oc: 24                     # 2D 网络输出通道数
n_classes: 20                         # 语义类别数
trans_2d_to_3d: "flosp_depth"         # 2D→3D 方法: flosp / flosp_depth

# 损失开关
CE_ssc_loss: true
sem_scal_loss: true
geo_scal_loss: true
fp_loss: true
context_prior: false
relation_loss: false

# 深度监督
use_stereo_depth_gt: false
use_lidar_depth_gt: false

# 级联分类
cascade_cls: false
occluded_cls: false
```

通过 `source env_{dataset}.sh` 切换数据集环境（设置 `DATA_CONFIG` 和 `DATA_LOG` 环境变量）。

---

## 6. 损失函数 (`occdepth/loss/`)

| 损失 | 文件 | 行号 | 描述 |
|------|------|------|------|
| `CE_ssc_loss` | `ssc_loss.py` | 90 | 带类别权重的交叉熵，忽略 255 |
| `sem_scal_loss` | `ssc_loss.py` | 43 | 逐类 precision/recall/specificity BCE |
| `geo_scal_loss` | `ssc_loss.py` | 16 | 二值（空/非空）几何尺度 BCE |
| `KL_sep` | `ssc_loss.py` | 6 | KL 散度（frustum 损失用） |
| `DepthClsLoss` | `depth_loss.py` | 7 | 深度 bin 分类 BCE |
| `compute_super_CP_multilabel_loss` | `CRP_loss.py` | 4 | 上下文关系多标签 BCE，带正负样本平衡 |
| `SSCMetrics` | `sscMetrics.py` | 40 | Scene Completion IoU + Semantic IoU |

---

## 7. 数据集支持

| 数据集 | 场景尺寸 | 类别数 | 输入类型 | 深度监督 |
|--------|---------|--------|---------|---------|
| SemanticKITTI | 256×256×32 | 20 | 立体图像 | 立体深度 / LiDAR 深度 |
| NYUv2 | 240×144×240 | 13 | RGB-D | GT 深度 |
| Sweeper | 80×80×48 | 24 | 立体图像 | 立体深度 |
| TartanAir | — | — | RGB-D | GT 深度（合成） |

---

## 关键设计决策总结

1. **深度感知特征分配**：`FlospDepth` 学习深度分布，对几何投影的 3D 特征进行注意力加权（`x3ds * x3ds_depth * 100`），解决传统投影的深度模糊性问题
2. **双模式 2D→3D**：`flosp`（纯几何，无参数）vs `flosp_depth`（深度感知），通过配置 `trans_2d_to_3d` 切换
3. **多视图融合**：SFA 用余弦相似度加权融合多视图特征，共视体素由特征相似度加权，单视图体素权重提升
4. **多尺度投影**：在 2D 特征尺度 [1, 2, 4, 8] 上分别做 SFA 投影，捕捉不同分辨率的上下文
5. **多任务损失**：6 种损失函数联合优化语义分割、几何完整性、深度估计和上下文关系
