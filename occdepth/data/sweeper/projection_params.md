# SweeperDataset 几何投影参数详解

本文档解释 `occdepth/data/sweeper/sweeper_dataset.py` 中与 3D 体素到 2D 图像投影相关的核心参数，包括其物理意义、计算方法以及在 pipeline 中的作用。

---

## 1. 物理场景参数

```python
# sweeper_dataset.py:151-156
self.scene_size = (0.8, 0.8, 0.48)       # (X, Y, Z) 物理范围, 单位: 米
self.vox_origin = np.array([0.1, -0.4, -0.1])  # 体素网格原点, 单位: 米
self.voxel_size = 0.01                     # 每个体素边长 = 1cm
```

**物理意义:**
- `scene_size`: 场景是一个 `0.8m × 0.8m × 0.48m` 的长方体，分别对应车辆的前后方向 (X)、左右方向 (Y)、上下方向 (Z)
- `vox_origin`: 体素网格 (0,0,0) 索引处体素中心的**世界坐标**。以扫地机 body 为原点，X 正方向为前进方向，则该原点在 body 前方 0.1m、左侧 0.4m、下方 0.1m
- `voxel_size`: 每个体素边长 1cm，是坐标到体素索引的换算基准

**全分辨率体素网格计算:**
```
nx = 0.8 / 0.01 = 80
ny = 0.8 / 0.01 = 80
nz = 0.48 / 0.01 = 48
全分辨率: 80 × 80 × 48 = 307,200 个体素
```

---

## 2. 投影尺度参数

```python
# sweeper_dataset.py:149-150
self.project_scale = 2        # 由 config 传入, 3D 投影下采样因子
self.output_scale = math.ceil(self.project_scale / 2)  # = 1
```

项目使用**两个尺度**的投影 (`scale_3ds = [1, 2]`):

| `scale_3d` | 体素大小 | 网格尺寸 | 体素数 | 用途 |
|-----------|---------|---------|-------|------|
| 1 (output_scale) | 1cm | 80×80×48 | 307,200 | CE loss、语义分割输出 |
| 2 (project_scale) | 2cm | 40×40×24 | 38,400 | frustum (FP) loss |

投影时使用的体素大小为 `voxel_size × scale_3d`，即:
- `scale_3d=1`: 1cm/体素 → 保留全部细节
- `scale_3d=2`: 2cm/体素 → 粗粒度，用于 frustum 级别的局部损失计算

---

## 3. 投影核心函数: `vox2pix`

定义在 `occdepth/data/utils/helpers.py:94-169`，将体素中心坐标从 3D 世界投影到 2D 图像像素。

### 输入参数

| 参数 | 来源 | 形状 | 含义 |
|------|------|------|------|
| `cam_E` | `T_velo_2_cam[idx_view]` | (4,4) | body→相机的刚体变换矩阵 (外参) |
| `cam_k` | `cam_k[idx_view]` | (3,3) | 相机内参矩阵 |
| `vox_origin` | `self.vox_origin` | (3,) | 体素原点世界坐标 |
| `voxel_size` | `self.voxel_size * scale_3d` | scalar | 当前尺度的体素大小 |
| `img_W` | 640 | scalar | 图像宽度 (像素) |
| `img_H` | 480 | scalar | 图像高度 (像素) |
| `scene_size` | `self.scene_size` | (3,) | 场景物理范围 (米) |
| `pattern_id` | `self.pattern_id` | scalar | 投影 pattern (见下) |

### 内部计算流程

```
Step 1: 生成 3D 体素索引网格
  vol_dim = scene_size / voxel_size          # 各维度体素数
  xv, yv, zv = meshgrid(range(vol_dim))      # 所有体素索引
  vox_coords = stack([xv, yv, zv])           # (N, 3), N = 80×80×48

Step 2: 体素索引 → 世界坐标 (米)
  cam_pts = vox_origin + vox_coords × voxel_size
  即: x_world = 0.1 + x_idx × 0.01
      y_world = -0.4 + y_idx × 0.01
      z_world = -0.1 + z_idx × 0.01

Step 3: 世界坐标 → 相机坐标 (刚体变换)
  cam_pts_cam = cam_E @ [cam_pts, 1]^T      # (4,4) × (N,4) → (N,3)
  包含旋转 R(3×3) 和平移 t(3×1)

Step 4: 相机坐标 → 像素坐标 (透视投影 + pattern偏移)
  u = fx × (x/z) + cx + pattern_offset_u
  v = fy × (y/z) + cy + pattern_offset_v

Step 5: FOV 过滤
  fov_mask = (u ≥ 0) & (u < W) & (v ≥ 0) & (v < H) & (z > 0)
```

### 返回值的形状和含义

```python
projected_pix  # (n_views, N, n_patterns, 2)
fov_mask       # (n_views, N, n_patterns)
pix_z          # (n_views, N)
```

- `projected_pix`: 每个体素在图像上投影的像素坐标 (u, v)。`n_patterns` 取决于 pattern_id
- `fov_mask`: 布尔掩码，标记哪些体素投影到图像 FOV 内 (以及深度 > 0)
- `pix_z`: 体素在相机坐标系下的深度值 (米)，用于后续计算

---

## 4. Pattern ID — 体素覆盖模式

定义在 `occdepth/data/utils/fusion.py:236`，`cam2allpixs` 函数内。

每个体素可以投影到图像上的多个像素，以处理体素"覆盖"多个像素的问题：

| pattern_id | 点数 | 像素偏移模式 |
|-----------|------|-------------|
| 0 | 1 | 仅体素中心 `[(0,0)]` |
| 1 | 5 | 中心 + 上下左右 `[(0,0), (0,-1), (-1,0), (1,0), (0,1)]` |
| 2 | 5 | 中心 + 四角 `[(0,0), (-1,-1), (1,1), (-1,1), (1,-1)]` |
| 3 | 9 | 3×3 邻域全覆盖 |

在 `igev_rr_depth` 模式下使用 `pattern_id=0`，因为 IGEV-RR 输出的深度图已经是稠密的，不需要多像素覆盖投票。

---

## 5. 相机参数加载

```python
# sweeper_dataset.py:47-109, get_sweeper_calib()
```

从各数据根目录下的 `occ_config.yaml` 加载，返回:

| 返回值 | 形状 | 含义 |
|--------|------|------|
| `T_velo_2_cam` | (2, 4, 4) | body→左相机、body→右相机的变换矩阵 |
| `proj_matrix` | (2, 3, 4) | 投影矩阵 `K @ [R\|t]` |
| `camk_3x4` | (3, 4) | 相机内参矩阵 (带第4列0填充) |

**关键计算:**
```python
T_cam_to_body = config['occupancy']['camera_external']  # 相机外参
T_body_2_caml = inv(T_cam_to_body)                      # body→左相机
T_caml_to_camr = [[1,0,0,-baseline], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
T_body_2_camr = T_caml_to_camr @ T_body_2_caml          # body→右相机
```

左右相机共用同一套内参 (rectified 双目系统):
```python
cam_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
```

---

## 6. 数据流汇总

```
occ_config.yaml
    │
    ▼
get_sweeper_calib()
    │
    ├── T_velo_2_cam (2, 4, 4)  ← body→camera 变换
    ├── proj_matrix (2, 3, 4)   ← K @ [R|t]
    └── cam_k (3, 3)            ← 内参
    │
    ▼
__getitem__() 中对每个 view × 每个 scale_3d 调用 vox2pix()
    │
    ├── projected_pix_{s}  → SFA 模块查表: 2D 特征在体素位置处的采样坐标
    ├── fov_mask_{s}       → 过滤 FOV 外体素的投影
    └── pix_z_{s}          → frustum loss 计算深度分布
    │
    ▼
data dict:
  ├── "img"                    # (2, 3, 480, 640) 左右目 RGB
  ├── "cam_k"                  # (2, 3, 3) 左右目内参
  ├── "T_velo_2_cam"           # (1 or 2, 4, 4) 外参
  ├── "projected_pix_1"        # (n_views, 307200, n_patterns, 2)
  ├── "fov_mask_1"             # (n_views, 307200, n_patterns)
  ├── "pix_z_1"                # (n_views, 307200)
  ├── "projected_pix_2"        # (n_views, 38400, n_patterns, 2)
  ├── "fov_mask_2"             # (n_views, 38400, n_patterns)
  ├── "pix_z_2"                # (n_views, 38400)
  ├── "target"                 # (80, 80, 48) 语义 GT
  └── "frustums_masks"         # frustum loss 用
```

---

## 7. CP_mega_matrix — Context Relation Prior 监督信号

```python
# sweeper_dataset.py:345-352
target_4_path = os.path.join(root, sequence, "occupancy_gt", "SLAM_SLAM_L_"+ frame_id+"_occ_gt_1_4.npy")
target_1_4 = np.load(target_4_path)
CP_mega_matrix = compute_CP_mega_matrix(target_1_4)
data["CP_mega_matrix"] = CP_mega_matrix
```

### 7.1 作用

`CP_mega_matrix` 是 **CRP (Context Relation Prior) 模块的监督信号**（ground-truth relation matrix）。CRP3D (`occdepth/models/CRP3D.py`) 会从 3D 特征中预测 voxel 与 supervoxel 之间的语义关系矩阵，而 `CP_mega_matrix` 就是这些预测关系的真值，通过 `CRP_loss.py` 中的 BCEWithLogitsLoss 进行监督。

### 7.2 构造过程

`compute_CP_mega_matrix` 定义于 `occdepth/data/utils/helpers.py:6-91`:

```
输入: target_1_4  (H', W', D')   — 1/4 分辨率下采样后的 GT
          每个体素称为一个 "supervoxel"，在原始分辨率的 2×2×2 范围内取众数语义标签

输出: CP_mega_matrix  (4, N, M)
          N = H' × W' × D'  (supervoxel 数量)
          M = (H'/2) × (W'/2) × (D'/2)  (mega-voxel 数量, 每个 mega-voxel 覆盖 2×2×2 个 supervoxel)
          4 个 channel 对应 4 种关系类型
```

**关系类型 (4 channels):**

| channel | 名称 | 含义 |
|---------|------|------|
| 0 | non-non-same | voxel 与 mega-voxel 均为 occupied 且类别相同 |
| 1 | non-non-diff | voxel 与 mega-voxel 均为 occupied 但类别不同 |
| 2 | empty-empty | voxel 与 mega-voxel 均为 free (类别 0) |
| 3 | nonempty-empty | 一方 occupied 另一方 free |

每个 `matrix[rel, i, j] ∈ {0, 1}` 表示第 i 个 supervoxel 与第 j 个 mega-voxel 之间是否存在第 rel 种关系。

### 7.3 与 CRP3D 模块的关系

```python
# CRP3D.py:54-84 forward()
x_mega_context = self.mega_context(x_agg)            # 3D 特征 → 下采样得到 mega 特征
x_context_prior_logit = self.context_prior_logits[rel](x_agg)  # 预测关系 logits
# shape: (bs, M, N) — 对应 CP_mega_matrix 的 (N, M) 转置

# relation_loss (CRP_loss.py):
loss = BCEWithLogitsLoss(pred_logits, CP_mega_matrix)
```

即:
1. 3D 特征经 `mega_context` (stride-2 conv) 下采样得到 mega-voxel 特征
2. 对每个 relation 类型，用 `context_prior_logits[rel]` 预测关系矩阵
3. 与 `CP_mega_matrix` 计算 BCEWithLogitsLoss，让模型学会体素间的上下文语义关系

### 7.4 在 pipeline 中的位置

```
target (80, 80, 48)  ── 下采样 ──► target_1_4 (H', W', D')
                                       │
                                       ▼
                              compute_CP_mega_matrix()
                                       │
                                       ▼
                              CP_mega_matrix (4, N, M)
                                       │
                                       ▼
                              CRP_loss: 与 CRP3D 预测的关系图计算 BCE
```

---

---

## 8. Frustums Masks & Frustums Class Dists — 局部视锥损失

```python
# sweeper_dataset.py:361-384
projected_pix_output = data["projected_pix_{}".format(self.output_scale)]  # scale_3d=1
pix_z_output = data["pix_z_{}".format(self.output_scale)]
frustums_masks, frustums_class_dists = compute_local_frustums(
    projected_pix_output,
    pix_z_output,
    target,
    self.img_W,
    self.img_H,
    dataset="kitti",
    n_classes=self.n_classes,
    size=self.frustum_size,        # =4, config 传入
)
```

### 8.1 核心思想

Frustum loss 是一个**区域级别的类别分布约束**。它不关心每个体素单独预测得准不准，而是检查模型的预测在每个局部区域内**整体类别的比例**是否与 GT 一致。

打个比方：CE loss 要求"每个体素分类正确"，frustum loss 要求"车这个物体在空间中的整体形状和位置大致对"。

### 8.2 构造过程

`compute_local_frustums` (`helpers.py:183-260`) 将图像划分为 `size × size` 的网格：

```
size=4 → 4×4 = 16 个 frustum 区域

图像 (480×640) 划分:
  frustum (0,0): 像素 x∈[0,160), y∈[0,120)   — 左上
  frustum (0,1): 像素 x∈[160,320), y∈[0,120)
  ...
  frustum (3,3): 像素 x∈[480,640), y∈[360,480) — 右下
```

**frustums_masks 的计算：**

```
对每个 frustum 区域:
  遍历左右视图:
    找到所有投影像素落在这个图像区域内的体素
    合并左右视图的结果 (union)
  输出: 一个 (80, 80, 48) 的 bool mask，标记哪些体素属于这个 frustum

形状: (16, 80, 80, 48)  — 每个 frustum 对应一个三维 mask
```

**frustums_class_dists 的计算：**

```
对每个 frustum 区域:
  取出 mask 内体素的 GT 语义标签
  统计每个类别出现的次数 (count)
  输出: 一个 (n_classes,) 的计数向量

形状: (16, 24)  — 每个 frustum 对应 24 个类别的计数
```

### 8.3 在 Frustum Loss 中的使用

```python
# OccDepth.py:862-896
pred_prob = softmax(ssc_pred)   # 模型预测的体素类别概率 (bs, 24, 80, 80, 48)

for frus in range(16):  # 遍历每个 frustum
    mask = frustums_masks[:, frus, :, :, :]  # (bs, 80, 80, 48) bool

    # 预测：该 frustum 内所有体素的概率求和 → 类别分布
    prob = (mask * pred_prob).sum()            # 各体素概率累加
    pred_dist = prob / prob.sum()              # 归一化为分布

    # GT：该 frustum 内各类别计数归一化
    gt_dist = batch_cnt[frus] / total_cnt      # 目标类别比例

    # KL 散度：让预测分布靠近 GT 分布
    loss = KL_divergence(pred_dist, gt_dist)
```

### 8.4 与 CE Loss 的区别

| | CE Loss | Frustum Loss |
|---|---|---|
| **粒度** | 逐体素 | 逐区域 (frustum) |
| **监督信号** | 每个体素的精确类别 | 区域内各类别的**比例** |
| **作用** | 精细的语义分类 | 形状级约束，防止大范围漏检 |
| **对噪声的鲁棒性** | 敏感 | 不敏感（统计分布） |

两个 loss 互补：CE 保证每个体素分类对，frustum loss 保证整体空间分布合理。

---

## 9. 与模型各模块的关系

| 模块 | 使用到的 projection 参数 |
|------|------------------------|
| SFA (2D→3D 特征投影) | `projected_pix_{scale}` 查表取 2D 特征 |
| FlospDepth (深度分布预测) | `cam_k`, `T_velo_2_cam`, `vox_origin`, `voxel_size`, `scene_size` |
| IGEV-RR (深度估计) | `cam_k` (用于 disp→depth 转换) |
| Frustum Loss | `pix_z_{scale}`, `fov_mask_{scale}`, `projected_pix_{scale}` |
| CE Loss | `target` (全分辨率 80×80×48) |
| CRP Loss (relation_loss) | `CP_mega_matrix` (从 target_1_4 构造) |
