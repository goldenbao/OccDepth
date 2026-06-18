# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

OccDepth is a depth-aware method for 3D Semantic Scene Completion (SSC). It supports both stereo image and RGB-D inputs. The project implements a 2D-to-3D feature projection pipeline with depth-aware feature fusion and 3D convolutional networks for occupancy prediction.

## Setup

- Conda environment with Python 3.7, PyTorch 1.13.1, CUDA 11.7
- Install: `pip install -r requirements.txt && conda install -c bioconda tbb=2020.2`
- Source `env_{dataset}.sh` before any script to set `DATA_CONFIG` and `DATA_LOG` env vars

## Key Commands

```bash
# Train (4 GPUs recommended)
source env_semanticKITTI.sh  # or env_NYU.sh / env_sweeper.sh
python occdepth/scripts/train.py logdir=${DATA_LOG} n_gpus=4 batch_size_per_gpu=1

# Generate output predictions
python occdepth/scripts/generate_output.py n_gpus=4 batch_size_per_gpu=1

# Evaluate
python occdepth/scripts/eval.py n_gpus=1 batch_size_per_gpu=1
```

## Project Structure

```
occdepth/
├── config/                    # Hydra YAML configs
│   ├── semantic_kitti/        # SemanticKITTI configs (256x256x32 scene)
│   ├── NYU/                   # NYUv2 configs
│   └── sweeper/               # Custom sweeper configs (80x80x48 scene)
├── data/                      # Dataset-specific modules
│   ├── semantic_kitti/        # KITTI dataset, preprocessing, params
│   ├── NYU/                   # NYU dataset, preprocessing, params
│   ├── sweeper/               # Custom sweeper dataset
│   ├── tartanair/             # TartanAir dataset
│   └── utils/                 # Shared data utilities (fusion, torch_util)
├── loss/                      # Loss functions
│   ├── ssc_loss.py            # CE, semantic scaling, geometric scaling, KL loss
│   ├── CRP_loss.py            # Context Relation Prior multi-label loss
│   ├── depth_loss.py          # Depth classification loss
│   └── sscMetrics.py          # IoU/mIoU metrics
├── models/                    # Network architecture
│   ├── OccDepth.py            # Main LightningModule — orchestrates entire pipeline
│   ├── unet2d.py              # 2D backbone (EfficientNet-based)
│   ├── unet3d_kitti.py        # 3D UNet decoder for KITTI (256x256x32)
│   ├── unet3d_nyu.py          # 3D UNet decoder for NYU
│   ├── unet3d_sweeper.py      # 3D UNet decoder for sweeper (80x80x48)
│   ├── SFA.py                 # Stereo Soft Feature Assignment (2D→3D projection)
│   ├── flosp_depth/           # FLOSP depth-aware 3D feature module
│   ├── f2v/                   # Frustum-to-voxel transformation
│   ├── CRP3D.py              # Context Relation Prior module
│   ├── DDR.py                # Depth Discretization & Redistribution
│   ├── modules.py            # Shared 3D blocks (Conv, Upsample, Downsample, SegmentationHead)
│   └── mobilenet/            # MobileNet alternative backbone
└── scripts/                   # Entry points
    ├── train.py               # Training script
    ├── eval.py                # Evaluation script (uses trainer.test)
    ├── generate_output.py     # Inference → saves .pkl predictions
    └── visualization/         # Visualization scripts per dataset
```

## Architecture

### Base Pipeline (shared by all modes)

1. **2D Backbone** (`UNet2D`): EfficientNet-based encoder producing multi-scale features at scales 1, 2, 4, 8
2. **2D→3D Projection** (`SFA`): Projects 2D features into 3D voxel space using frustum sampling with camera geometry
3. **Depth-Aware Features**: Depth-guided weighting of projected 3D features — method depends on `trans_2d_to_3d`:
   - `"flosp"`: No depth weighting (projection only)
   - `"flosp_depth"`: Learned depth estimation via FlospDepth (can use GT stereo depth or lidar depth for supervision)
   - `"igev_rr_depth"`: Frozen IGEV-RR stereo model provides depth (see below)
4. **3D UNet Decoder** (`UNet3D`): 3D convolutional encoder-decoder with skip connections, CRP (Context Relation Prior), cascade classification (occupancy → semantics), and optional occluded class prediction
5. **Losses**: CE loss (with class weighting), geometric scaling loss, semantic scaling loss, frustum-based KL loss, CRP relation loss, depth loss

### IGEV-RR Depth Mode (`trans_2d_to_3d: "igev_rr_depth"`)

Replaces FlospDepth with a frozen IGEV-RR stereo matching model as the depth source.

**Pipeline:**
1. IGEV-RR takes the stereo pair (left/right images in [0, 255] float32)
2. Outputs disparity → converted to depth via baseline-focal-length
3. Depth is encoded into an 80-bin LID (Learned Index Distribution) with bilinear interpolation + Gaussian smoothing + avg_pool2d(8,8)
4. FrustumGridGenerator + bilinear Sampler projects the depth distribution into voxel space
5. The resulting `x3ds_depth` weights the SFA geometric features: `x3ds = x3ds * x3ds_depth * 100`

**Key files:**
- `occdepth/models/igev_rr_wrapper.py`: Wraps IGEV-RR with manual package import from external codebase at `/home/data/bino_stereo/.../code`. Loads checkpoint, freezes all params.
- `occdepth/models/OccDepth.py` lines 243-302: igev_rr_depth init (SFA projects, IGEVRRWrapper, grid_generator, sampler)
- `occdepth/models/OccDepth.py` lines 507-538: Forward pass in igev_rr_depth mode
- `occdepth/models/OccDepth.py` lines 543+: `_forward_igev_rr_depth()` method

### Student Distillation (`use_igev_student: true`)

Trains a FlospDepth student network supervised by the frozen IGEV-RR teacher — no GT depth labels required.

**How it works:**
- IGEV-RR teacher produces dense depth pseudo-labels (`depth_teacher`)
- FlospDepth student predicts a depth distribution (`depth_pred`) from 2D features
- DepthClsLoss (BCE on 80-bin classification) supervises the student against teacher
- At inference, voxel weights are fused: `x3ds_depth_fused = alpha * x3ds_depth_teacher + (1-alpha) * x3ds_student`
- Default `alpha = 0.7` (teacher-dominant fusion)

**Config:**
- `use_igev_student: true/false` — enable student
- `igev_student_fuse_alpha: 0.7` — teacher weight in fusion

**Key code:**
- `occdepth/models/OccDepth.py` lines 304-319: Student init (FlospDepth + DepthClsLoss)
- `occdepth/models/OccDepth.py` lines 514-536: Student forward + fusion in `_forward_2d_to_3d`
- `occdepth/loss/depth_loss.py`: DepthClsLoss — downsamples GT depth 1/8, one-hot encodes, BCE with prediction

## Configuration

All configs use Hydra (omegaconf). Each dataset has a YAML config specifying:
- `full_scene_size`: 3D voxel grid dimensions (e.g., [256, 256, 32] for KITTI)
- `trans_2d_to_3d`: `"flosp"`, `"flosp_depth"`, or `"igev_rr_depth"` — 2D-to-3D transformation method
- `feature` / `feature_2d_oc`: 3D and 2D feature channel dimensions
- `project_scale`: Downsample factor for 3D projection
- `cascade_cls`: Enable two-stage classification (binary occupancy → semantics)
- `context_prior`: Enable CRP module
- Loss toggles: `CE_ssc_loss`, `sem_scal_loss`, `geo_scal_loss`, `fp_loss`, `relation_loss`

### IGEV-RR Config Options

- `igev_rr_ckpt`: Path to IGEV-RR .pth checkpoint (required for igev_rr_depth mode)
- `igev_rr_max_disp`: Maximum disparity in pixels (default 192)
- `use_igev_student`: Enable FlospDepth student trained by IGEV-RR teacher
- `igev_student_fuse_alpha`: Teacher weight in voxel fusion (default 0.7)
- `use_stereo_depth_gt: false` / `use_lidar_depth_gt: false` / `use_depth_gt: false` — all false in igev_rr_depth mode since teacher replaces GT depth
- `depth_loss_weight: 0.0` — no GT depth loss needed when using teacher

## Long-Tail Loss Configuration

Loss type and class weighting are configurable to handle class imbalance (Sweeper has 24 classes with 96.68% empty voxels among valid voxels).

### SSC Loss Type (`ssc_loss_type`)

Controls which loss function is used for the main SSC head:

| Value | Loss | Description |
|-------|------|-------------|
| `"ce"` | `CE_ssc_loss` | Weighted CrossEntropy (default, backward compatible) |
| `"focal"` | `FocalLoss` | `-α_t(1-p_t)^γ log(p_t)` — down-weights easy samples, focuses on hard/rare classes |
| `"dice"` | `DiceLoss` | `1 - (2\|A∩B\|+smooth)/(\|A\|+\|B\|+smooth)` — optimizes IoU per class, imbalance-agnostic |
| `"ce+dice"` | CE + Dice | Combines CE global gradient with Dice per-class constraint |
| `"focal+dice"` | Focal + Dice | Combines focal down-weighting with Dice per-class optimization |

Parameters: `focal_gamma: 2.0`, `dice_smooth: 1.0`

### Class Weight Mode (`class_weight_mode`)

| Mode | Sweeper | KITTI |
|------|---------|-------|
| `"uniform"` | `[0.05, 1, 1, ...]` (default) | Frequency-based (from train.py) |
| `"frequency"` | `1/log(freq+0.001)` computed from actual label scan of 16750 files | Same as current behavior |

Frequency data (from `occdepth/data/sweeper/params.py`):
```
empty 3247895988 → weight 0.046
floor   35567452 → weight 0.058
wall    17088224 → weight 0.060
wire      337580 → weight 0.079
shoe        5923 → weight 0.115
pet          746 → weight 0.151
pet_waste      0 → weight capped at 1/log(1+0.001) ≈ 1000 (zero-shot class)
carpet    200153 → weight 0.082
paper     453345 → weight 0.077
blocks      7105 → weight 0.113
other    4033218 → weight 0.066
```

**Key files:**
- `occdepth/loss/ssc_loss.py`: `CE_ssc_loss`, `FocalLoss`, `DiceLoss` implementations
- `occdepth/data/sweeper/compute_frequencies.py`: One-off frequency statistics script
- `occdepth/data/sweeper/params.py`: Hardcoded `sweeper_class_frequencies` array
- `occdepth/loss/ablation_plan.md`: Ablation experiments for loss comparison
- `occdepth/loss/loss_summary.md`: Full documentation of all loss functions

## Datasets

- **SemanticKITTI**: 19 semantic classes + 1 free class, 256x256x32 scene, stereo depth + lidar depth available
- **NYUv2**: 12 classes, 240x144x240 scene, RGB-D input
- **Sweeper**: Custom dataset, 80x80x48 scene, 24 classes, stereo depth supervision
- **TartanAir**: Synthetic dataset

## Multi-View Mode

When `multi_view_mode=True`, the model processes multiple camera views per sample. The 2D backbone can optionally share gradients across views (`share_2d_backbone_gradient`) to save GPU memory at a slight accuracy cost.

## IGEV-RR Weight Protection (Three-Layer Safeguard)

IGEV-RR weights must never change during training. Three independent mechanisms enforce this:

1. **`train()` override** (line 352): Calls `self.igev_rr.eval()` after every `super().train()`, preventing BatchNorm running stats drift even when `requires_grad=False`.
2. **`on_save_checkpoint` override** (line 360): Replaces IGEV-RR entries in the checkpoint with freshly-loaded .pth weights before every save, preventing checkpoint file corruption.
3. **`load_state_dict` override** (line 323): After loading any checkpoint, immediately re-instantiates IGEV-RR from the original .pth, undoing any corruption that may have occurred during checkpoint load.

**Critical note:** Without these guards, `model.load_from_checkpoint()` → `load_state_dict(checkpoint["state_dict"])` overwrites IGEV-RR weights with whatever was saved in the checkpoint — even if that checkpoint had corrupted weights from a previous save.

### Additional IGEV-RR Safeguards

- **Empty checkpoint path** → `FileNotFoundError` (was silently using random weights)
- **Parameter count mismatch** → `RuntimeError` in `_load_ckpt()` with details on missing keys
- **Auto-relaxed strict loading**: When a checkpoint is missing `igev_student_depth.*` keys (e.g., loading an old checkpoint that predates the student), `load_state_dict` auto-relaxes strict mode so the student head can be randomly initialized.

## Multi-Root Data Support

The sweeper dataset (`occdepth/data/sweeper/sweeper_dataset.py`) accepts `root` as either a string or a list of strings:

- Single root: `root="/path/to/data"`
- Multiple roots: `root=["/path/to/data1", "/path/to/data2"]`

When multiple roots are provided, data from all roots is merged and shuffled together. This enables training on data collected from different environments, camera rigs, or lighting conditions.

The DataModule (`sweeper_dm.py`) converts config `data_roots` (YAML list) into a Python list passed to the dataset. In eval.py, the fallback logic handles both `config.data_roots` (list) and `config.data_root` (single str).

## Supplementary Documentation

The following `.md` files contain detailed reference material and are loaded on demand:
- `occdepth/arch_overview.md` — Architecture notes
- `occdepth/data/sweeper/projection_params.md` — Sweeper camera projection parameters
- `occdepth/loss/loss_summary.md` — All loss functions, formulas, config switches, and per-dataset class weights
- `README.md` — Project-level README
