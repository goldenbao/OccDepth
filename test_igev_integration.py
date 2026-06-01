"""Quick smoke test for OccDepth with igev_rr_depth mode (runs inside Docker)."""
import sys, os
sys.path.insert(0, "/home/data/OCC/OccDepth")
sys.path.insert(0, "/home/data/bino_stereo/binocularstereovision/scratch_igev_rr_except_mn2_on_sceneflow_w_aug/code")

from omegaconf import OmegaConf, DictConfig
import torch
import numpy as np

from occdepth.data.sweeper.params import sweeper_class_names, sweeper_class_weights
from occdepth.models.OccDepth import OccDepth

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---- Build OmegaConf config matching the YAML ----
cfg_str = """
dataset: sweeper
frustum_size: 8
project_scale: 2
n_relations: 4
lr: 2e-4
weight_decay: 1e-4
fp_loss: true
context_prior: true
relation_loss: true
CE_ssc_loss: true
sem_scal_loss: true
geo_scal_loss: true
n_classes: 24
feature: 32
feature_2d_oc: 32
trans_2d_to_3d: igev_rr_depth
cascade_cls: true
occluded_cls: false
sem_step_decay_loss: false
multi_view_mode: true
share_2d_backbone_gradient: true
full_scene_size: [80, 80, 48]
backbone_2d_name: tf_efficientnet_b3_ns
return_up_feats: 1
project_1_2: true
project_1_4: true
project_1_8: true
use_stereo_depth_gt: false
use_lidar_depth_gt: false
use_depth_gt: false
depth_loss_weight: 0.0
igev_rr_ckpt: /home/data/bino_stereo/binocularstereovision/scratch_igev_rr_except_mn2_on_sceneflow_w_aug/ckpt/igev_rr_260209_2024_ep63.pth
igev_rr_max_disp: 192
enable_log: false
deterministic: false
batch_size_per_gpu: 1
n_gpus: 1
"""
cfg = OmegaConf.create(cfg_str)

class_names = sweeper_class_names
class_weights = torch.as_tensor(sweeper_class_weights, dtype=torch.float32)

print("1. Instantiating OccDepth with igev_rr_depth ...")
model = OccDepth(
    class_names=class_names,
    class_weights=class_weights,
    class_weights_occ=None,
    full_scene_size=(80, 80, 48),
    project_res=[1, 2, 4, 8],
    config=cfg,
    infer_mode=False,
)
model = model.to(device)
model.eval()
print("   OK")

# Count params
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Total params: {total:,}")
print(f"   Trainable params: {trainable:,}")
print(f"   IGEV-RR frozen params: {sum(p.numel() for p in model.igev_rr.parameters()):,}")

# ---- Dummy batch ----
B, n_views, C_, H, W = 1, 2, 3, 480, 640
n_patterns = 5  # pattern_id=1 in sweeper dataset
print(f"\n2. Creating dummy batch ({B}x{n_views}x{C_}x{H}x{W}) ...")

# Image in ImageNet-normalized range (mean=0, std=1 approx)
dummy_img = torch.randn(B, n_views, C_, H, W, device=device) * 0.2

# projected_pix for project_scale=2: (40*40*24 = 38400 voxels at 0.02m)
# shape per sample: [n_views, N_voxels, n_patterns, 2]
N_vox_2 = 38400
N_vox_1 = 307200  # output_scale=1: (80*80*48 = 307200 voxels at 0.01m)

scale_3ds = [1, 2]
batch = {"img": dummy_img}
for s in scale_3ds:
    Nv = N_vox_1 if s == 1 else N_vox_2
    # Generate realistic pixel coords: x in [0, W), y in [0, H)
    pix_x = torch.randint(0, W, (n_views, Nv, n_patterns, 1), device=device)
    pix_y = torch.randint(0, H, (n_views, Nv, n_patterns, 1), device=device)
    proj_pix = torch.cat([pix_x, pix_y], dim=-1)  # (n_views, Nv, n_patterns, 2)
    # Mask ~5% voxels as out-of-FOV (realistic)
    fov_mask = torch.rand(n_views, Nv, n_patterns, device=device) > 0.05
    batch[f"projected_pix_{s}"] = [proj_pix]
    batch[f"fov_mask_{s}"] = [fov_mask]

# Camera params (lists mimicking collate output — each tensor per sample)
# Move all camera params to GPU to match image tensor device
dev = device
cam_k_list = [
    torch.eye(3, dtype=torch.float64, device=dev).unsqueeze(0).repeat(n_views, 1, 1)
]
T_list = [
    torch.eye(4, dtype=torch.float32, device=dev).unsqueeze(0).repeat(n_views, 1, 1)
]
ida_list = [
    torch.eye(4, dtype=torch.float32, device=dev).unsqueeze(0).repeat(n_views, 1, 1)
]
batch["cam_k"] = cam_k_list
batch["T_velo_2_cam"] = T_list
batch["ida_mats"] = ida_list

print("3. Running forward pass ...")
torch.cuda.empty_cache()
with torch.no_grad():
    out = model(batch)
print("   OK")

print(f"\n4. Output keys: {list(out.keys())}")
if "ssc_logit" in out:
    print(f"   ssc_logit: {out['ssc_logit'].shape}")
    print(f"   range=[{out['ssc_logit'].min().item():.2f}, {out['ssc_logit'].max().item():.2f}]")

print("\n=== PASSED ===")
