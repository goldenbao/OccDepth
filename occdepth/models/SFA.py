import torch
import torch.nn as nn


class SFA(nn.Module):
    """
    Stereo Soft Feature Assignment (no learned parameters).
    Projects 2D features into 3D voxel space using pre-computed frustum pixel indices.
    Multi-view fusion uses cosine-similarity weighting: shared-view voxels are weighted
    by feature similarity, while single-view voxels get a boost.
    """
    def __init__(self, scene_size, dataset, project_scale):
        super().__init__()
        self.scene_size = scene_size
        self.dataset = dataset
        self.project_scale = project_scale

    def forward(self, x2d, projected_pix, fov_mask):
        n_views, c, h, w = x2d.shape

        # Gather per-view features at projected pixel locations, with average pooling
        # over multiple projection patterns and zero-filling for out-of-FOV voxels.
        src_feature = []
        n_views_weights = []
        for idx in range(n_views):
            src = x2d[idx].view(c, -1)
            zeros_vec = torch.zeros(c, 1).type_as(src)
            src = torch.cat([src, zeros_vec], 1)
            pix_x, pix_y = projected_pix[idx, :, :, 0], projected_pix[idx, :, :, 1]
            img_indices = pix_y * w + pix_x
            weights = torch.clone(img_indices)
            weights[fov_mask[idx]] = 1
            weights[~(fov_mask[idx])] = 0
            img_indices[~(fov_mask[idx])] = h * w
            img_indices = img_indices.expand(c, -1, -1).long()  # c, HWD
            sub_src_feature = torch.gather(src, 1, img_indices[:, :, 0])
            for pattern_idx in range(1, img_indices.shape[2]):
                sub_src_feature += torch.gather(src, 1, img_indices[:, :, pattern_idx])
            sub_weights = torch.sum(weights, 1)
            sub_src_feature = sub_src_feature / sub_weights
            sub_weights = sub_weights / sub_weights
            sub_src_feature = torch.where(
                torch.isnan(sub_src_feature),
                torch.full_like(sub_src_feature, 0),
                sub_src_feature,
            )
            sub_weights = torch.where(
                torch.isnan(sub_weights), torch.full_like(sub_weights, 0), sub_weights
            )
            src_feature.append(sub_src_feature)
            n_views_weights.append(sub_weights)

        # Multi-view fusion: each voxel's feature is a cosine-similarity-weighted average
        # across all view pairs. Voxels visible in only one view get a unit weight boost
        # to avoid attenuation from zero-paired terms.
        n_views_weights = torch.stack(n_views_weights)
        cos_weight = torch.zeros(n_views, n_views, projected_pix.shape[1]).type_as(x2d)
        src_feature = torch.stack(src_feature)
        sum_wheight_feature = torch.zeros(c, projected_pix.shape[1]).type_as(x2d)
        sum_norm = torch.zeros(projected_pix.shape[1]).type_as(x2d)
        for idx_i in range(n_views):
            for idx_j in range(idx_i + 1, n_views):
                cos_weight_clone = cos_weight.clone()
                weight_ij = (
                    n_views_weights[idx_i] * n_views_weights[idx_j]
                )  ## pick the voxels that both in their frustums
                weight_diff = n_views_weights[idx_i] - n_views_weights[idx_j]
                weight_i_vec = torch.zeros(projected_pix.shape[1]).type_as(
                    n_views_weights
                )
                weight_j_vec = torch.zeros(projected_pix.shape[1]).type_as(
                    n_views_weights
                )
                weight_i_vec[
                    weight_diff > 0
                ] = 1  ## pick  the voxels that only in the idx_i frustum
                weight_j_vec[
                    weight_diff < 0
                ] = 1  ## pick  the voxels that only in the idx_j frustum
                cos_weight_single =torch.cosine_similarity(src_feature[idx_i],src_feature[idx_j],0)*weight_ij
                cos_weight_clone[idx_i,idx_j] = cos_weight_single + weight_i_vec
                cos_weight_clone[idx_j,idx_i] = cos_weight_single + weight_j_vec

                sum_wheight_feature += (
                    cos_weight_clone[idx_i, idx_j] * src_feature[idx_i]
                    + cos_weight_clone[idx_j, idx_i] * src_feature[idx_j]
                )
                # Count non-zero weight directions (not sum of weights), so that:
                # co-visible voxels divide by 2 (keep cos_sim in numerator),
                # single-view voxels divide by 1 (retain full feature strength).
                sum_norm += (cos_weight_clone[idx_i, idx_j] > 0).float() \
                          + (cos_weight_clone[idx_j, idx_i] > 0).float()

        if(n_views > 1):
            sum_norm = torch.where(
                sum_norm > 0, sum_norm, torch.ones_like(sum_norm)
            )
            sum_wheight_feature = sum_wheight_feature / sum_norm
            
            # sum_wheight_feature = sum_wheight_feature / (
            #     n_views * (n_views - 1)
            # )  # n_views is greater than 1
        else:
            sum_wheight_feature = src_feature[0]
        src_feature = sum_wheight_feature
        if self.dataset == "NYU":
            x3d = src_feature.reshape(
                c,
                self.scene_size[0] // self.project_scale,
                self.scene_size[2] // self.project_scale,
                self.scene_size[1] // self.project_scale,
            )
            x3d = x3d.permute(0, 1, 3, 2)
        elif self.dataset == "kitti":
            x3d = src_feature.reshape(
                c,
                self.scene_size[0] // self.project_scale,
                self.scene_size[1] // self.project_scale,
                self.scene_size[2] // self.project_scale,
            )
        elif self.dataset == "sweeper":
            x3d = src_feature.reshape(
                c,
                self.scene_size[0] // self.project_scale,
                self.scene_size[1] // self.project_scale,
                self.scene_size[2] // self.project_scale,
            )

        return x3d
