# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from occdepth.models.modules import (
    SegmentationHead,
    SegmentationHeadCascadeCLS,
    SegmentationHeadOccludedCLS,
)
from occdepth.models.CRP3D import CPMegaVoxels
from occdepth.models.modules import Process, Upsample, Downsample, Convblock3d

class UNet3D(nn.Module):
    def __init__(
        self,
        class_num,
        norm_layer,
        full_scene_size, # 你的扫地机原始尺寸 (80, 80, 48)
        feature,
        project_scale,   # 此时建议传入 2
        context_prior=None,
        bn_momentum=0.1,
        cascade_cls=False,
        occluded_cls=False,
        infer_mode=False,
    ):
        super(UNet3D, self).__init__()
        self.project_scale = project_scale
        self.full_scene_size = full_scene_size
        self.feature = feature
        self.cascade_cls = cascade_cls
        self.occluded_cls = occluded_cls
        self.infer_mode = infer_mode

        # 🎯 【核心修改点 1】重新规划 UNet 的三层空间分辨率
        # 假设 full_scene_size=(80, 80, 48), project_scale=2
        size_l1 = (
            int(self.full_scene_size[0] / project_scale), # L1 尺度: 40x40x24 (1/2 尺度)
            int(self.full_scene_size[1] / project_scale),
            int(self.full_scene_size[2] / project_scale),
        )
        # 🎯 L2 尺度直接作为我们的关系计算层：20x20x12 (刚好是 1/4 尺度！)
        size_l2 = (size_l1[0] // 2, size_l1[1] // 2, size_l1[2] // 2) 
        
        # L3 尺度继续下采样作为最底层的深层全局特征: 10x10x6 (1/8 尺度)
        size_l3 = (size_l2[0] // 2, size_l2[1] // 2, size_l2[2] // 2)

        dilations = [1, 2, 3]
        
        # 骨干下采样网络保持不变
        self.process_l1 = nn.Sequential(
            Process(self.feature, norm_layer, bn_momentum, dilations=[1, 2, 3]),
            Downsample(self.feature, norm_layer, bn_momentum),
        )
        self.process_l2 = nn.Sequential(
            Process(self.feature * 2, norm_layer, bn_momentum, dilations=[1, 2, 3]),
            Downsample(self.feature * 2, norm_layer, bn_momentum),
        )

        # 上采样解码网络保持不变
        self.up_13_l2 = Upsample(self.feature * 4, self.feature * 2, norm_layer, bn_momentum)
        self.up_12_l1 = Upsample(self.feature * 2, self.feature, norm_layer, bn_momentum)
        
        if self.project_scale == 1:
            self.up_l1_lfull = Convblock3d(self.feature, self.feature // 2, norm_layer, bn_momentum, stride=1)
        else:
            self.up_l1_lfull = Upsample(self.feature, self.feature // 2, norm_layer, bn_momentum)
            
        if self.cascade_cls:
            self.ssc_head = SegmentationHeadCascadeCLS(self.feature // 2, self.feature // 2, class_num, dilations)
        else:
            self.ssc_head = SegmentationHead(self.feature // 2, self.feature // 2, class_num, dilations)

        if self.occluded_cls:
            self.occluded_head = SegmentationHeadOccludedCLS(self.feature // 2, self.feature // 2, class_num, dilations)

        # 🎯 【核心修改点 2】将关系嵌入层从 L3 提拔到 L2 视图
        self.context_prior = context_prior
        if context_prior:
            # 以前输入的是 self.feature * 4 (L3), 现在改听 L2 的特征 self.feature * 2
            # 传入的尺寸强制指定为 1/4 尺度的 size_l2 (即 20x20x12)
            self.CP_mega_voxels = CPMegaVoxels(
                self.feature * 2, size_l2, bn_momentum=bn_momentum
            )

    def forward(self, input_dict):
        res = {}
        x3d_l1 = input_dict["x3d"]       # 1/2 尺度 特征
        x3d_l2 = self.process_l1(x3d_l1) # 1/4 尺度 特征
        x3d_l3 = self.process_l2(x3d_l2) # 1/8 尺度 特征

        # 🎯 【核心修改点 3】在 L2 阶段截获并计算长距离关系损失
        if self.context_prior:
            # 让关系网络去吃 1/4 尺度的 x3d_l2 特征！
            ret = self.CP_mega_voxels(x3d_l2)
            x3d_l2 = ret["x"]            # 关系增强后的 1/4 特征
            for k in ret.keys():
                res[k] = ret[k]          # 此时导出的 P_logits 物理尺寸天然就是 [B, 4, 4800, 600]！

        # 解码器金字塔融合
        x3d_up_l2 = self.up_13_l2(x3d_l3) + x3d_l2
        x3d_up_l1 = self.up_12_l1(x3d_up_l2) + x3d_l1
        x3d_up_lfull = self.up_l1_lfull(x3d_up_l1)

        if not self.infer_mode:
            res["x3d_l1"] = x3d_up_l1
            res["x3d_l2"] = x3d_up_l2
            res["x3d_l3"] = x3d_l3

        # 分割头输出
        if self.cascade_cls:
            ssc_logit_full, ssc_logit_full_occ = self.ssc_head(x3d_up_lfull)
            res["ssc_logit"] = ssc_logit_full
            if not self.infer_mode: res["occ_logit"] = ssc_logit_full_occ
        else:
            res["ssc_logit"] = self.ssc_head(x3d_up_lfull)

        if self.occluded_cls:
            occluded_logit_full = self.occluded_head(x3d_up_lfull)
            if not self.infer_mode: res["occluded_logit"] = occluded_logit_full
            
        return res