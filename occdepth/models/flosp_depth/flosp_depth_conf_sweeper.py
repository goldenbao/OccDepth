final_dim = (480, 640)

flosp_depth_conf = {
    "x_bound": [0.1, 0.9, 0.01], # 前方 0.8m
    "y_bound": [-0.4, 0.4, 0.01],
    "z_bound": [-0.1, 0.38, 0.01],
    "d_bound": [0.1, 0.9, 0.01],     # 深度预测范围
    "final_dim": final_dim,     # 输入图像尺寸
    "output_channels": 64,     # feature channels
    "downsample_factor": 8,    
    "depth_net_conf": dict(in_channels=64,mid_channels=128),
    "disc_cfg": dict(mode="LID"),    # depth discretization
    "agg_voxel_mode": "mean",     # voxel聚合方式
}