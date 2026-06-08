import os
import shutil
import random
import re
from pathlib import Path
import glob
import numpy as np
from tqdm import tqdm
from occdepth.data.NYU.preprocess import _downsample_label

def batch_process(target_dir):
    # 🎯 指定目标目录
   
    # 获取目录下所有的 .npy 文件，但排除掉已经是 _1_4 的文件
    all_files = glob.glob(os.path.join(target_dir, "*.npy"))
    npy_files = [f for f in all_files if not f.endswith("_1_4.npy")]
    
    print(f"🚀 找到待处理的原始体素文件共: {len(npy_files)} 个")
    
    # 开始循环处理，带进度条
    for file_path in tqdm(npy_files, desc="Processing Voxel Downsampling"):
        # 拆分文件名和后缀
        file_dir, file_name = os.path.split(file_path)
        name_without_ext, ext = os.path.splitext(file_name)
        
        # 拼接出新的文件名，例如: xxx_1_4.npy
        new_file_name = f"{name_without_ext}_1_4{ext}"
        new_file_path = os.path.join(file_dir, new_file_name)
        
        # ⚡️ 检查是否已经存在处理好的文件，存在则直接跳过（方便断点续传）
        if os.path.exists(new_file_path):
            continue
            
        try:
            # 1. 读取原始数据
            voxel_gt = np.load(file_path)
            
            # 🎯 【新增：清洗 255 标签】
            # 找出所有等于 255 的地方，强行赋值为 0
            if np.any(voxel_gt == 255):
                voxel_gt[voxel_gt == 255] = 0
                
                # 🎯 【核心要求：覆盖原始 npy 文件】
                # 清洗完数据后立即写回，确保原始资产的数据纯净
                np.save(file_path, voxel_gt)
            
            # 获取当前体素的实际 shape 作为函数的输入尺寸参数
            current_shape = voxel_gt.shape
            
            # 2. 执行下采样
            voxel_downsampled = _downsample_label(
                label=voxel_gt, 
                voxel_size=current_shape, 
                downscale=4
            )
            
            # 3. 保存新文件
            np.save(new_file_path, voxel_downsampled)
            
        except Exception as e:
            print(f"\n❌ 处理文件时发生错误 {file_name}: {str(e)}")


# ==================== 配置路径 ====================
BASE_DIR = Path("/home/data/OCC/OccData/sweeper_data/beidong_fanwuti/white_tiles/light_Advanced_2")

# 源文件夹路径
SRC_OCC = BASE_DIR / "occupancy_gt"
SRC_LEFT = BASE_DIR / "left_sync"
SRC_RIGHT = BASE_DIR / "right_sync"
SRC_DEPTH = BASE_DIR / "depth_maps"

# 目标文件夹路径
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"

# 划分比例
TRAIN_RATIO = 0.9  # 90% 训练集，10% 测试集
# ==================================================

def setup_directories():
    """创建训练集和测试集的目录结构"""
    for mode in ["train", "test"]:
        (BASE_DIR / mode / "occupancy_gt").mkdir(parents=True, exist_ok=True)
        (BASE_DIR / mode / "left_sync").mkdir(parents=True, exist_ok=True)
        (BASE_DIR / mode / "right_sync").mkdir(parents=True, exist_ok=True)
        (BASE_DIR / mode / "depth_maps").mkdir(parents=True, exist_ok=True)

def extract_timestamp(filename):
    """从文件名中提取核心时间戳（例如从 SLAM_SLAM_L_TX0_293443_640X480_occ_gt.npy 中提取 293443）"""
    match = re.search(r'TX0_(\d+)_640X480', filename)
    return match.group(1) if match else None

def split_dataset():
    # 1. 初始化目录
    setup_directories()
    
    # 2. 以 occupancy_gt 为基准获取所有样本
    occ_files = sorted(list(SRC_OCC.glob("*_occ_gt.npy")))
    if not occ_files:
        print(f"❌ 错误：在 {SRC_OCC} 下没有找到 _occ_gt.npy 文件！")
        return

    print(f"邻居发现：共找到 {len(occ_files)} 帧数据。")

    # 3. 随机打乱并划分
    random.seed(42)  # 固定随机种子，确保如果运行多次，结果是可复现的
    random.shuffle(occ_files)
    
    split_idx = int(len(occ_files) * TRAIN_RATIO)
    train_occ_files = occ_files[:split_idx]
    test_occ_files = occ_files[split_idx:]
    
    print(f"📊 划分结果：训练集 (Train): {len(train_occ_files)} 帧 | 测试集 (Test): {len(test_occ_files)} 帧")

    # 4. 执行联动拷贝
    for mode, file_list in [("train", train_occ_files), ("test", test_occ_files)]:
        print(f"\n正在拷贝 {mode} 集合的数据...")
        copied_count = 0
        missing_count = 0

        for occ_path in file_list:
            timestamp = extract_timestamp(occ_path.name)
            if not timestamp:
                print(f"⚠️ 无法解析文件名的组件时间戳: {occ_path.name}")
                continue
            
            # 构建对应的 4 个文件的路径
            files_to_copy = {
                "occupancy_gt": (occ_path, BASE_DIR / mode / "occupancy_gt" / occ_path.name),
                "left_sync": (SRC_LEFT / f"SLAM_SLAM_L_TX0_{timestamp}_640X480.jpg", BASE_DIR / mode / "left_sync" / f"SLAM_SLAM_L_TX0_{timestamp}_640X480.jpg"),
                "right_sync": (SRC_RIGHT / f"SLAM_SLAM_R_TX0_{timestamp}_640X480.jpg", BASE_DIR / mode / "right_sync" / f"SLAM_SLAM_R_TX0_{timestamp}_640X480.jpg"),
                "depth_maps": (SRC_DEPTH / f"SLAM_SLAM_L_TX0_{timestamp}_640X480_depth_meter.npy", BASE_DIR / mode / "depth_maps" / f"SLAM_SLAM_L_TX0_{timestamp}_640X480_depth_meter.npy")
            }

            # 检查四胞胎文件是否都存在
            all_exist = True
            for name, (src, _) in files_to_copy.items():
                if not src.exists():
                    print(f"❌ 配对文件缺失！找不到 {name} -> {src.name}")
                    all_exist = False
            
            if not all_exist:
                missing_count += 1
                continue  # 只要漏了一个，这一帧就废了，跳过

            # 批量执行拷贝
            for src, dst in files_to_copy.values():
                shutil.copy2(src, dst)  # copy2 会保留文件的原始元数据（修改时间等）
            
            copied_count += 1
            if copied_count % 100 == 0:
                print(f" -> 已成功同步拷贝 {copied_count} 帧...")

        print(f"✨ {mode} 集合处理完毕！成功拷贝: {copied_count} 帧" + (f"，因文件不全跳过: {missing_count} 帧" if missing_count > 0 else ""))

if __name__ == "__main__":
    # split_dataset()
    # print("\n🎉 数据集完美划分并对齐拷贝完成！")
    
    #下采样 target 
    
    # TEST_DIR
    target_dir = TEST_DIR / "occupancy_gt" #/home/data/OCC/OccData/sweeper_data/low/wood_floor/475+6207_sun/test/occupancy_gt"
    batch_process(target_dir)
    print("🏁 所有文件下采样处理完成！")
    