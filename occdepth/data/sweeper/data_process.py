import os
import shutil
import random
import re
from pathlib import Path

# ==================== 配置路径 ====================
BASE_DIR = Path("/home/project/OccData/sweeper_data/low/wood_floor/475+6207_sun")

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
    split_dataset()
    print("\n🎉 数据集完美划分并对齐拷贝完成！")