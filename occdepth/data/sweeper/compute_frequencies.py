"""Scan all occ_gt label files under data roots and compute per-class voxel frequencies.

Run:
  python occdepth/data/sweeper/compute_frequencies.py

Results are printed as a numpy array ready to copy into params.py.
"""

import glob
import os

import numpy as np

# Match the data roots from config
DATA_ROOTS = [
    "/home/data/OCC/OccData/sweeper_data/beidong_fanwuti/white_tiles/light_Advanced_2",
    "/home/data/OCC/OccData/sweeper_data/low/wood_floor/475+6207_sun",
]

N_CLASSES = 11


def main():
    total_counts = np.zeros(N_CLASSES, dtype=np.int64)
    total_invalid = 0
    total_voxels = 0
    n_files = 0

    for root in DATA_ROOTS:
        pattern = os.path.join(root, "**", "*occ_gt.npy")
        files = glob.glob(pattern, recursive=True)
        print(f"{root}: found {len(files)} label files")

        for fp in files:
            labels = np.load(fp)  # (80, 80, 48) uint8
            counts = np.bincount(labels.ravel(), minlength=N_CLASSES)
            total_counts += counts[:N_CLASSES]
            invalid = np.sum(labels == 255)
            total_invalid += invalid
            total_voxels += labels.size
            n_files += 1

    valid_voxels = total_voxels - total_invalid
    print(f"\nScanned {n_files} files, {total_voxels} total voxels")
    print(f"Invalid (255): {total_invalid} ({100*total_invalid/total_voxels:.2f}%)")
    print(f"Valid voxels:  {valid_voxels} ({100*valid_voxels/total_voxels:.2f}%)")
    print()

    # Print per-class stats
    print(f"{'Class':>3s} {'Count':>15s} {'% Valid':>10s} {'% Non-empty':>12s}")
    nonempty_total = total_counts[1:].sum()
    for i in range(N_CLASSES):
        pct_valid = 100 * total_counts[i] / max(valid_voxels, 1)
        pct_nonempty = 100 * total_counts[i] / max(nonempty_total, 1)
        print(f"{i:3d} {total_counts[i]:15d} {pct_valid:9.4f}% {pct_nonempty:11.4f}%")
    print(f"\nNon-empty total: {nonempty_total}")

    # Print numpy array for params.py
    weights = 1.0 / np.log(np.array(total_counts, dtype=np.float64) + 0.001)
    print("\n=== Copy into params.py ===")
    print(f"sweeper_class_frequencies = np.array([")
    for i in range(N_CLASSES):
        suffix = "," if i < N_CLASSES - 1 else ",])\n"
        print(f"    {total_counts[i]:>15d},{suffix}")
    print(f"Copy weights (1/log(freq+0.001)):")
    print(f"# sweeper_class_weights_frequency = torch.FloatTensor([")
    for i in range(N_CLASSES):
        suffix = "," if i < N_CLASSES - 1 else ",])"
        print(f"    {weights[i]:.6f},{suffix}")


if __name__ == "__main__":
    main()
