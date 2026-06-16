"""
Remap occupancy GT labels from full 22 categories to categories_lite.

Reads existing *_occ_gt.npy files from {base_dir}/occupancy_gt/,
remaps labels according to config's categories → categories_lite mapping,
and saves results to {base_dir}/occupancy_gt_lite/.

Original files are NOT modified.
"""

import os
import sys
import yaml
import argparse
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def build_label_map(categories, categories_lite):
    """
    Build a 256-element uint8 lookup table.

    Args:
        categories: Full list of category names (22 items).
        categories_lite: Lite list of category names (10 items).

    Returns:
        label_map: ndarray (256,) uint8 — label_map[old_label] = new_label
    """
    n_lite = len(categories_lite)
    label_others = n_lite + 1  # new LABEL_OTHERS

    # Default: all labels (0-255) map to themselves
    label_map = np.arange(256, dtype=np.uint8)

    # FREE (0) → 0, UNKNOWN (255) → 255 (already correct by identity init)

    # Build lookup: category_name → new_label_id
    lite_lookup = {}
    for i, cat in enumerate(categories_lite):
        lite_lookup[cat] = i + 1  # labels start at 1

    # Remap each original category
    for old_id, cat in enumerate(categories, start=1):
        if cat in lite_lookup:
            label_map[old_id] = lite_lookup[cat]
        else:
            label_map[old_id] = label_others

    # Original LABEL_OTHERS (len(categories)+1) → new LABEL_OTHERS
    label_map[len(categories) + 1] = label_others

    return label_map


def main():
    parser = argparse.ArgumentParser(
        description="Remap occupancy GT labels from full categories to categories_lite"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="occdepth/data/sweeper/config.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    categories = config['semantic']['categories']
    categories_lite = config['semantic']['categories_lite']

    logging.info(f"Original categories ({len(categories)}): {categories}")
    logging.info(f"Lite categories ({len(categories_lite)}): {categories_lite}")

    label_others_lite = len(categories_lite) + 1
    logging.info(f"New LABEL_OTHERS = {label_others_lite}")

    # Build mapping
    label_map = build_label_map(categories, categories_lite)

    non_identity = np.where(label_map != np.arange(256, dtype=np.uint8))[0]
    logging.info(f"Label mapping (non-identity entries):")
    for old in non_identity:
        logging.info(f"  {old:3d} → {label_map[old]:3d}")

    # Directories
    base_dir = config['output']['base_dir']
    src_dir = os.path.join(base_dir, "occupancy_gt")
    dst_dir = os.path.join(base_dir, "occupancy_gt_lite")

    if not os.path.isdir(src_dir):
        logging.error(f"Source directory not found: {src_dir}")
        sys.exit(1)

    os.makedirs(dst_dir, exist_ok=True)

    # Find all occ_gt files
    npy_files = sorted(Path(src_dir).glob("*.npy"))
    if not npy_files:
        logging.error(f"No *_occ_gt.npy files found in {src_dir}")
        sys.exit(1)

    logging.info(f"\nFound {len(npy_files)} occupancy GT files to remap")
    logging.info(f"Source: {src_dir}")
    logging.info(f"Output: {dst_dir}\n")

    total_remapped = 0
    for npy_path in npy_files:
        try:
            occ = np.load(npy_path)
            occ_remapped = label_map[occ]

            dst_path = Path(dst_dir) / npy_path.name
            np.save(dst_path, occ_remapped.astype(np.uint8))

            # Count how many voxels changed
            changed = np.sum(occ != occ_remapped)
            total_remapped += int(changed)
            if changed > 0:
                logging.info(f"  {npy_path.name}: {changed} voxels remapped")

        except Exception as e:
            logging.error(f"  Failed to process {npy_path.name}: {e}")

    logging.info(f"\nDone. {len(npy_files)} files processed, {total_remapped} total voxels remapped.")
    logging.info(f"Output directory: {dst_dir}")


if __name__ == "__main__":
    main()
