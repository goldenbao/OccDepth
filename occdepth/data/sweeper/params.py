import numpy as np
import torch


# sweeper_class_names = [
#     "empty",
#     "floor",
#     "wall",
#     "chair",
#     "table",
#     "sofa",
#     "cabinet",
#     "bed",
#     "wire",
#     "shoe",
#     "cloth",
#     "trashcan",
#     "charging dock",
#     "fan",
#     "human",
#     "pet",
#     "pet_waste",
#     "liquid",
#     "carpet",
#     "plant",
#     "Stain",
#     "paper or tissue",
#     "building blocks",
#     "other object",
# ]

sweeper_class_names = [
    "empty",
    "floor",
    "wall",
    "wire",
    "shoe",
    "pet",
    "pet_waste",
    "carpet",
    "paper or tissue",
    "building blocks",
    "other object",
]

# Per-class voxel counts computed from 16750 label files across all data roots.
# Used for frequency-based class weighting (class_weight_mode="frequency").
# Class 6 (pet_waste) has zero voxels in the current dataset.
sweeper_class_frequencies = np.array([
    3247895988,  # 0  empty
    35567452,    # 1  floor
    17088224,    # 2  wall
    337580,      # 3  wire
    5923,        # 4  shoe
    746,         # 5  pet
    0,           # 6  pet_waste
    200153,      # 7  carpet
    453345,      # 8  paper or tissue
    7105,        # 9  building blocks
    4033218,     # 10 other object
])

sweeper_class_weights = torch.FloatTensor([0.05, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])