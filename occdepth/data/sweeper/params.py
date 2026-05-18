import numpy as np
import torch


sweeper_class_names = [
    "empty",
    "floor",
    "wall",
    "chair",
    "table",
    "sofa",
    "cabinet",
    "bed",
    "wire",
    "shoe",
    "cloth",
    "trashcan",
    "charging dock",
    "fan",
    "human",
    "pet",
    "pet_waste",
    "liquid",
    "carpet",
    "plant",
    "Stain",
    "paper or tissue",
    "building blocks",
    "other object",
]

sweeper_class_weights = torch.FloatTensor([0.05, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])