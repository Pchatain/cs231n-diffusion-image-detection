"""
This is a file with utility functions for loading and processing data.
"""

import os
import glob
import numpy as np
import torch


def load_midjourney_dataset(dataset_name="debug_dataset"):
    """
    Loads a midjourney dataset from .pt file

    Dataset has shape (n_images, 512, 512, 3)
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(current_dir, f'../datasets/midjourney/{dataset_name}/{dataset_name}.pt')
    dataset = torch.load(dataset_path)
    return dataset
    
def load_real_dataset(dataset_name="laion-art-debug"):
    """
    Loads a LaION dataset from .npy file. These are real images
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(current_dir, f'../datasets/laion-art-debug/real_47.npy')
    dataset = np.load(dataset_path)
    return dataset

if __name__ == "__main__":
    print("lloading dataset mmidjourney")
    dataset = load_midjourney_dataset()
    print(f'Loaded dataset of size {len(dataset)}')
    print(f"shape of example 0 is {dataset[0].shape}")
    print("...............................")
    print("loading dataset real")
    dataset = load_real_dataset()
    print(f'Loaded dataset of size {len(dataset)}')
    print(f"shape of example 0 is {dataset[0].shape}")