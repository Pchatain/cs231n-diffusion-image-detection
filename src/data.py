"""
This is a file with utility functions for loading and processing data.
"""

import os
import glob
import numpy as np
import torch


def load_midjourney_dataset(dataset_name=""):
    """
    Loads a midjourney dataset from .pt file

    Dataset has shape (n_images, 512, 512, 3) and is a tensor
    """
    if dataset_name == "":
        dataset_name = "debug_dataset"
    current_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(current_dir, f'../datasets/midjourney/{dataset_name}/{dataset_name}.pt')
    dataset = torch.load(dataset_path)
    return dataset
    
def load_real_dataset(dataset_name=""):
    """
    Loads a LaION dataset from .npy file. These are real images.
    return a tensor of shape (n_images, 512, 512, 3)
    """
    if dataset_name == "":
        dataset_name = "laion-art-debug"
    current_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(current_dir, f'../datasets/laion-art-debug/real_47.npy')
    dataset = np.load(dataset_path)
    # convert to tensor
    dataset = torch.from_numpy(dataset)
    return dataset


def load_training_dataset(real_imgs_path="", fake_imgs_path=""):
    """
    Given real_imgs and fake_imgs returns a dataset of shape (n_images, 512, 512, 3) and labels of shape (n_images,)

    If empty strings, uses the debug datasets
    """
    real_images = load_real_dataset(real_imgs_path)
    fake_images = load_midjourney_dataset(fake_imgs_path)
    real_labels = torch.ones(len(real_images))
    fake_labels = torch.zeros(len(fake_images))
    images = torch.cat((real_images, fake_images))
    labels = torch.cat((real_labels, fake_labels))
    return images, labels


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