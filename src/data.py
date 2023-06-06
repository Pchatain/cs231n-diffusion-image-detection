"""
This is a file with utility functions for loading and processing data.
"""

import os
import glob
import numpy as np
import torch


def load_training_dataset(real_imgs_path="", fake_imgs_path="", balance_datasets=True):
    """
    Given real_imgs and fake_imgs returns a dataset of shape (n_images, 512, 512, 3) and labels of shape (n_images,)

    If empty strings, uses the debug datasets.

    Balances to match the number in the real dataset.
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    if fake_imgs_path == "":
        fake_imgs_path = "debug2"
    fake_dataset_path = os.path.join(current_dir, f'../datasets/midjourney/{fake_imgs_path}/{fake_imgs_path}.pt')
    fake_images = torch.load(fake_dataset_path)

    if real_imgs_path == "":
        real_imgs_path = "laion-art-debug/real_47"
    real_dataset_path = os.path.join(current_dir, f'../datasets/{real_imgs_path}.npy')
    real_images = np.load(real_dataset_path)
    real_images = torch.from_numpy(real_images)

    real_labels = torch.ones(len(real_images))
    if balance_datasets:
        fake_images = fake_images[:len(real_images)]
        fake_labels = torch.zeros(len(fake_images))
    else:
        fake_labels = torch.zeros(len(fake_images))
    images = torch.cat((real_images, fake_images))
    labels = torch.cat((real_labels, fake_labels))
    return images, labels


if __name__ == "__main__":
    print("loading dataset debug")
    dataset = load_training_dataset()
    print(f'Loaded dataset of size {len(dataset)}')
    print(f"shape of example 0 is {dataset[0].shape}")

    # print("lloading dataset mmidjourney")
    # dataset = load_midjourney_dataset()
    # print(f'Loaded dataset of size {len(dataset)}')
    # print(f"shape of example 0 is {dataset[0].shape}")
    # print("...............................")
    # print("loading dataset real")
    # dataset = load_real_dataset(data_path="laion-art-debug/real_658.npy")
    # print(f'Loaded dataset of size {len(dataset)}')
    # print(f"shape of example 0 is {dataset[0].shape}")