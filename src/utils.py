"""
Utiility functions for the project
"""


import argparse

def get_training_args():
    """
    Loads the training arguments
    """
    parser = argparse.ArgumentParser(description='Train a logistic regression classifier on the midjourney dataset')
    parser.add_argument('--real', type=str, default="laion-art/real_658", help='path to real images')
    parser.add_argument('--fake', type=str, default="full_660", help='path to fake images')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train for')
    args = parser.parse_args()
    return args