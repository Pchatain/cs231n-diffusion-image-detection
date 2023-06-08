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
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--notes', type=str, default="", help='notes for the run')
    parser.add_argument('--log_all_images', action='store_true', help='log all images to tensorboard')
    parser.add_argument('--model', type=str, default="logistic_regression", help='model to train')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument("--train_frac", type=float, default=0.7, help="fraction of data to use for training")
    parser.add_argument("--val_frac", type=float, default=0.15, help="fraction of data to use for validation")
    parser.add_argument("--optimizer", type=str, default="SGD", help="optimizer for gradient descent")
    parser.add_argument('--kfold', type=int, default=0, help='kfold cross validation')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--sweep_id', type=str, default="", help='sweep id')
    parser.add_argument('--sweep_count', type=int, default=1, help='number of runs to do for this agent')

    args = parser.parse_args()
    return args