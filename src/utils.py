"""
Utiility functions for the project
"""


import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import einops

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
    parser.add_argument('--tsne', action='store_true', help='plot tsne')

    args = parser.parse_args()
    return args


def create_tsne_plot(images, labels):
    """
    Images come in with shape (n_images, 3, 512, 512)
    """
    # Convert images to numpy array
    images_np = images.numpy()
    images_np = einops.rearrange(images_np, 'n c h w -> n (c h w)')

    # Apply TSNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(images_np)

    # Get class labels as numpy array
    labels_np = labels.numpy()

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels_np, cmap='coolwarm')
    plt.colorbar(ticks=[0, 1])
    plt.title('TSNE Plot of the real and fake images')
    plt.xlabel('TSNE Dimension 1')
    plt.ylabel('TSNE Dimension 2')
    plt.show()

    # save image
    plt.savefig('tsne_raw_images.png')