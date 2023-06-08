"""
This file loads a pre-trained image model and uses this for classification.

Adapted from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
import numpy as np

import time
from tempfile import TemporaryDirectory
import os
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, models, transforms
import wandb

from data import load_training_dataset

from utils import get_training_args

WANDB_PROJECT_NAME = "cs231n"


class Trainer():
    """
    A class for training a pre-trained model
    """
    def __init__(self, model_ft, dataloaders) -> None:
        self.dataloaders = dataloaders
        self.model_ft = model_ft
        # Data augmentation and normalization for training
        # Just normalization for validation
        # data_transforms = {
        #     'train': transforms.Compose([
        #         transforms.RandomResizedCrop(224),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     ]),
        #     'val': transforms.Compose([
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     ]),
        # }
        self.class_names = ["real", "fake"]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def _train_model(self, model, criterion, optimizer, scheduler, num_epochs=25):
        """
        Args:
            model: PyTorch model
            criterion: Loss function
            optimizer: Optimizer for parameters
            scheduler: Instance of ``torch.optim.lr_scheduler``
            num_epochs: Number of epochs
        
        Returns:
            model: Trained Model with best validation accuracy
        """
        since = time.time()

        # Create a temporary directory to save training checkpoints
        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

            torch.save(model.state_dict(), best_model_params_path)
            best_acc = 0.0

            for epoch in range(num_epochs):
                print(f'Epoch {epoch}/{num_epochs - 1}')
                print('-' * 10)

                # Each epoch has a training and validation phase
                log_dict = {}
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()  # Set model to training mode
                    else:
                        model.eval()   # Set model to evaluate mode

                    running_loss = 0.0
                    # running_corrects = 0
                    all_preds = []
                    all_labels = []
                    running_total_size = 0

                    # Iterate over data.
                    for inputs, labels in self.dataloaders[phase]:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            labels = labels.type(torch.long)
                            loss = criterion(outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        # running_corrects += torch.sum(preds == labels.data)
                        # add to all preds and labels on cpu
                        all_preds += preds.cpu().numpy().tolist()
                        all_labels += labels.cpu().numpy().tolist()
                        
                        running_total_size += len(labels)
                    if phase == 'train':
                        scheduler.step()

                    epoch_loss = running_loss / running_total_size
                    epoch_acc = torch.mean((torch.tensor(all_preds) == torch.tensor(all_labels)).float())
                    epoch_f1 = f1_score(all_labels, all_preds, average='macro')

                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}')
                    
                    log_dict[f'{phase}_loss'] = epoch_loss
                    log_dict[f'{phase}_acc'] = epoch_acc
                    log_dict[f'{phase}_f1'] = epoch_f1
                    

                    # deep copy the model
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), best_model_params_path)

                print()
                wandb.log(log_dict)

            time_elapsed = time.time() - since
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Best val Acc: {best_acc:4f}')

            # load best model weights
            model.load_state_dict(torch.load(best_model_params_path))
        self.model_ft = model

    def imshow(self, inp, title=None):
        """Display image for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated


    def visualize_model(self, num_images=6):
        """
        Args:
            model: PyTorch model
            num_images: number of images to display
        """
        model = self.model_ft
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.dataloaders['val']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title(f'predicted: {self.class_names[preds[j]]}')
                    self.imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)


    def train_pretrained_model(self, epochs=25):
        """
        Trains a pretrained resnet
        Resets the last layer to be a linear layer with 2 outputs, the logitts for real class
        and the logits for fake class
        """
        num_ftrs = self.model_ft.classifier[-1].in_features
        self.model_ft.classifier[-1] = nn.Linear(num_ftrs, 2)
        #num_ftrs = self.model_ft.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
        #self.model_ft.fc = nn.Linear(num_ftrs, 2)

        self.model_ft = self.model_ft.to(self.device)

        criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(self.model_ft.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        self._train_model(self.model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=epochs)

        # self.visualize_model()


def main():
    args = get_training_args()

    BATCH_SIZE = 16
    images, labels = load_training_dataset(real_imgs_path=args.real, fake_imgs_path=args.fake)
    print(f'Loaded dataset of size {len(images)}')

    # Initialize Weights and Biases run
    run = wandb.init(
        # entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        notes=args.notes,
        save_code=True,
        config=args,
    )
    assert run is not None

    # split into train and val
    train_size = int(0.8 * len(images))
    val_size = len(images) - train_size
    full_dataset = torch.utils.data.TensorDataset(images, labels)
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])


    # create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dataloaders = {'train': train_loader, 'val': val_loader}

    # instantiate model
    model_ft = models.efficientnet_b0(weights='IMAGENET1K_V1')
    trainer = Trainer(model_ft, dataloaders)
    model_ft = trainer.train_pretrained_model(args.epochs)


if __name__ == '__main__':
    main()