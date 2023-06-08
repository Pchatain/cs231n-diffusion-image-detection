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

import einops

from data import load_training_dataset

from utils import get_training_args

WANDB_PROJECT_NAME = "cs231n"

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(512*512*3, 2)
    
    def forward(self, x):
        x = x.view(-1, 512*512*3)
        # set dtype to float32
        x = x.type(torch.float32)
        x = self.linear(x)
        return x


class Trainer:
    """
    A class for training a pre-trained model
    """

    def __init__(self, model_ft, dataloaders, args) -> None: #log_all_images, model_name="resnet") -> None:
        self.dataloaders = dataloaders
        self.model_ft = model_ft
        self.log_all_images = args.log_all_images
        self.model_name = args.model
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

        # Observe that all parameters are being optimized
        self.lr = args.lr #default 0.001
        if args.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(self.model_ft.parameters(), lr=self.lr, momentum=0.9)
        elif args.optimizer.lower() == "adagrad":
            self.optimizer = optim.Adagrad(self.model_ft.parameters(), lr=self.lr)
        elif args.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(self.model_ft.parameters(), lr=self.lr)

        # Decay LR by a factor of 0.1 every 7 epochs
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)


        if "resnet" in self.model_name:
            num_ftrs = self.model_ft.fc.in_features
            # Here the size of each output sample is set to 2.
            # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
            self.model_ft.fc = nn.Linear(num_ftrs, 2)
        elif 'efficientnet' in self.model_name:
            num_ftrs = self.model_ft.classifier[-1].in_features
            self.model_ft.classifier[-1] = nn.Linear(num_ftrs, 2)
        elif 'vit' in self.model_name:
            num_ftrs = self.model_ft.heads.head.in_features
            self.num_classes = 2
            self.model_ft.heads.head = nn.Linear(num_ftrs, self.num_classes)
            if isinstance(self.model_ft.heads.head, nn.Linear):
                nn.init.zeros_(self.model_ft.heads.head.weight)
                nn.init.zeros_(self.model_ft.heads.head.bias)



    def log_images(self, inputs, labels, preds, epoch):
        """
        Args:
            inputs: Batch of images
            labels: Ground truth labels
            preds: Predicted labels
            epoch: Epoch number
        """
        image_log = []
        n_incorrect = 0
        n_total = 0
        for i, image in enumerate(inputs):
            test_image = image.cpu().numpy()
            einops_image = einops.rearrange(test_image, "c h w -> h w c")
            n_total += 1
            if self.log_all_images or labels[i] != preds[i]:
                image_log.append(
                    wandb.Image(
                        einops_image,
                        caption=f"Label: {self.class_names[labels[i]]}, Predicted: {self.class_names[preds[i]]}",
                    )
                )
                n_incorrect += 1
        print(f"Logging {n_incorrect} images to wandb for epoch {epoch}. There were {n_incorrect} incorrect out of {n_total} ...")
        print(f"Accuracy: {1 - n_incorrect/n_total}")
        wandb.log(
            {
                "image": image_log,
                "epoch": epoch,
                "predicted_label": preds,
                "ground_truth_label": labels,
            },
        )

    def one_step(self, model, log_dict, phase, best_model_params_path="", epoch=-1, best_acc=0.0):
        """
        Args:
            model: Model to train
            log_dict: Dictionary to log to wandb
            phase: Phase of training (train or val or test)
            best_model_params_path: Path to save best model parameters
            epoch: Epoch number
        """
        if phase == "train":
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        # running_corrects = 0
        all_preds = []
        all_labels = []
        all_inputs = []
        running_total_size = 0

        # Iterate over data.
        for inputs, labels in self.dataloaders[phase]:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                labels = labels.type(torch.long)
                loss = self.criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    self.optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            # running_corrects += torch.sum(preds == labels.data)
            # add to all preds and labels on cpu
            all_preds += preds.cpu().numpy().tolist()
            all_labels += labels.cpu().numpy().tolist()
            if phase != "train":
                all_inputs += inputs.cpu().numpy().tolist()

            running_total_size += len(labels)

        if phase == "train":
            self.scheduler.step()

        epoch_loss = running_loss / running_total_size
        epoch_acc = torch.mean(
            (torch.tensor(all_preds) == torch.tensor(all_labels)).float()
        )
        epoch_f1 = f1_score(all_labels, all_preds, average="macro")

        print(
            f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}"
        )

        log_dict[f"{phase}_loss"] = epoch_loss
        log_dict[f"{phase}_acc"] = epoch_acc
        log_dict[f"{phase}_f1"] = epoch_f1

        # log images to wandb
        if phase != "train":
            self.log_images(inputs, all_labels, all_preds, epoch)

        # deep copy the model
        if phase == "val" and epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), best_model_params_path)

        return best_acc


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
            for i, (inputs, labels) in enumerate(self.dataloaders["val"]):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images // 2, 2, images_so_far)
                    ax.axis("off")
                    ax.set_title(f"predicted: {self.class_names[preds[j]]}")
                    self.imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)


    def train_model(self, epochs=25):
        """
        Trains a model for a given number of epochs
        """
        self.model_ft = self.model_ft.to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        since = time.time()

        # Create a temporary directory to save training checkpoints
        model = self.model_ft
        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

            torch.save(model.state_dict(), best_model_params_path)
            best_acc = 0.0

            for epoch in range(epochs):
                print(f"Epoch {epoch}/{epochs - 1}")
                print("-" * 10)

                # Each epoch has a training and validation phase
                log_dict = {}
                for phase in ["train", "val"]:
                    best_acc = self.one_step(model, log_dict, phase=phase, best_model_params_path=best_model_params_path, epoch=epoch, best_acc=best_acc)

                print()
                wandb.log(log_dict, commit=True)

            time_elapsed = time.time() - since
            print(
                f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
            )
            print(f"Best val Acc: {best_acc:4f}")

            # load best model weights
            model.load_state_dict(torch.load(best_model_params_path))
        self.model_ft = model

        # evauluate model on test set
        test_dict = {}
        self.one_step(self.model_ft, test_dict, phase="test", epoch=epochs)
        wandb.log(test_dict, commit=True)

        # self.visualize_model()


def main():
    args = get_training_args()

    images, labels = load_training_dataset(
        real_imgs_path=args.real, fake_imgs_path=args.fake
    )
    print(f"Loaded dataset of size {len(images)}")

    # Initialize Weights and Biases run
    run = wandb.init(
        # entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        notes=args.notes,
        save_code=True,
        config=args,
    )
    assert run is not None

    # split into train and val and test
    train_frac, val_frac, test_frac = args.train_frac, args.val_frac, 1 - args.train_frac - args.val_frac
    print(f'train_frac: {train_frac}, val_frac: {val_frac}, test_frac: {test_frac}')
    train_size = int(train_frac * len(images))
    val_size = int(val_frac * len(images))
    test_size = len(images) - train_size - val_size
    
    full_dataset = torch.utils.data.TensorDataset(images, labels)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(0)
    )
    # for ds in [val_dataset, test_dataset]:
    #     ctr = collections.Counter()
    #     for _, target in ds:
    #         ctr[target] += 1
    #     print("ctr is", ctr)
    # assert(False)

    # create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True
    )
    dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}

    # instantiate model
    if args.model == "logistic_regression":
        model_ft = LogisticRegression()
    elif args.model == "resnet18":
        model_ft = models.resnet18(weights="IMAGENET1K_V1")
    elif args.model == "resnet34":
        model_ft = models.resnet34(weights="IMAGENET1K_V1")
    elif args.model == 'efficientnet_b0':
        model_ft = models.efficientnet_b0(weights='DEFAULT')
    elif args.model == 'vit':
        model_ft = models.vit_b_16(weights='DEFAULT')
    else:
        raise ValueError(f"Unknown model type {args.model}")
    trainer = Trainer(model_ft, dataloaders, args) #log_all_images=args.log_all_images, model_name=args.model)
    model_ft = trainer.train_model(args.epochs)


if __name__ == "__main__":
    main()
