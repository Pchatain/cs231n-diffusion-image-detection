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
from tqdm import tqdm
from tempfile import TemporaryDirectory
import os
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, models, transforms
import wandb
from sklearn.model_selection import KFold

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
        tqdm.write(f"Logging {len(image_log)} images to wandb for epoch {epoch}. There were {n_incorrect} incorrect out of {n_total} ...")
        tqdm.write(f"Accuracy: {1 - n_incorrect/n_total}")
        wandb.log(
            {
                "image": image_log,
                "epoch": epoch,
                "predicted_label": preds,
                "ground_truth_label": labels,
            },
        )

    def one_step(self, model, log_dict, phase, best_model_params_path="", epoch=-1, best_f1=0.0):
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

        tqdm.write(
            f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}"
        )

        log_dict[f"{phase}_loss"] = epoch_loss
        log_dict[f"{phase}_acc"] = epoch_acc
        log_dict[f"{phase}_f1"] = epoch_f1

        # log images to wandb
        if phase != "train":
            self.log_images(inputs, all_labels, all_preds, epoch)

        # deep copy the model
        if phase == "val" and epoch_f1 > best_f1:
            best_f1 = epoch_f1
            torch.save(model.state_dict(), best_model_params_path)

        return epoch_acc, best_f1


    def train_model(self, epochs=25):
        """
        Trains a model for a given number of epochs
        """
        self.model_ft = self.model_ft.to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        self.optimizer = optim.SGD(self.model_ft.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

        since = time.time()

        # Create a temporary directory to save training checkpoints
        model = self.model_ft
        best_acc = 0.0
        best_f1 = 0.0
        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir, "best_model_params.pt")
            torch.save(model.state_dict(), best_model_params_path)

            for epoch in tqdm(range(epochs), desc="Epochs"):
                # Each epoch has a training and validation phase
                log_dict = {}
                for phase in ["train", "val"]:
                    best_acc, best_f1 = self.one_step(model, log_dict, phase=phase, best_model_params_path=best_model_params_path, epoch=epoch, best_f1=best_f1)

                wandb.log(log_dict, commit=True)

            time_elapsed = time.time() - since
            tqdm.write(f"Best val Acc: {best_acc:4f}")

            # load best model weights
            model.load_state_dict(torch.load(best_model_params_path))
        self.model_ft = model

        # evauluate model on test set
        if "test" in self.dataloaders:
            test_dict = {}
            self.one_step(self.model_ft, test_dict, phase="test", epoch=epochs)
            wandb.log(test_dict, commit=True)
        return best_acc, best_f1
        


def instantiate_model(args):
    """
    Makes a model instance based on the model type
    """
    if args.model == "logistic_regression":
        model_ft = LogisticRegression()
    elif args.model == "resnet18":
        model_ft = models.resnet18(weights="IMAGENET1K_V1")
    elif args.model == "resnet34":
        model_ft = models.resnet34(weights="IMAGENET1K_V1")
    elif args.model == "efficientnet_b0":
        model_ft = models.efficientnet_b0(weights="IMAGENET1K_V1")
    elif "efficientnet" in args.model:
        # patch for misnamed model to default to b0
        model_ft = models.efficientnet_b0(weights="IMAGENET1K_V1")
    else:
        raise ValueError(f"Unknown model type {args.model}")
    return model_ft


def run_cross_validation(args, full_dataset):
    """
    Trains the model using cross validation.
    """
    # Create the KFold object
    kf = KFold(n_splits=args.kfold, shuffle=True)
    # Lists to store the datasets for each fold
    train_datasets, val_datasets = [], []
    # Split the data into k folds
    for train_indices, val_indices in kf.split(full_dataset):
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
    # Loop over each fold
    best_acc = 0.0
    best_f1 = 0.0
    best_model = None
    for fold in tqdm(range(args.kfold), desc="Folds"):
        train_loader = torch.utils.data.DataLoader(
            train_datasets[fold], batch_size=args.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_datasets[fold], batch_size=args.batch_size, shuffle=True
        )
        dataloaders = {"train": train_loader, "val": val_loader}

        model_ft = instantiate_model(args)
        trainer = Trainer(model_ft, dataloaders, args)
        accuracy, f1_score = trainer.train_model(args.epochs)
        if f1_score > best_f1:
            best_acc = accuracy
            best_model = trainer.model_ft
            best_f1 = f1_score
    return best_model, best_acc, best_f1

def main(args):
    # Initialize Weights and Biases run
    run = wandb.init(
        # entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        notes=args.notes,
        save_code=True,
        config=args,
    )
    assert run is not None
    args = wandb.config

    images, labels = load_training_dataset(
        real_imgs_path=args.real, fake_imgs_path=args.fake
    )
    tqdm.write(f"Loaded dataset of size {len(images)}")

    # split into train and val and test
    train_frac, val_frac, test_frac = args.train_frac, args.val_frac, 1 - args.train_frac - args.val_frac
    tqdm.write(f'train_frac: {train_frac}, val_frac: {val_frac}, test_frac: {test_frac}')
    train_size = int(train_frac * len(images))
    val_size = int(val_frac * len(images))
    test_size = len(images) - train_size - val_size

    full_dataset = torch.utils.data.TensorDataset(images, labels)
    best_acc = 0.0
    best_f1 = 0.0
    try: # catches cudaOOM
        if args.kfold > 0:
            tqdm.write(f"Using train and validation split for cross validation. frac: {train_frac + val_frac}")
            train_dataset, test_dataset = torch.utils.data.random_split(
                full_dataset, [train_size + val_size, test_size]
            )
            best_model, best_acc, best_f1 = run_cross_validation(args, train_dataset)
            tqdm.write(f"Final Best accuracy: {best_acc}, best f1: {best_f1}")

        else:
            tqdm.write("Doing normal training with train, val, and test splits.")
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size, test_size]
            )

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

            model_ft = instantiate_model(args)
            trainer = Trainer(model_ft, dataloaders, args)
            best_acc, best_f1 = trainer.train_model(args.epochs)
    except RuntimeError as exc:
        print(f"RuntimeError: {exc}")
        wandb.log({"RuntimeError": str(exc)})
        wandb.log({"final_best_acc": -1, "final_best_f1": -1})

    wandb.log({"final_best_acc": best_acc, "final_best_f1": best_f1})


if __name__ == "__main__":
    args = get_training_args()
    if args.sweep_id != "":
        wandb.agent(
            args.sweep_id,
            function=lambda: main(args),
            # entity=WANDB_ENTITY_NAME,
            project=WANDB_PROJECT_NAME,
            count=args.sweep_count,
        )
    else:
        main(args)
