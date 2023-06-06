"""
This file loads in the datasets and trains a logistic regression model to classify the images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import argparse

from data import load_training_dataset

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(512*512*3, 1)
    
    def forward(self, x):
        x = x.view(-1, 512*512*3)
        # set dtype to float32
        x = x.type(torch.float32)
        x = self.linear(x)
        return x
    
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx} Loss {loss.item()}')

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target.unsqueeze(1).float()).item()
            pred = torch.round(torch.sigmoid(output))
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)}%)')

def main():
    """
    By default trains a logistic regression classifier on the debug dataset
    """
    parser = argparse.ArgumentParser(description='Train a logistic regression classifier on the midjourney dataset')
    parser.add_argument('--real', type=str, default="", help='path to real images')
    parser.add_argument('--fake', type=str, default="", help='path to fake images')
    args = parser.parse_args()

    BATCH_SIZE = 32
    images, labels = load_training_dataset(real_imgs_path=args.real, fake_imgs_path=args.fake)
    print(f'Loaded dataset of size {len(images)}')

    # split into train and test
    train_size = int(0.8 * len(images))
    test_size = len(images) - train_size
    full_dataset = torch.utils.data.TensorDataset(images, labels)
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])


    # create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # create model
    model = LogisticRegression()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model created on device {device}")

    # create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    # train and test
    for epoch in range(10):
        print(f'Epoch {epoch}')
        train(model, train_loader, optimizer, criterion, device)
        test(model, test_loader, criterion, device)
    print(f"Now testing on the train set")
    test(model, train_loader, criterion, device)
    
    # save model
    torch.save(model.state_dict(), 'logistic_regression.pt')

if __name__ == '__main__':
    main()