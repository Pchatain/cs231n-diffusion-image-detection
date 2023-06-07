"""
This file loads in the datasets and trains a logistic regression model to classify the images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score

import argparse
import plotly.graph_objects as go

from data import load_training_dataset
from utils import get_training_args

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
    """
    Get's the F1 score of the model, as well as the accuracy.
    Returns the F1 score and the accuracy
    """
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target.unsqueeze(1).float()).item()
            pred = torch.round(torch.sigmoid(output))
            correct += pred.eq(target.view_as(pred)).sum().item()

            all_preds.extend(pred.view(-1).tolist())
            all_targets.extend(target.view(-1).tolist())
            
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    f1 = f1_score(all_targets, all_preds)

    print(f'Test set: Average loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy}%), F1 Score: {f1}')

    return f1, accuracy


def main():
    """
    By default trains a logistic regression classifier on the debug dataset
    """
    args = get_training_args()

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
    f1_train_lst, f1_test_lst = [], []
    accuracy_train_lst, accuracy_test_lst = [], []
    for epoch in range(args.epochs):
        print(f'Epoch {epoch}')
        train(model, train_loader, optimizer, criterion, device)
        print("________")
        f1_train, accuracy_train = test(model, train_loader, criterion, device)
        print("________")
        f1_test, accuracy_test = test(model, test_loader, criterion, device)
        print("___________________________")
        f1_train_lst.append(f1_train)
        f1_test_lst.append(f1_test)
        accuracy_train_lst.append(accuracy_train)
        accuracy_test_lst.append(accuracy_test)
    

    # use plotly to plot the f1 scores vs epochs
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(args.epochs)), y=f1_train_lst, mode='lines+markers', name='Train F1 Score'))
    fig.add_trace(go.Scatter(x=list(range(args.epochs)), y=f1_test_lst, mode='lines+markers', name='Test F1 Score'))
    fig.update_layout(title='F1 Score vs Epochs', xaxis_title='Epochs', yaxis_title='F1 Score')
    fig.show()
    # save the fig as a png
    fig.write_image("logistic_regression_f1.png")

    # use plotly to plot the accuracy vs epochs
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(args.epochs)), y=accuracy_train_lst, mode='lines+markers', name='Train Accuracy'))
    fig.add_trace(go.Scatter(x=list(range(args.epochs)), y=accuracy_test_lst, mode='lines+markers', name='Test Accuracy'))
    fig.update_layout(title='Accuracy vs Epochs', xaxis_title='Epochs', yaxis_title='Accuracy')
    fig.show()
    # save the fig as a png
    fig.write_image("logistic_regression_accuracy.png")

    
    # save model
    torch.save(model.state_dict(), 'logistic_regression.pt')

if __name__ == '__main__':
    main()