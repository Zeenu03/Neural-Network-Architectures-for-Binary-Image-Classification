"""
Example code of how to use the TensorBoard in PyTorch.
This code uses a lot of different functions from TensorBoard
and tries to have them all in a compact way, it might not be
super clear exactly what calls does what, for that I recommend
watching the YouTube video.

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
*    2020-04-17 Initial coding
*    2022-12-19 Small revision of code, checked that it works with latest PyTorch version
"""

# Imports
import os
import torch
import torchvision
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard

# Simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
import torchvision.transforms as transforms  # Add missing import

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

in_channels = 1
num_classes = 10
num_epochs = 2
learning_rate = 0.001
batch_size = 64

# Downloading the MNIST dataset
if not os.path.exists("datasets"):
    os.makedirs("datasets")
    if not os.path.exists("datasets/MNIST"):
        train_dataset = torchvision.datasets.MNIST(root="datasets", train=True, transform=transforms.ToTensor(), download=True)

# Load Data
train_dataset = torchvision.datasets.MNIST(
    root="datasets", train=True, transform=transforms.ToTensor(), download=False
)
import shutil

if os.path.exists("ES335-Assignment-4-matrix-minds/Tensor_board_tut/runs"):                                   
    shutil.rmtree("runs")
                                            
 # Initialize network
model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
writer = SummaryWriter(f"runs/MNIST/trying_out_tensorboard")
step = 0
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)   
for epoch in range(num_epochs):
    losses = []
    accuracies= []
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        # writer = SummaryWriter(f"runs/MNIST/trying_out_tensorboard")
        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
        
        # Calculate 'running' training accuracy
        _, predictions = scores.max(1)
        num_correct = (predictions == targets).sum()
        running_train_acc = float(num_correct) / float(data.shape[0])
        accuracies.append(running_train_acc) 
        # writer= SummaryWriter()
        # writer.add_scaler('Training loss', loss, global_step=step)
        # writer.add_scaler('Training Accuracy', running_train_acc, global_step=step)
        # writer.add_image('mnist_images', data, global_step=step)
        step += 1
        writer.close()
        