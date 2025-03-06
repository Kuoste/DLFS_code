# imports

import torch
import torch.optim as optim

import numpy as np
from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from dlfs_kuoste import pytorch

torch.manual_seed(20190325);

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1305,), (0.3081,))
])

# Load MNIST data
# https://pytorch.org/docs/stable/data.html
train_dataset = MNIST(root='../mnist_data/',
                      train=True, 
                      download=True,
                      transform=img_transforms)

test_dataset = MNIST(root='../mnist_data/',
                     train=False, 
                     download=True,
                     transform=img_transforms)

# Create data loaders
# https://pytorch.org/docs/stable/data.html
train_loader = DataLoader(dataset=train_dataset,
                                           batch_size=60, 
                                           shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                                           batch_size=60, 
                                           shuffle=False)

class ConvNet(pytorch.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = pytorch.ConvLayer(1, 14, 5, activation=nn.Tanh(),
                               dropout=0.8)
        self.conv2 = pytorch.ConvLayer(14, 7, 5, activation=nn.Tanh(), flatten=True,
                               dropout=0.8)
        self.dense1 = pytorch.DenseLayer(28 * 28 * 7, 32, activation=nn.Tanh(),
                                 dropout=0.8)
        self.dense2 = pytorch.DenseLayer(32, 10)

    def forward(self, x: Tensor) -> Tensor:
        pytorch.assert_dim(x, 4)
            
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.dense1(x)
        x = self.dense2(x)
        return x,


# Model instantiation
model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Trainer instantiation
trainer = pytorch.Trainer(model, optimizer, criterion)

# Training
trainer.fit(train_dataloader = train_loader,        
            test_dataloader = test_loader,
            epochs = 1,
            eval_every = 1)

def test_accuracy(model):
    model.eval()
    accuracies = []
    for X_batch, y_batch in test_loader:
        output = model(X_batch)[0]
        accuracy_batch = (torch.max(output, dim=1)[1] == y_batch).type(torch.float32).mean().item()
        accuracies.append(accuracy_batch)
    return torch.Tensor(accuracies).mean().item()

print("The accuracy of the model is", test_accuracy(model))

