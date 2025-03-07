# Load the MNIST dataset and train a CNN model without using the dlfs_kuoste package.
# Study resources: https://pytorch.org/tutorials/beginner/nn_tutorial.html

from typing import Tuple
import time
import torch
import torch.optim as optim
from torch.optim import Optimizer
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

torch.manual_seed(20190325);

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

class Trainer(object):
    def __init__(self, model: nn.Module, optimizer: Optimizer, criterion: _Loss):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion

    def fit(self, train_dataloader: DataLoader, test_dataloader: DataLoader, epochs: int = 100, eval_every: int = 10) -> None:


        for epoch in range(epochs):
            start = time.time()
            self.model.train()

            for X_batch, y_batch in train_dataloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()

            if (eval_every is not None):
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}", end="")

                if epoch % eval_every == 0:
                    total_loss, accuracy = self.evaluate(test_dataloader)
                    print(f", Average loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}", end="")

                end = time.time()
                print(f", ", end='')

            print(f"Time: {end - start:.2f}s")

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = self.model(X_batch)
                total_loss += self.criterion(output, y_batch).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(y_batch.view_as(pred)).sum().item()

        total_loss /= len(dataloader.dataset)
        accuracy = correct / len(dataloader.dataset)
        return total_loss, accuracy

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
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=480, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=480, 
                                           shuffle=False)

# Model definition
class ConvNet(nn.Module):
    def __init__(self) -> None:
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Model instantiation
model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Trainer instantiation
trainer = Trainer(model, optimizer, criterion)

# Stopwatch start
start = time.time()

# Training
trainer.fit(train_dataloader = train_loader,        
            test_dataloader = test_loader,
            epochs = 4,
            eval_every = 2)

# Evaluation
total_loss, accuracy = trainer.evaluate(test_loader)

# Stopwatch end
end = time.time()
print(f"Total execution time: {end - start}, Accuracy: {accuracy:.4f}")
