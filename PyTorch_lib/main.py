import numpy as np
from PIL import Image 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

import torchvision 
import torchvision.transforms as transforms

transform = transforms.compos([
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_data,batch_size=32, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=32, shuffle=True, num_workers=2)

image, label = train_data[0]
image.size()

class_names = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

class NeuralNet(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(3, 12, 5) # (12, 28, 28)
    self.pool = nn.MaxPool2d(2, 2) # (12, 14, 14)
    self.conv2 = nn.Conv2d(12, 24, 5) # (24, 10, 10) -> (24, 5, 5)
    self.fcl = nn.Linear(24 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)
  
  def foward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
  
net = NeuralNet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(100):
  print(f'Training epoch {epoch}...')

  running_loss = 0.0

  for i, data in enumerate(train_loader):
    inputs, labels = data

    optimizer.zero_grad()

    outputs = net(inputs)

    loss = loss_function(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

  print(f'Loss: {running_loss / len(train_loader):.4f}')
