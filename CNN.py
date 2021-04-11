import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3 ,1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, data):
        data = self.conv1(data)
        data = F.relu(data)
        data = self.conv2(data)
        data = F.relu(x)
        data = F.max_pool2d(data, 2)
        data = self.dropout1(data)
        data = torch.flatten(data, 1)
        data = self.fc1(data)
        data = F.relu(data)
        data = F.dropout2(data)
        data = self.fc2(data)
        output = F.log_softmax(x, dim=1)
        return output

