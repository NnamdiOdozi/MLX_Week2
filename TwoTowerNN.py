import torch
import torch.nn as nn
import torch.optim as optim

class Tower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Input dimension: 128
        # Hidden layers: 128 -> 64
        # Output dimension: 64
        self.fc1 = torch.nn.Linear(128, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

