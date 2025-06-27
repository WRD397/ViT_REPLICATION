import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


class TwoLayerCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # (B, 32, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 32, 32, 32)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B, 64, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 64, 16, 16)
            nn.Flatten(),  # (B, 64*16*16)
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

