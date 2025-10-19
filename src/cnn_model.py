import torch
import torch.nn as nn
import torch.nn.functional as F

# Here I have implemented the CNN architecture as per the specifications provided.
# Refer Readme.md for more details.

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1) # 1st Convolution with 16 filters
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #pooling to reduce dimensionality

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) # 2nd Convolution with 32 filters
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) #pooling to reduce dimensionality

        self.fc1 = nn.Linear(32 * 16 * 16, 128) # Fully connected layers # 64x64 → 16x16 after 2 poolings
        self.dropout = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(128, num_classes) #Output layer (K = num_classes)

    def forward(self, x):
        # Conv → ReLU → Pool
        x = self.pool1(F.relu(self.conv1(x)))
        # Conv → ReLU → Pool
        x = self.pool2(F.relu(self.conv2(x)))
        # Flatten
        x = x.view(x.size(0), -1)
        # FC + Dropout + ReLU
        x = self.dropout(F.relu(self.fc1(x)))
        # Output 
        x = self.fc2(x)
        return x
