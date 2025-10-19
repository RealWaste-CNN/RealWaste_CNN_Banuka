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

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) #pooling to reduce dimensionality

        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))   # [B,16,32,32]
        x = self.pool2(F.relu(self.conv2(x)))   # [B,32,16,16]
        x = self.pool3(F.relu(self.conv3(x)))   # [B,64,8,8]

        #Flatten + FC
        x = x.view(x.size(0), -1)               
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)      
        return x                   
if __name__ == "__main__":
    model = SimpleCNN(num_classes=9)
    sample = torch.randn(4, 3, 64, 64)
    out = model(sample)