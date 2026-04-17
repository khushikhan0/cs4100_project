import torch.nn.functional as F

from torch import nn

class CNN(nn.Module):
    def __init__(self, in_channels=1, n_outputs=4):
        super(CNN, self).__init__()

        # 1st convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=16,
            kernel_size=3,
            padding=1)
        
        # Max pooling layer 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2nd convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=16, 
            out_channels=32, 
            kernel_size=3,
            padding=1)
        
        # 3 convolutional layer
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # Fully connected layer
        self.n_outputs = n_outputs
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(64, n_outputs)

    def forward(self, x):
        n_batch, n_time_frames, n_channels, n_mels = x.shape
        x = x.reshape(n_batch, n_channels, n_mels, n_time_frames)

        x = F.relu(self.conv1(x))  # Apply first convolutional layer and ReLU activation func
        x = self.pool(x)           # Apply max pooling

        x = F.relu(self.conv2(x))  # Apply second convolutional layer and ReLU activation func
        x = self.pool(x)           # Apply max pooling

        x = F.relu(self.conv3(x))  # Apply third convolutional layer and ReLU activation func
        x = self.pool(x)
        
        x = self.global_pool(x)
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor

        x = self.sigmoid(self.fc1(x))
        return x