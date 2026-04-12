import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from spectrogramdataset import SpectrogramDataset
from cnn import CNN

# Reading in data
mp3_df = pd.read_csv('../data/cleaned/fma_cleaned_dataset_emotion_labels.csv', low_memory=False)
spectrograms_df = np.load('../data/cleaned/fma_spectrograms.npy')


# Targets to predict
y = mp3_df[['emotion_joy_excitement_softmax', 
            'emotion_peaceful_content_softmax', 
            'emotion_anger_tension_softmax', 
            'emotion_sadness_softmax']].to_numpy()

# Converting targets to binary for multi label classification
y = (y > 0.25).astype(np.float32)

n_outputs = y.shape[1]

# Passing in spectrograms
X = spectrograms_df
n_samples, n_mels, n_timeframes = X.shape
X = X.reshape(n_samples, 1, n_mels, n_timeframes)
n_samples, num_channels, n_mels, n_timeframes = X.shape

# Splitting data into train, val, and test sets manually to preserve insertion order
X_split_idx = int(X.shape[0] * 0.6)
y_split_idx = int(y.shape[0] * 0.6)
X_train, X_other = X[:X_split_idx, :], X[X_split_idx:, :] # 60% training, 40% other (for val and test)
y_train, y_other = y[:y_split_idx, :], y[y_split_idx:, :] # 60% training, 40% other (for val and test)

X_val_split_idx = int(X_other.shape[0] * 0.5)
y_val_split_idx = int(y_other.shape[0] * 0.5)

X_val, X_test = X[:X_val_split_idx, :], X[X_val_split_idx:, :] # 50% validation, 50% test
y_val, y_test = y[:y_val_split_idx, :], y[y_val_split_idx:, :] # 50% validation, 50% test

# Initialize the datasets
img_dims = (256, 256)
transform = transforms.ToTensor()
train_ds = SpectrogramDataset(spectrograms=X_train, 
                              labels=y_train, 
                              transform=transform, 
                              resize=img_dims)
val_ds = SpectrogramDataset(spectrograms=X_val, 
                             labels=y_val,
                             transform=transform, 
                             resize=img_dims)
test_ds = SpectrogramDataset(spectrograms=X_test, 
                             labels=y_test,
                             transform=transform, 
                             resize=img_dims)

# Initialize the data loaders; insertion order matters, so we won't shuffle
batch_size = 10
train_dataloader = DataLoader(dataset=train_ds, 
                              batch_size=batch_size, 
                              shuffle=False)
val_dataloader = DataLoader(dataset=val_ds, 
                            batch_size=batch_size, 
                            shuffle=False)
test_dataloader = DataLoader(dataset=test_ds, 
                             batch_size=batch_size, 
                             shuffle=False)

# Defining model class
device = "cuda" if torch.cuda.is_available() else "cpu"

# Since the spectrograms are grayscale, in-channels=1
model = CNN(in_channels=1, num_classes=n_outputs).to(device)
    

# Define the loss function for multi-label classification
criterion = nn.BCEWithLogitsLoss()

# Define the optimizer and learning rate
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs=20
for epoch in range(num_epochs):
  # Iterate over training batches
  sum_loss = 0
  loss_len = 0
  for _, batch in enumerate(tqdm(train_dataloader)):
    # Move data and targets to GPU, faster performance
    data, targets = batch # Expected data shape: (samples, channels, mels, time frames)

    data = data.to(device)
    targets = targets.to(device)

    # Predicted output
    preds = model(data)

    # Calculating Binary Cross Entropy loss
    loss = criterion(preds, targets)
    sum_loss += loss.item()
    loss_len += 1

    # Accumulate gradients
    optimizer.zero_grad()

    # Computes the gradients of the loss w.r.t. model parameters/Backward pass
    loss.backward()

    # Update the weights
    optimizer.step()

