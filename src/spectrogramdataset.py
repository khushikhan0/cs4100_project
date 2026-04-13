import torch

from torch.utils.data import Dataset

# Defining a custom dataset class
class SpectrogramDataset(Dataset):
    def __init__(self, spectrograms, labels, transform=None, resize=None):
        self.spectrograms = spectrograms
        self.labels = labels
        self.transform = transform
        self.resize = resize

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.spectrograms[idx]
        emotion_labels = self.labels[idx]
        
        # Convert numpy array/image to Pytorch tensor
        if self.transform:
            img = self.transform(img)

        # Convert targets to Pytorch tensor type
        emotion_labels = torch.tensor(emotion_labels, dtype=torch.float32)

        if not torch.is_tensor(img) or not torch.is_tensor(emotion_labels):
            raise TypeError("Expected a torch.Tensor: Either the spectrogram or labels are incorrect types")
        
        if img.shape[1] != 1:
            n_timeframes, n_channels, n_mels = img.shape
            img = img.reshape(n_channels, n_timeframes, n_mels)

        # Returning (spectrogram, emotional labels)
        return img, emotion_labels