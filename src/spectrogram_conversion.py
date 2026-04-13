import pandas as pd
import numpy as np
import librosa
import librosa.display
import os

# code copy-pasted from fma_spectrogram_generation.ipynb so we can import it into main


## Parameters 
SAMPLE_RATE = 22050
DURATION = 30       
N_MELS = 128         
HOP_LENGTH = 512
N_FFT = 2048
TARGET_FRAMES = 1292  

def mp3_to_spectrogram(mp3_path, sr=SAMPLE_RATE, duration=DURATION, dims=(128, 1291)):
    """
    Loads an MP3 file and converts it to a mel spectrogram.
    Returns a 2D numpy array (n_mels x time_frames), or None if the file fails.
    """
    try:
        # Loading the audio
        y, sr = librosa.load(mp3_path, sr=sr, duration=duration, mono=True)

        # Computing mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )

        # Converting to log scale 
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalizing to [0, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())

        return mel_spec_norm.reshape(dims[0], dims[1])

    except Exception as e:
        print(f"Failed to process {mp3_path}: {e}")
        return None


def pad_or_truncate(spec, target_frames=TARGET_FRAMES):
    if spec.shape[1] >= target_frames:
        return spec[:, :target_frames]
    else:
        pad_width = target_frames - spec.shape[1]
        return np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')