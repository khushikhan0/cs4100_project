from local_search.local_search import *
from cnn import CNN
from spectrogramdataset import SpectrogramDataset
from spectrogram_conversion import mp3_to_spectrogram, pad_or_truncate

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import librosa
import librosa.display

INPUT_FILE_PATH = 'input_mp3s'
MODEL_STATE_DICT_PATH = 'CNN_state_dict.pth'

def get_input_mp3s() -> list[str]:
    '''
    Returns all the MP3 files in the input directory where they are placed by the user.
    '''
    file_list = os.listdir(INPUT_FILE_PATH)
    mp3_files = [f for f in file_list if f.endswith('.mp3')]
    #print(mp3_files)
    return mp3_files


def file_names_without_extensions(names) -> list[str]:
    '''
    Returns all the file names in the given list without the file extension. Assumes each 
    file name only contains one period.
    '''
    return [name.split('.')[0] for name in names]


def load_cnn():
    '''
    Creates the CNN and loads its model weights from the saved state dict file.
    '''
    model = CNN(in_channels=1, n_outputs=4)
    model.load_state_dict(torch.load(MODEL_STATE_DICT_PATH))
    model.eval()

    return model


def convert_all_mp3s(input_mp3s):
    '''
    Converts all the MP3 files in the input list into spectrograms. Returns an array of spectrograms,
    where each spectrogram is represented as a 2D numpy array. Each spectrogram is truncated or padded
    so that it is a standard length. The ith spectrogram in the output array corresponds to the ith
    mp3 in the input list.
    '''
    spectrograms = [mp3_to_spectrogram(f'{INPUT_FILE_PATH}/{mp3_path}') for mp3_path in input_mp3s]
    specs_adjusted = np.array([pad_or_truncate(s) for s in spectrograms])
    return specs_adjusted

def score_spectrograms(model, spectrograms):
    '''
    Scores each spectrogram using the given model to obtain a list of emotion scores. Each emotion score is
    a vector with four entries, corresponding to joy/excitement, sadness, calm/peace, and anger/tension. Most
    of this is copy-pasted from the CNN training code tbh

    The ith score corresponds to the ith spectrogram.
    '''
    X = spectrograms #np.array(spectrograms)
    #print(X.shape)
    #print(X)
    n_samples, n_mels, n_timeframes = X.shape
    X = X.reshape(n_samples, 1, n_mels, n_timeframes)
    n_samples, num_channels, n_mels, n_timeframes = X.shape

    img_dims = (256, 256)
    transform = transforms.ToTensor()

    # couldn't be bothered to troubleshoot dimension problems so I just yoinked the CNN training code lmao
    # I'm sure there's a more elegant way to do this though
    ds = SpectrogramDataset(spectrograms=X, 
                                labels=[1] * len(X), 
                                transform=transform, 
                                resize=img_dims)
    
    dataloader = DataLoader(dataset=ds, 
                              batch_size=1, 
                              shuffle=False)
    
    spec_scores = []
    for _, batch in enumerate(dataloader):
        # Move data and targets to GPU, faster performance
        data, _ = batch # Expected data shape: (samples, channels, mels, time frames)

        # Predicted output
        score = model(data)
        spec_scores.append(score)

    #print(spec_scores)
    return spec_scores


def format_build_score_dict(song_names, raw_vectors) -> dict[str, dict[EmotionCategory, float]]:
    '''
    Constructs a dictionary mapping song names to their emotion score dicts. The raw vectors provided
    are torch tensors output by the CNN model.
    '''
    songs_to_scores = {}
    for song_name, song_score in zip(song_names, raw_vectors):
        #print(song_score)
        emotion_scores = {
            EmotionCategory.JOY_EXCITEMENT: song_score[0][0].item(),
            EmotionCategory.CALM_PEACE: song_score[0][1].item(),
            EmotionCategory.ANGER_TENSION: song_score[0][2].item(),
            EmotionCategory.SADNESS: song_score[0][3].item()
        }
        songs_to_scores[song_name] = emotion_scores

    return songs_to_scores


def main():
    playlist_length = 0
    while playlist_length < 2:
        input(f'Press [enter] once all song MP3s have been placed in the {INPUT_FILE_PATH} folder. ')
        input_mp3s = get_input_mp3s()
        playlist_length = len(input_mp3s)
        if playlist_length < 2:
            print(f'Not enough songs found in folder. A playlist needs at least two songs.')
        
    song_names = file_names_without_extensions(input_mp3s)
    print(f'\nFound {playlist_length} songs')
    print(f'Song names:')
    for name in song_names: 
        print(f'- {name}')

    model = load_cnn()
    spectrograms = convert_all_mp3s(input_mp3s)
    raw_score_vectors = score_spectrograms(model, spectrograms)
    songs_to_scores = format_build_score_dict(song_names, raw_score_vectors)
    print('Calculated emotion scores for songs.\n')
    #print(songs_to_scores)

    desired_emotion_nodes = get_score_nodes_from_stdin(len(songs_to_scores))
    print_out_emotion_nodes(desired_emotion_nodes, len(songs_to_scores))

    target_scores = populate_emotion_scores_at_each_playlist_position(desired_emotion_nodes, playlist_length)

    optimal_ordering, optimal_scores = SimulatedAnnealing(songs_to_scores, target_scores, 100, 0.9).search()
    #optimal_ordering, optimal_scores = HillClimbing(songs_to_scores, target_scores).search()
    print_output(optimal_ordering, optimal_scores)

    

if __name__ == "__main__":
    main()
