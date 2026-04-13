from local_search.local_search import *
from cnn import CNN
from spectrogram_conversion import mp3_to_spectrogram, pad_or_truncate

import torch
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
    model = CNN()
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
    spectrograms = [mp3_to_spectrogram(mp3_path) for mp3_path in input_mp3s]
    specs_adjusted = [np.array([pad_or_truncate(s) for s in spectrograms])]
    return specs_adjusted

def score_spectrograms(model, spectrograms):
    '''
    Scores each spectrogram using the given model to obtain a list of emotion scores. Each emotion score is
    a vector with four entries, corresponding to joy/excitement, sadness, calm/peace, and anger/tension.
    '''
    for spec in spectrograms:
        score = model(spec)
        print(score)


def main():
    input(f'Press [enter] once all song MP3s have been placed in the {INPUT_FILE_PATH} folder. ')

    input_mp3s = get_input_mp3s()
    playlist_length = len(input_mp3s)

    print(f'Found \n{playlist_length} songs')
    print(f'Song names:')
    for name in file_names_without_extensions(input_mp3s):
        print(f'- {name}')

    model = load_cnn()
    spectrograms = convert_all_mp3s(input_mp3s)
    score_spectrograms(model, spectrograms)

    

if __name__ == "__main__":
    main()
