from enum import Enum 
import numpy as np

class EmotionCategory(Enum):
    JOY_EXCITEMENT = 'joy/excitement'
    SADNESS = 'sadness'
    CALM_PEACE = 'calm/peace'
    ANGER_TENSION = 'anger/tension'

    def to_score_vector(self):
        '''
        Returns the raw score corresponding to this emotion label as a numpy array.
        joy/excitement --> (1, 0, 0, 0)
        sadness --> (0, 1, 0, 0)
        calm/peace --> (0, 0, 1, 0)
        anger/tension --> (0, 0, 0, 1)
        '''
        score_array = np.zeros(len(EmotionCategory))
        emotion_index = list(EmotionCategory).index(self)
        score_array[emotion_index] = 1
        return score_array

# all these functions are just for getting user input
# btw yes I wrote everything here by hand including the docstrings, no AI used

def get_user_input_songs_from_stdin() -> dict[str, dict[EmotionCategory, float]]:
    '''
    Returns a dictionary mapping each song name to a dictionary mapping emotions to scores:
    joy, sadness, calm, and anger.

    Currently this function reads the song names and all four scores from stdin. In our final product we will
    just take the song names as input and calculate the scores using our neural network.
    '''
    songs_to_scores = {}
    while True:
        song_name = input('Enter song name (leave blank if no more): ')
        if not song_name:
            break 

        emotion_scores = {}
        for emotion in EmotionCategory:
            score = float(input(f'Enter {emotion.value} score (must be decimal between 0 and 1): '))
            assert score >= 0 and score <= 1
            emotion_scores[emotion] = score

        songs_to_scores[song_name] = emotion_scores

    return songs_to_scores

def get_score_nodes_from_stdin(playlist_length) -> dict[int, EmotionCategory]:
    '''
    Returns a dictionary mapping specific positions/indices in the playlist to an emotion at that position.
    Each position is an integer, and the first position in the playlist is position 0. The position must be
    less than the length of the playlist, which is provided. Reads inputs from stdin.
    '''
    def print_emotion_indices_list():
        return ', '.join([str(i) + ' for ' + emotion.value for i, emotion in enumerate(EmotionCategory) ])
    
    desired_emotion_nodes = {}
    while True:
        index = input(f'Enter a playlist position (number from 0-{playlist_length-1}, leave blank if no more): ')
        if not index:
            break 
        index = int(index)
        assert index >= 0 and index < playlist_length
        emotion_index = int(input(f'Enter an integer corresponding to an emotion ({print_emotion_indices_list()}): '))
        desired_emotion_nodes[index] = list(EmotionCategory)[emotion_index]

    # this forces the user to add at least two nodes (beginning and end) mostly just to make our lives easier
    if 0 not in desired_emotion_nodes:
        emotion_index = int(input(f'What emotion should the playlist start with? ({print_emotion_indices_list()}) '))
        desired_emotion_nodes[0] = list(EmotionCategory)[emotion_index]

    if playlist_length-1 not in desired_emotion_nodes:
        emotion_index = int(input(f'What emotion should the playlist end with? ({print_emotion_indices_list()}) '))
        desired_emotion_nodes[playlist_length-1] = list(EmotionCategory)[emotion_index]

    return desired_emotion_nodes

# these print functions are mostly for debugging

def print_out_user_songs(songs_to_scores):
    print('\n----- SONGS -----')
    for song in songs_to_scores:
        scores = songs_to_scores[song]
        print(f'name: {song}')
        print(f'scores: {', '.join([str(scores[emotion]) + ' ' + emotion.value for emotion in EmotionCategory])}\n')

def print_out_emotion_nodes(desired_emotion_nodes, playlist_length):
    print('\n----- EMOTION NODES -----')
    for i in range(playlist_length):
        print(f'position {i}: {desired_emotion_nodes[i].value if i in desired_emotion_nodes else '(none specified)'}')
    print()

def handle_inputs():
    '''
    Gets all inputs required from the user from stdin. Returns a dictionary mapping each song name to
    emotion scores, as well as a dictionary mapping specific playlist positions (indices) to emotions at
    those positions that the user has specified. Currently this gets all the scores from stdin, but in
    our final product we'll be calculating the scores using our neural network.
    '''
    songs_to_scores = get_user_input_songs_from_stdin()
    assert len(songs_to_scores) > 0, 'need at least two songs to make a playlist lol'
    print_out_user_songs(songs_to_scores)
    desired_emotion_nodes = get_score_nodes_from_stdin(len(songs_to_scores))
    print_out_emotion_nodes(desired_emotion_nodes, len(songs_to_scores))
    return songs_to_scores, desired_emotion_nodes

# this is where the actual meat of the local search program starts

def populate_emotion_scores_at_each_playlist_position(user_provided_emotion_nodes, playlist_length) -> list[np.array]:
    '''
    Basically takes the emotion nodes dictionary constructed from the user input and extrapolates it into a
    list of target emotion scores at each possible index in the playlist. (Constructs the loss function we
    want to approximate with the songs given by the user.) Returns this "loss function" as a list of numpy arrays,
    where each item in the list represents the target emotion scores at that spot in the playlist.
    '''
    score_list = [None] * playlist_length

    # fills in all the scores at the positions that have been specified by the user
    for playlist_pos in user_provided_emotion_nodes:
        score_list[playlist_pos] = user_provided_emotion_nodes[playlist_pos].to_score_vector()

    # linearly interpolates between each of the nodes to fill in that values we don't have yet

def fill_in_missing_scores(score_list) -> list[np.array]:
    '''
    The user probably didn't specify the emotion at each point in the playlist, so we have to fill in the gaps between
    the nodes where we have emotion labels. Linearly interpolates between each node to fill in intermediate values. For
    example, if we have (1, 0, 0, 0), ____, (0, 1, 0, 0), then we fill in the blank with (0.5, 0.5, 0, 0). Returns a new
    copy of the list with the missing values filled in.
    '''
    indices_with_values = []
    pass
    

def main():
    songs_to_scores, desired_emotion_nodes = handle_inputs()
    playlist_length = len(songs_to_scores)


if __name__ == '__main__':
    main()