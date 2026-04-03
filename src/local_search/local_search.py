from enum import Enum 

class EmotionCategory(Enum):
    JOY_EXCITEMENT = 'joy/excitement'
    SADNESS = 'sadness'
    CALM_PEACE = 'calm/peace'
    ANGER_TENSION = 'anger/tension'

# all these functions are just for getting user input

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

    if 0 not in desired_emotion_nodes:
        emotion_index = int(input(f'What emotion should the playlist start with? ({print_emotion_indices_list()}) '))
        desired_emotion_nodes[0] = list(EmotionCategory)[emotion_index]

    if playlist_length-1 not in desired_emotion_nodes:
        emotion_index = int(input(f'What emotion should the playlist end with? ({print_emotion_indices_list()}) '))
        desired_emotion_nodes[playlist_length-1] = list(EmotionCategory)[emotion_index]

    return desired_emotion_nodes

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
    

def main():
    songs_to_scores, desired_emotion_nodes = handle_inputs()

if __name__ == '__main__':
    main()