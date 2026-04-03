from enum import Enum 
import numpy as np
import random

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

    score_list_filled = fill_in_missing_scores(score_list)
    return score_list_filled

def fill_in_missing_scores(score_list) -> list[np.array]:
    '''
    The user probably didn't specify the emotion at each point in the playlist, so we have to fill in the gaps between
    the nodes where we have emotion labels. Linearly interpolates between each node to fill in intermediate values. For
    example, if we have (1, 0, 0, 0), ____, (0, 1, 0, 0), then we fill in the blank with (0.5, 0.5, 0, 0). Returns a new
    copy of the list with the missing values filled in.
    '''
    assert score_list[0] is not None, 'there should be a node at the beginning of the playlist'
    assert score_list[-1] is not None, 'there should be a node at the end of the playlist'

    indices_with_values = []
    for i, score in enumerate(score_list):
        if score is not None:
            indices_with_values.append(i)

    score_list_filled = []
    index_of_next_filled_index = 0 # funny confusing variable name, this is an index into indices_with_values
    step = None

    # invariant: on every iteration, we add exactly one item to the new list
    # we're also guaranteed that the first and last entries already have scores provided
    for i, score in enumerate(score_list):
        #print(score_list_filled)
        if i == indices_with_values[index_of_next_filled_index]: # if we're at a node
            score_list_filled.append(score)
            index_of_next_filled_index += 1

            # we're guaranteed the last entry has a provided score, so we exit to prevent a list index error
            # there's probably a more elegant way to do this but idk
            if i == len(score_list) - 1:
                continue

            # we want to transition from the current node to the next one using a constant step size in between
            prev_specified_scores = score
            next_specified_scores = score_list[indices_with_values[index_of_next_filled_index]]
            pos_diff = indices_with_values[index_of_next_filled_index] - i
            step = (next_specified_scores - prev_specified_scores) / pos_diff

        else:
            interpolated_score = score_list_filled[i-1] + step
            score_list_filled.append(interpolated_score)

    assert len(score_list_filled) == len(score_list)
    return score_list_filled


def convert_score_dict_to_vector(emotion_scores) -> np.array:
    '''
    Converts a dictionary that maps emotion categories to scores into a numpy array.
    '''
    l = []
    for emotion in EmotionCategory:
        l.append(emotion_scores[emotion])
    return np.array(l)


def hill_climbing(songs_to_scores, target_scores, num_iterations=1000) -> tuple[list[str], list[np.array]]:
    '''
    Returns the final ordering of songs in the playlist, as well as the list of the corresponding scores
    for each song. Takes the dictionary assembled previously that maps songs to emotion score dictionaries,
    and the list of vectors representing the target score at each position in the playlist.

    Performs simple hill climbing.
    '''
    def get_neighboring_state(cur_list):
        # returns a new list with two random entries swapped
        i1 = random.randrange(0, len(cur_list))
        i2 = i1
        while i2 == i1:
            i2 = random.randrange(0, len(cur_list))
        
        new_list = []
        for i in range(len(cur_list)):
            if i == i1:
                new_list.append(cur_list[i2])
            elif i == i2:
                new_list.append(cur_list[i1])
            else:
                new_list.append(cur_list[i])

        return new_list
    
    def judge_song_list(cur_list) -> float:
        # returns how "good" the current playlist ordering is, with higher being better
        # I'm calling it a fitness value instead of a score value to avoid confusion with the emotion score vectors
        overall_fitness = 0
        cur_scores = [convert_score_dict_to_vector(score_dict) for _, score_dict in cur_list]
        for cur_score_array, target in zip(cur_scores, target_scores):
            
            # currently I'm using the normalized dot product of current score vector and the target vector
            # as the judge of how good the current scores are
            # I was also thinking of just taking the MSE between them instead but idk which one is better
            dot = np.dot(cur_score_array / np.linalg.norm(cur_score_array), target / np.linalg.norm(target))
            overall_fitness += dot

        return overall_fitness

    song_list = list(songs_to_scores.items())
    #print(song_list)
    cur_fitness = judge_song_list(song_list)

    # this is just the hill climbing algorithm we went over in class
    for i in range(num_iterations):
        new_song_list = get_neighboring_state(song_list)
        new_fitness = judge_song_list(new_song_list)
        if new_fitness > cur_fitness:
            song_list = new_song_list 
            cur_fitness = new_fitness 
    
    song_names, song_scores = list(zip(*song_list))
    return song_names, song_scores


# this is just so the output looks nice-ish

def print_output(optimal_ordering, optimal_scores):
    print('\n----- RESULTS -----')
    print(f'final ordering: {' --> '.join(optimal_ordering)}\n')
    print(f'scores:')
    for i, tup in enumerate(zip(optimal_ordering, optimal_scores)):
        song_name, score_dict = tup
        print(f'song {i}: {song_name}: {', '.join([str(score_dict[emotion]) + ' ' + emotion.value for emotion in EmotionCategory])}')



example_songs_to_scores = {
    'a': { EmotionCategory.JOY_EXCITEMENT: 0, EmotionCategory.SADNESS: 0.2, EmotionCategory.CALM_PEACE: 0, EmotionCategory.ANGER_TENSION: 0 },
    'b': { EmotionCategory.JOY_EXCITEMENT: 0, EmotionCategory.SADNESS: 0, EmotionCategory.CALM_PEACE: 0, EmotionCategory.ANGER_TENSION: 1 },
    'c': { EmotionCategory.JOY_EXCITEMENT: 0.6, EmotionCategory.SADNESS: 1, EmotionCategory.CALM_PEACE: 0.4, EmotionCategory.ANGER_TENSION: 0 },
    'd': { EmotionCategory.JOY_EXCITEMENT: 0.8, EmotionCategory.SADNESS: 0.8, EmotionCategory.CALM_PEACE: 0, EmotionCategory.ANGER_TENSION: 1 },
    'e': { EmotionCategory.JOY_EXCITEMENT: 0, EmotionCategory.SADNESS: 0, EmotionCategory.CALM_PEACE: 1, EmotionCategory.ANGER_TENSION: 0 },
    'f': { EmotionCategory.JOY_EXCITEMENT: 0, EmotionCategory.SADNESS: 1, EmotionCategory.CALM_PEACE: 0.5, EmotionCategory.ANGER_TENSION: 0 },
    'g': { EmotionCategory.JOY_EXCITEMENT: 0, EmotionCategory.SADNESS: 0, EmotionCategory.CALM_PEACE: 0, EmotionCategory.ANGER_TENSION: 1 },
}

example_desired_emotion_nodes = {
    0: EmotionCategory.JOY_EXCITEMENT,
    len(example_songs_to_scores)-1: EmotionCategory.SADNESS,
    2: EmotionCategory.ANGER_TENSION,
    4: EmotionCategory.ANGER_TENSION
}



def main():
    #songs_to_scores, desired_emotion_nodes = handle_inputs()

    songs_to_scores = example_songs_to_scores
    desired_emotion_nodes = example_desired_emotion_nodes
    print_out_user_songs(songs_to_scores)
    print_out_emotion_nodes(desired_emotion_nodes, len(songs_to_scores))

    playlist_length = len(songs_to_scores)
    target_scores = populate_emotion_scores_at_each_playlist_position(desired_emotion_nodes, playlist_length)

    optimal_ordering, optimal_scores = hill_climbing(songs_to_scores, target_scores)
    print_output(optimal_ordering, optimal_scores)

if __name__ == '__main__':
    main()