from local_search import *
import os

INPUT_FILE_PATH = 'input_mp3s'

def get_input_mp3s() -> list[str]:
    '''
    Returns all the MP3 files in the input directory where they are placed by the user.
    '''
    file_list = os.listdir(INPUT_FILE_PATH)
    mp3_files = [f for f in file_list if f.endswith('.mp3')]
    print(mp3_files)
    return mp3_files


def file_names_without_extensions(names) -> list[str]:
    '''
    Returns all the file names in the given list without the file extension. Assumes each 
    file name only contains one period.
    '''
    return [name.split('.')[0] for name in names]


def main():
    input_mp3s = get_input_mp3s()
    print(f'Song names:{''}')

if __name__ == "__main__":
    main()
