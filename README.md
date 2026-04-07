# CS4100 Final Project  
## AI-Assisted Playlist Ordering to Maximize Emotional Impact
Team Members: Khushi Khan, Dustin Zhang, Kayla Handley, Koena Gupta

## Code  
Notebooks and scripts:
1. [[`download_data.py`](https://github.com/khushikhan0/cs4100_project/blob/main/src/download_data.py)]: downloads the `fma_small.zip` file containing 8,000 tracks of 30s, 8 balanced genres (7.2GB) and `fma_metadata.zip` file containing metadata and features for all tracks. More information can be found from the [dataset source](https://github.com/mdeff/fma)
2. [[`fma_dataset_processed.ipynb`](https://github.com/khushikhan0/cs4100_project/blob/main/src/fma_dataset_processed.ipynb)]: downloads
3. [[`fma_spectrogram_generation.ipynb`](https://github.com/khushikhan0/cs4100_project/blob/main/src/fma_spectrogram_generation.ipynb)]: does x, y, and z
4. [[`fma_emotion_labeling.ipynb`](https://github.com/khushikhan0/cs4100_project/blob/main/src/data_labeling/fma_emotion_labeling.ipynb)]: does x, y, and z
5. [[`model.ipynb`](https://github.com/khushikhan0/cs4100_project/blob/main/src/model.ipynb)]: contains Convolutional Neural Network (CNN) implementation to label songs based on spectrogram inputs generated from mp3 files
