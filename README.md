# CS4100 Final Project  
## AI-Powered Playlist Ordering
Team Members: Khushi Khan, Dustin Zhang, Kayla Handley, Koena Gupta

## Code  
Notebooks and scripts:
1. [[`download_data.py`](https://github.com/khushikhan0/cs4100_project/blob/main/src/download_data.py)]: downloads the `fma_small.zip` file containing 8,000 tracks of 30s, 8 balanced genres (7.2GB) and `fma_metadata.zip` file containing metadata and features for all tracks. More information can be found from the [dataset source](https://github.com/mdeff/fma)
2. [[`fma_dataset_processed.ipynb`](https://github.com/khushikhan0/cs4100_project/blob/main/src/fma_dataset_processed.ipynb)]: ...
3. [[`fma_spectrogram_generation.ipynb`](https://github.com/khushikhan0/cs4100_project/blob/main/src/fma_spectrogram_generation.ipynb)]: ...
4. [[`fma_emotion_labeling.ipynb`](https://github.com/khushikhan0/cs4100_project/blob/main/src/data_labeling/fma_emotion_labeling.ipynb)]: ...
5. [[`model.ipynb`](https://github.com/khushikhan0/cs4100_project/blob/main/src/model.ipynb)]: contains training and analysis of Convolutional Neural Network (CNN) model to predict emotional labels for songs based on spectrogram inputs.
6. [[`cnn.py`](https://github.com/khushikhan0/cs4100_project/blob/main/src/cnn.py)]: contains the CNN class.
7. [[`cnn_main.py`](https://github.com/khushikhan0/cs4100_project/blob/main/src/cnn_main.py)]: trains and saves the CNN model.
8. [[`main.py`](https://github.com/khushikhan0/cs4100_project/blob/main/src/main.py)]: ...
9. [[`spectrogram_conversion.py`](https://github.com/khushikhan0/cs4100_project/blob/main/src/spectrogram_conversion.py)]: ...
10. [[`spectrogram_dataset.py`](https://github.com/khushikhan0/cs4100_project/blob/main/src/spectrogramdataset.py)]: contains the SpectrogramDataset class.

## Usage: 
1. Clone and navigate to the repository.
```bash
git clone https://github.com/khushikhan0/cs4100_project.git
cd cs4100_project
```
2. Download the required packages.
```bash
pip install -r requirements.txt
```
4. Download the data.
```bash
download_data.py
```
3. Process the data.
4. 
