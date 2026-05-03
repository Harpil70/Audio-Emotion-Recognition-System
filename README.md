# Audio Emotion Recognition System

This project is an **Audio Emotion Recognition System (AERS)** built using Python. It uses a Convolutional Neural Network (CNN) to classify human emotions from speech audio files. The system processes audio signals, extracts Mel Spectrograms, and trains a deep learning model to predict emotions such as Happy, Sad, Angry, Fearful, Disgust, Surprised, and Neutral.

## Datasets

The model is trained and evaluated using two popular public datasets for speech emotion recognition:

1. **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**
   - Includes speech and song, audio and video. This project utilizes the audio-only speech files.
   - [Download RAVDESS Dataset on Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

2. **TESS (Toronto Emotional Speech Set)**
   - A set of 200 target words spoken in the carrier phrase "Say the word _____" by two actresses and portraying seven emotions.
   - [Download TESS Dataset on Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

## Features

- **Audio Processing**: Loads audio data and converts them to Mel Spectrograms using `librosa`.
- **Data Augmentation**: Enhances the training dataset by applying time stretching and injecting random noise to improve model generalization.
- **Deep Learning Model**: A sequential CNN built with `TensorFlow` and `Keras` including multiple Conv2D layers, Batch Normalization, MaxPooling, Global Average Pooling, and Dropout layers for robust feature extraction and classification.
- **Class Balancing**: Utilizes `sklearn.utils.class_weight` to compute and apply class weights, ensuring the model doesn't bias towards majority classes.

## Tech Stack & Dependencies

- **Python 3.x**
- **Pandas & NumPy**: For data manipulation and numerical operations.
- **Librosa**: For audio/music analysis and feature extraction.
- **Pillow (PIL)**: For image resizing and handling spectrograms as images.
- **Scikit-learn (sklearn)**: For dataset splitting and computing class weights.
- **TensorFlow / Keras**: For building and training the neural network.
- **Matplotlib**: For plotting training history (accuracy/loss curves).

## How It Works

1. **Data Loading**: Parses folder structures and filenames to map audio files to their respective emotion labels.
2. **Feature Extraction (`extract_from_path`)**: 
   - Audio files are loaded and standardly padded/truncated to a 3-second duration.
   - Mel Spectrograms are generated, converted to decibels, normalized, and resized into 128x128 images.
3. **Data Augmentation (`augment`)**: Synthesizes new training examples to increase the dataset size (only applied to the RAVDESS training set).
4. **Model Training**: A CNN is compiled using the Adam optimizer and Sparse Categorical Crossentropy loss. It trains over 100 epochs with validation tracking.

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/Harpil70/Audio-Emotion-Recognition-System.git
   cd Audio-Emotion-Recognition-System
   ```
2. Download the datasets from the links above and extract them into a `dataset/` directory inside the project root:
   - `dataset/RAVDESS/`
   - `dataset/TESS/`
3. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib librosa pillow scikit-learn tensorflow
   ```
4. Open and run the Jupyter Notebook:
   ```bash
   jupyter notebook AERS.ipynb
   ```
