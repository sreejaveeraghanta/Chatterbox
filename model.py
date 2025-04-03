# Dataset from Kaggle: https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en?resource=download
# Used the datasets Ravdess and Crema to train our model

#TODO delete DEBUGGING sections after model is trained
# imports
import os
import torch 
import librosa
import torchaudio 
import numpy as np
from sklearn.model_selection import train_test_split

ravdess = './data/Ravdess/audio_speech_actors_01-24'
crema = "./data/Crema"

#TODO: if not using later on, delete these lines of code (no need for a device)
## For cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## For apple silicon 
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


################ DATA PROCESSING ################ 
# Process Ravdess dataseti (numerical labelling)
""" Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised) """

ravdess_emotions = {"1":"neutral", "2":"calm", "3":"happy", "4":"sad", "5":"angry", "6":"fearful", "7":"disgust", "8":"surprised"}

ravdess_audio_files = [] 
ravdess_audio_labels = [] 
for directory in os.listdir(ravdess):
    folder_path = os.path.join(ravdess, directory)
    for file in os.listdir(folder_path): 
        ravdess_audio_files.append(file)
        ravdess_audio_labels.append(file.split("-")[2][1])

# Process the Crema dataset (string labelling)
"""
SAD - sadness;
ANG - angry;
DIS - disgust;
FEA - fear;
HAP - happy;
NEU - neutral.
"""
crema_emotions = {"NEU":"neutral", "HAP":"happy", "SAD":"sad", "ANG":"angry", "FEA":"fearful", "DIS":"disgust"}
crema_audio_files = [] 
crema_audio_labels = []
for file in os.listdir(crema):
    crema_audio_files.append(file)
    crema_audio_labels.append(file.split("_")[2])


################ FEATURE EXTRACTION ################ 
#TODO extract useful features from the audio files (mel spectogram) 
# values that are useful from the

#General speech/music processing	sr=22050 (default)
def extract_mel_spectrogram(file_path, sr=22050, n_mels=128, hop_length=512):
    """Extracts a Mel spectrogram from an audio file using librosa."""
    #loading the audio files 
    y, sr = librosa.load(file_path, sr=sr)
    #compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    #power to db
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max) 
    return mel_spec_db
   
#store the ravdess labels 
ravdess_features = []
ravdess_labels = []

for directory in os.listdir(ravdess):
    folder_path = os.path.join(ravdess, directory)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

         #extract emotion 
        emotion_label = file.split('-')[2][1] 
        mel_spec = extract_mel_spectrogram(file_path)
        if mel_spec is not None:
            ravdess_features.append(mel_spec)
            ravdess_labels.append(emotion_label)

crema_features = []
crema_labels = []

for file in os.listdir(crema):
    file_path = os.path.join(crema, file)
    #extract emotion
    emotion_label = file.split('_')[2]  
    mel_spec = extract_mel_spectrogram(file_path)
    if mel_spec is not None:
        crema_features.append(mel_spec)
        crema_labels.append(emotion_label)

# Getting train and test sets
# 80-20 split because this is a smaller dataset
ravdess_train_x, ravdess_test_x, ravdess_train_y, ravdess_test_y = train_test_split(
        ravdess_audio_files, ravdess_audio_labels, test_size=0.20, random_state=42)

# 70-30 split because this dataset is larger
crema_train_x, crema_test_x, crema_train_y, crema_test_y = train_test_split(
        crema_audio_files, crema_audio_labels, test_size=0.30, random_state=42)

print("done feature extraction")



################ MODEL SET UP ################ 
# Train a model and save the trained model
# Get predictions based on the saved model
def predict(audio_data): 
    #TODO fit the model after training and return the correct label from the model
    return ""







