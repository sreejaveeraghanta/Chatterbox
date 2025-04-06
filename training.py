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
from model import *

ravdess = './data/Ravdess/audio_speech_actors_01-24'
crema = "./data/Crema"

#TODO: if not using later on, delete these lines of code (no need for a device)
## For cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## For apple silicon 
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


################ DATA PROCESSING ################ 
# Process Ravdess dataseti (numerical labelling)
print("Started loading datasets")
""" Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised) """

ravdess_emotions = {"1":"neutral", "2":"calm", "3":"happy", "4":"sad", "5":"angry", "6":"fearful", "7":"disgust", "8":"surprised"}

print("Loading Ravdess")
ravdess_audio_files = [] 
ravdess_audio_labels = [] 
for directory in os.listdir(ravdess):
    folder_path = os.path.join(ravdess, directory)
    for file in os.listdir(folder_path): 
        ravdess_audio_files.append(os.path.join(folder_path,file))
        ravdess_audio_labels.append(int(file.split("-")[2][1]))

# Process the Crema dataset (string labelling)
"""
SAD - sadness;
ANG - angry;
DIS - disgust;
FEA - fear;
HAP - happy;
NEU - neutral.
"""
print("Loading Crema")
crema_emotions = {"NEU":"neutral", "HAP":"happy", "SAD":"sad", "ANG":"angry", "FEA":"fearful", "DIS":"disgust"}
crema_audio_files = [] 
crema_audio_labels = []
for file in os.listdir(crema):
    crema_audio_files.append(os.path.join(crema,file))
    crema_audio_labels.append(file.split("_")[2])

# Getting train and test sets
# 80-20 split because this is a smaller dataset
ravdess_train_x, ravdess_test_x, ravdess_train_y, ravdess_test_y = train_test_split(
        ravdess_audio_files, ravdess_audio_labels, test_size=0.20, random_state=42)

# 70-30 split because this dataset is larger
crema_train_x, crema_test_x, crema_train_y, crema_test_y = train_test_split(
        crema_audio_files, crema_audio_labels, test_size=0.30, random_state=42)

print("Finished loading datasets")
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
    if mel_spec_db.shape[1] > 130: 
        mel_spec_db = mel_spec_db[:, :130]
    if mel_spec_db.shape[1] < 130: 
        padding = 130 - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0,0), (0, padding)), mode='constant')
    return mel_spec_db
   
#store the ravdess labels 

## This takes a while to do, because the datasets are large, and we are loading up two datasets
ravdess_train_features = torch.zeros((len(ravdess_train_x), 128, 130), dtype=torch.float32)
ravdess_test_features = torch.zeros((len(ravdess_test_x), 128, 130), dtype=torch.float32)

crema_train_featueres = []
crema_train_featueres = []

print("Started feature extraction")
for i, file in enumerate(ravdess_train_x): 
    ravdess_train_features[i] = torch.tensor(extract_mel_spectrogram(file), dtype=torch.float32)

for i, file in enumerate(ravdess_test_x): 
    ravdess_test_features[i] = torch.tensor(extract_mel_spectrogram(file), dtype=torch.float32)


#TODO uncomment when training the whole thing
# for i, file in enumerate(crema_train_x): 
    # crema_train_x[i] = extract_mel_spectrogram(file)

# for i, file in enumerate(crema_test_x): 
    # crema_test_x[i] = extract_mel_spectrogram(file)
print("Finished feature extraction")

################ TRAINING THE MODEL ################ 
# Train a model and save the trained model
# Get predictions based 
print("Started training")
train(ravdess_train_features, torch.tensor(ravdess_train_y, dtype=torch.long),9)
print("Finished training")







