import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import OneHotEncoder


ravdess = './data/Ravdess/audio_speech_actors_01-24'

#mapping emotion codes to labels
ravdess_emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

ravdess_audio_files = []
ravdess_audio_labels = []

#go through the ravdess dataset
for directory in os.listdir(ravdess):
    folder_path = os.path.join(ravdess, directory)
    if os.path.isdir(folder_path):
        print(f"Exploring folder: {folder_path}")
        for file in os.listdir(folder_path):
            if file.endswith('.wav'):
                ravdess_audio_files.append(os.path.join(folder_path, file))
                emotion_code = file.split("-")[2]
                if emotion_code in ravdess_emotions:
                    ravdess_audio_labels.append(ravdess_emotions[emotion_code])
                else:
                    print(f"Warning: Unknown emotion code {emotion_code} in file {file}")
                print(f"Found file: {file}, Emotion: {ravdess_emotions.get(emotion_code, 'Unknown')}")

#check the loaded files
print(f"Loaded {len(ravdess_audio_files)} audio files.")
print(f"First 5 files: {ravdess_audio_files[:5]}")
print(f"First 5 labels: {ravdess_audio_labels[:5]}")

paths = ravdess_audio_files
labels = ravdess_audio_labels
#make df
df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels


print(df['label'].value_counts())

#featyre exrtraction using le
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

#apply feature extraction to all files in the dataset
X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))

#convert list to numpy
X = np.array([x for x in X_mfcc])

#make x work with LSTM
X = np.expand_dims(X, -1)

#ENCODE lables using one hot
enc = OneHotEncoder()
y = enc.fit_transform(df[['label']])
y = y.toarray()
#test
print(X.shape, y.shape)
#Onehot encode the labels
enc = OneHotEncoder()
y = enc.fit_transform(df[['label']])
y = y.toarray()

#check
print(X.shape, y.shape)

#LSTM MODEL USED
model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(40, 1)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(8, activation='softmax')  # 8 emotion categories
])

#COMPILE MODEL
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# Train
history = model.fit(X, y, validation_split=0.2, epochs=50, batch_size=64)

