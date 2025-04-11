import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Okay this is the path to RAVDESS dataset
ravdess = './data/Ravdess/audio_speech_actors_01-24'

# this is the emotion mapping
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

#for collectign the audio files
ravdess_audio_files = []
ravdess_audio_labels = []

#from ur file
for directory in os.listdir(ravdess):
    folder_path = os.path.join(ravdess, directory)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.wav'):
                ravdess_audio_files.append(os.path.join(folder_path, file))
                emotion_code = file.split("-")[2]
                ravdess_audio_labels.append(ravdess_emotions.get(emotion_code, "Unknown"))

#making df with both audio file and lables
df = pd.DataFrame({"speech": ravdess_audio_files, "label": ravdess_audio_labels})
#see the df
print(df['label'].value_counts())


# okay feature extraction function with mfcc instead still using librosa moduldee
def mfcccc(filename, n_mfcc=40, max_len=87):
    try:
        y, val = librosa.load(filename, duration=3, offset=0.5)
        y = y + 0.005 * np.random.normal(0, 1, len(y))  # Add noise for augmentation
        mfcc = librosa.feature.mfcc(y=y, sr=val, n_mfcc=n_mfcc)

        if mfcc.shape[1] < max_len:
            padding = np.zeros((n_mfcc, max_len - mfcc.shape[1]))
            mfcc = np.concatenate((mfcc, padding), axis=1)
        else:
            mfcc = mfcc[:, :max_len]

        return mfcc.T  #return MFCC
    except Exception as e:
        print(f"error{filename}: {e}")
        return 0


#we got to extract MFCC features
df['features'] = df['speech'].apply(lambda x: mfcccc(x))
X = np.array(df['features'].tolist())
y = df['label']

#scale and standardize the features
scaler = StandardScaler()
X = np.array([scaler.fit_transform(x) for x in X])

#One hot encode labels
encoder = OneHotEncoder()
y = encoder.fit_transform(y.values.reshape(-1, 1)).toarray()

#we want to use CNN model so we have to reshapr for that
X = np.expand_dims(X, -1)

#split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#with CNN and LSTM
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(8, activation='softmax')
])

#compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Train model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=50,batch_size=16)

# how did it do ?
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
