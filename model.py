# imports
import os
import torch
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
# Load feature extractor and model
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")
model = Wav2Vec2ForSequenceClassification.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")

# Define directories for the datasets
ravdess = './data/Ravdess/audio_speech_actors_01-24'
crema = "./data/Crema"

# For CUDA or Apple Silicon
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset processing for Ravdess
ravdess_emotions = {"1": "neutral", "2": "calm", "3": "happy", "4": "sad", "5": "angry", "6": "fearful", "7": "disgust", "8": "surprised"}
ravdess_audio_files = []
ravdess_audio_labels = []

# Process Ravdess files
for directory in os.listdir(ravdess):
    folder_path = os.path.join(ravdess, directory)
    for file in os.listdir(folder_path):
        ravdess_audio_files.append(file)
        ravdess_audio_labels.append(file.split("-")[2][1])

# Dataset processing for Crema
crema_emotions = {"NEU": "neutral", "HAP": "happy", "SAD": "sad", "ANG": "angry", "FEA": "fearful", "DIS": "disgust"}
crema_audio_files = []
crema_audio_labels = []

# Process Crema files
for file in os.listdir(crema):
    crema_audio_files.append(file)
    crema_audio_labels.append(file.split("_")[2])

# Feature extraction: Mel spectrogram
def extract_mel_spectrogram(file_path, sr=22050, n_mels=128, hop_length=512):
    """Extracts a Mel spectrogram from an audio file using librosa."""
    y, sr = librosa.load(file_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

# Extract features from Ravdess
ravdess_features = []
ravdess_labels = []
for directory in os.listdir(ravdess):
    folder_path = os.path.join(ravdess, directory)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        emotion_label = file.split('-')[2][1]
        mel_spec = extract_mel_spectrogram(file_path)
        if mel_spec is not None:
            ravdess_features.append(mel_spec)
            ravdess_labels.append(emotion_label)

# etract features from Crema
crema_features = []
crema_labels = []
for file in os.listdir(crema):
    file_path = os.path.join(crema, file)
    emotion_label = file.split('_')[2]
    mel_spec = extract_mel_spectrogram(file_path)
    if mel_spec is not None:
        crema_features.append(mel_spec)
        crema_labels.append(emotion_label)

# Split datasets
ravdess_train_x, ravdess_test_x, ravdess_train_y, ravdess_test_y = train_test_split(
    ravdess_audio_files, ravdess_audio_labels, test_size=0.20, random_state=42)
crema_train_x, crema_test_x, crema_train_y, crema_test_y = train_test_split(
    crema_audio_files, crema_audio_labels, test_size=0.30, random_state=42)

def predict_emotion(file_path, model, feature_extractor, device):
    #load file sample rate for Wav2Vec2
    speech, sr = librosa.load(file_path, sr=16000)

    #extract features
    inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)

    #move inpust to devide
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.to(device)
    model.eval()

    #prediction with model
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class_id = torch.argmax(logits).item()

    return model.config.id2label[predicted_class_id]

# Evaluate model on the test set
y_true = []
y_pred = []

#attemps to go through Ravdess test set
for file_name, label in zip(ravdess_test_x, ravdess_test_y):
    file_path = None
    for folder in os.listdir(ravdess):
        folder_path = os.path.join(ravdess, folder)
        candidate_path = os.path.join(folder_path, file_name)
        if os.path.exists(candidate_path):
            file_path = candidate_path
            break

    if file_path is not None:
        true_emotion = ravdess_emotions.get(label)
        predicted_emotion = predict_emotion(file_path, model, feature_extractor, device)

        y_true.append(true_emotion)
        y_pred.append(predicted_emotion)
    else:
        print(f"File not found: {file_name}")

#callification report
print("Accuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))
