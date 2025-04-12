# Dataset from Kaggle: https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en?resource=download
# Used the datasets Ravdess and Crema-d to train our model

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# does preprocessing on the ravdess dataset and encodes the labels for use in the model 
# using the LabelEncoder from sklearn.preprocessing
def process_ravdess_data(): 
    ravdess_folder = './data/Ravdess/audio_speech_actors_01-24'
    label_encoder = LabelEncoder()
    emotion_labels = {
        "01":"neutral",
        "02":"calm",
        "03":"happy",
        "04":"sad",
        "05":"anger",
        "06":"fear",
        "07":"disgust",
        "08":"surprised",
    }
    ravdess_df = pd.DataFrame()
    paths = []
    labels = []
    for directory in os.listdir(ravdess_folder): 
        folder_path = os.path.join(ravdess_folder, directory)
        for file in os.listdir(folder_path): 
            emotion = file.split("-")[2]
            labels.append(emotion_labels[emotion])
            paths.append(os.path.join(folder_path, file))
    int_labels = label_encoder.fit_transform(labels)
    ravdess_df['paths'] = paths
    ravdess_df['emotions'] = labels
    ravdess_df['labels'] = int_labels
    return ravdess_df

# processes and loads the data from the CREMA dataset folder and does labelling in a similar manner 
# to the ravdess dataset loading
def process_crema_data():
    crema_folder = './data/Crema'
    label_encoder = LabelEncoder()
    emotion_labels = {
        "NEU":"neutral",
        "ANG":"anger",
        "DIS":"disgust",
        "FEA":"fear",
        "HAP":"happy",
        "SAD":"sad"
    }
    crema_df = pd.DataFrame()
    paths = []
    labels = []
    for file in os.listdir(crema_folder): 
        emotion = file.split("_")[2]
        labels.append(emotion_labels[emotion])
        paths.append(os.path.join(crema_folder,  file))
    int_labels = label_encoder.fit_transform(labels)
    crema_df['paths'] = paths
    crema_df['emotions'] = labels
    crema_df['labels'] = int_labels
    return crema_df

# Combines both the Crema and Ravdess datasets into a larger dataset and returns a 
# dataframe for use, drops previous label encodings and encodes the labels again to 
# better represent the data from the combined datasets
def process_both_datasets(): 
    crema = process_crema_data() 
    ravdess = process_ravdess_data()
    data = pd.DataFrame() 
    data = pd.concat([crema, ravdess])
    # Reset the labels correctly if training on both datasets
    data = data.drop('labels', axis=1)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data['emotions'])
    data['labels'] = labels
    return data

# returns the decoded labels corresponding to the dataset being used 
def get_labels(model_type): 
    if model_type == "crema": 
        crema_df = process_crema_data()
        labels = dict(zip(crema_df['labels'], crema_df['emotions']))
        return labels
    if model_type == "ravdess":
        ravdess_df = process_ravdess_data()
        labels = dict(zip(ravdess_df['labels'], ravdess_df['emotions']))
        return labels
    if model_type == "both":
        both_df = process_both_datasets()
        labels = dict(zip(both_df['labels'], both_df['emotions']))
        return labels
