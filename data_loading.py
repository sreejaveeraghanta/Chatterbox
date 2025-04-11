import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

def get_mfcc(audio_file):
    y, sr = librosa.load(audio_file, duration=5, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = librosa.util.fix_length(mfcc, size=128, axis=1)
    ## normlize the features
    mfcc = (mfcc - np.mean(mfcc))/(np.std(mfcc))
    return mfcc

def extract_features(data_frame):
    features = []
    for file in data_frame['paths']: 
        features.append(get_mfcc(file))
    return features

def create_train_test_sets(data_frame): 
    X = extract_features(data_frame)
    y = data_frame['labels'].values
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    x_train = torch.tensor(np.array(x_train), dtype=torch.float32)
    x_test = torch.tensor(np.array(x_test), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.long)
    y_test = torch.tensor(np.array(y_test), dtype=torch.long)

    ## get the correct shape of the data
    x_train = x_train.unsqueeze(1)
    x_test = x_test.unsqueeze(1)

    return x_train, x_test, y_train, y_test

def get_dataLoaders(data_frame, batch_size):
    x_train, x_test, y_train, y_test = create_train_test_sets(data_frame)
    train_dataset = TensorDataset(x_train, y_train) 
    test_dataset = TensorDataset(x_test, y_test) 

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
