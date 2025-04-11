# Dataset from Kaggle: https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en?resource=download
# Used the datasets Ravdess and Crema to train our model

# imports
import torch 
import numpy as np
from model import *
from preprocessing import *
from data_loading import get_dataLoaders

ravdess = './data/Ravdess/audio_speech_actors_01-24'
crema = "./data/Crema"

## For cuda
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## For apple silicon 
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

## Train and test the model on the ravdess dataset
# print("Training on the ravdess dataset")
# df = process_ravdess_data()
# train_loader, test_loader = get_dataLoaders(df, batch_size=64)
# print("training")
# train(train_loader, num_classes=8, epochs=100)
# print("testing")
# eval(test_loader, num_classes=8)


# Train and test the model on the crema dataset
print("Training on the crema dataset")
df = process_crema_data()
train_loader, test_loader = get_dataLoaders(df, batch_size=64)
print("training")
train(train_loader, num_classes=6, epochs=50)
print("testing")
eval(test_loader, num_classes=6)


# ## Train and test the model on both datasets combined
# print("Training on both datasets")
# df = process_both_datasets()
# train_loader, test_loader = get_dataLoaders(df, 64)
# print("training")
# train(train_loader, num_classes=8, epochs=30)
# print("testing")
# eval(test_loader, num_classes=8)







