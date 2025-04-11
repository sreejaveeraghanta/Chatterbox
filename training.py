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


print("loading data loaders")
df = process_crema_data()
train_loader, test_loader = get_dataLoaders(df, 64)

print("starting training")
train(train_loader, num_classes=8, epochs=50)
print("Done training")

print("started evaluating")
eval(test_loader, num_classes=8)
print("Done evaluating")







