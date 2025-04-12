
import torch 
import numpy as np
from model import *
from preprocessing import *
from data_loading import get_dataLoaders
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

ravdess = './data/Ravdess/audio_speech_actors_01-24'
crema = "./data/Crema"

## For cuda
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## For apple silicon 
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Train and test the model on the ravdess dataset
print("Training on the ravdess dataset")
df = process_ravdess_data()
train_loader, test_loader = get_dataLoaders(df, batch_size=32)
print("training")
train(train_loader, num_classes=8, epochs=100, model_type="ravdess")
print("testing")
confusions = eval(test_loader, num_classes=8, model_type="ravdess")
matrix = ConfusionMatrixDisplay(confusion_matrix=confusions)
matrix.plot()
plt.title("Confusion Matrix for RAVDESS")
plt.savefig("Ravdess_Confusion_Matrix")


# Train and test the model on the crema dataset
print("Training on the crema dataset")
df = process_crema_data()
train_loader, test_loader = get_dataLoaders(df, batch_size=32)
print("training")
train(train_loader, num_classes=6, epochs=80, model_type="crema")
print("testing")
confusions = eval(test_loader, num_classes=6, model_type="crema")
matrix = ConfusionMatrixDisplay(confusion_matrix=confusions)
matrix.plot()
plt.title("Confusion Matrix for CREMA-D")
plt.savefig("Crema_Confusion_Matrix")


# Train and test the model on both datasets combined
print("Training on both datasets")
df = process_both_datasets()
train_loader, test_loader = get_dataLoaders(df, 64)
print("training")
train(train_loader, num_classes=8, epochs=80, model_type="both")
print("testing")
confusions = eval(test_loader, num_classes=8, model_type="both")
matrix = ConfusionMatrixDisplay(confusion_matrix=confusions)
matrix.plot()
plt.title("Confusion Matrix for Combined Datasets")
plt.savefig("Combined_Confusion_Matrix.png")












