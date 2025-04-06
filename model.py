import torch 
import torch.nn as nn
import torch.optim as optim 
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

class LSTM_Model(nn.Module): 
    def __init__(self, input_size=128, hidden_size=256, num_layer=2, num_classes=8, dropout=0.2):
        super(LSTM_Model, self).__init__()
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layer, dropout=dropout, batch_first=True)
        self.Linear = nn.Linear(hidden_size, num_classes)
    def forward(self, x): 
        x, _ = self.LSTM(x)
        x = x[:, -1, :]
        x = self.Linear(x)
        return x


def train(dataset_x, dataset_y, num_classes): 
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = LSTM_Model(num_classes=num_classes).to(device) 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    for e in range(20): 
        accuracy = 0
        total_loss = 0
        model.train()
        dataset_x, dataset_y = dataset_x.to(device), dataset_y.to(device)
        optimizer.zero_grad()
        outputs = model(dataset_x)
        loss = criterion(outputs, dataset_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = torch.argmax(outputs, 1)
        epoch_loss = total_loss/len(dataset_x)
        accuracy += (predicted == dataset_y).sum().item()
        accuracy = (accuracy / len(dataset_y)) * 100
        print(f"Epoch[{e+1}/{20}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), f'./models/speech_recognition_model.pth')

## TODO finish this to run the saved model on the test dataset for evaluating
def eval(dataset_x, dataset_y, num_classes): 
    model = LSTM_Model(num_classes=num_classes)
    model.load_state_dict(torch.load("./models/speech_recognition_model.pth"))
    model.eval()
    predictions = [] 
    true_labels = [] 
    with torch.no_grad(): 
        output = model(dataset_x)
        print(output)

    print(f"Test Accuracy: {accuracy_score(true_labels, output)*100:.2f}%")


def predict(audio_file): 
    #TODO predict the emotion given the audio
    return "Neutral"
