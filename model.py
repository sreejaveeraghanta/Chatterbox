import torch 
import torch.nn as nn
import torch.optim as optim 
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

class LSTM_Model(nn.Module): 
    def __init__(self, num_classes):
        super(LSTM_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.activation1 = nn.ReLU()
        self.maxPool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.activation2 = nn.ReLU()
        self.maxPool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.2)

        self.LSTM = nn.LSTM(64*8, 128, 2, dropout=0.3, batch_first=True)
        self.Linear = nn.Linear(128, num_classes)

    def forward(self, x):
        batch_size, channels, length, timesteps = x.size()
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.maxPool1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.maxPool2(x)
        x = x.view(x.size(0), x.size(3), -1)
        x, _ = self.LSTM(x)
        x = x[:, -1, :]
        x = self.Linear(x)
        return x


def train(train_loader, num_classes, epochs): 
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = LSTM_Model(num_classes=num_classes).to(device) 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(epochs): 
        model.train() 
        running_loss = 0.0
        accurate = 0 
        total = 0 
        for i, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step() 

            _, predicted = torch.max(outputs, 1)
            accurate += (predicted == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()
        
        epoch_loss = running_loss/len(train_loader)
        epoch_accuracy = 100 * (accurate/total)
        print(f"Epoch[{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    torch.save(model.state_dict(), f'./models/speech_recognition_model.pth')

## TODO finish this to run the saved model on the test dataset for evaluating
def eval(test_loader, num_classes): 
    model = LSTM_Model(num_classes=num_classes)
    model.load_state_dict(torch.load("./models/speech_recognition_model.pth"))
    model.eval()
    predictions = [] 
    true_labels = []
    with torch.no_grad(): 
        for i, (data, labels) in enumerate(test_loader):
            output = model(data)
            _, prediction = torch.max(output, 1)
            prediction = prediction.tolist()
            labels = labels.tolist()
            predictions += prediction
            true_labels += labels
        
    precision = precision_score(true_labels, predictions, average='macro') *100
    accuracy = accuracy_score(true_labels, predictions) *100
    recall = recall_score(true_labels, predictions, average='macro') * 100
    confusions = confusion_matrix(true_labels, predictions)
    print(f"Precision: {precision:.2f}, Recall: {recall:.4f}, Accuracy: {accuracy:.2f}%")
    matrix = ConfusionMatrixDisplay(confusion_matrix=confusions)
    matrix.plot()
    plt.savefig('./matrix.png')


def predict(audio_file): 
    model = LSTM_Model(num_classes=9)
    model.load_state_dict(torch.load("./models/speech_recognition_model.pth"))
    model.eval()
    with torch.no_grad(): 
        output = model(audio_file)
        print(output)
        _, prediction = torch.max(output,0)
    ravdess_emotions = {"1":"neutral", "2":"calm", "3":"happy", "4":"sad", "5":"angry", "6":"fearful", "7":"disgust", "8":"surprised"}
    return ravdess_emotions[str(prediction.item())]
