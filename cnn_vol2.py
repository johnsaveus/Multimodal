import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
import torch.nn as nn
from torchsummary import summary
import librosa
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Constants:
    sample_rate = 22050
    fft_size = 1024
    win_size = 1024
    feature_length = 1024
    hop_size = 512
    num_mels = 64

class SpectrogramDataset(Dataset):

    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        df['Path'] = df['Path'].str.replace(r'\\', '/', regex=True)
        df['Path'] = df['Path'].str.replace(r'\/', '/', regex=True)
        value_to_remove = "Data/raw/jazz/jazz.00054.wav"
        df = df[df['Path'] != value_to_remove]
        df = df.iloc[:5]
        self.wav_path = df['Path']
        self.label = df['Label']
        self._mapping()

    def _mapping(self):

        self.label_map = {'blues' : 0,
                          'classical' : 1,
                          'country' : 2,
                          'disco' : 3,
                          'hiphop' : 4,
                          'jazz' : 5,
                          'metal' : 6,
                          'pop' : 7,
                          'reggae' : 8,
                          'rock' : 9}
    def __len__(self):
        return len(self.wav_path)
    
    def __getitem__(self, index):
        signal = self._get_spectrogram(self.wav_path[index])
        target = self.label_map[self.label[index]]
        return signal, target
    
    def _get_spectrogram(self, song_path):
        y, _  = librosa.load(song_path, sr = Constants.sample_rate)
        S = librosa.stft(y = y, n_fft=Constants.fft_size, hop_length = Constants.hop_size, win_length = Constants.win_size)

        mel_basis = librosa.filters.mel(sr = Constants.sample_rate, n_fft=Constants.fft_size, n_mels = Constants.num_mels)
        mel_S = np.dot(mel_basis, np.abs(S))
        mel_S = np.log10(1+10*mel_S)
        mel_S = mel_S.T

        mel_S_transformed = self._resize_array(mel_S)
        mel_S_transformed = torch.tensor(mel_S_transformed.T)
        return mel_S_transformed.unsqueeze(dim = 0)
    
    def _resize_array(self, array):
        length = Constants.feature_length
        resize_array = np.zeros((length, array.shape[1]))
        if array.shape[0] >= length:
            resize_array = array[:length]
        else:
            resize_array[:array.shape[0]] = array
        return resize_array
    
def create_data_loader(train_data, batch_size):
    train_loader = DataLoader(train_data, batch_size = batch_size)
    return train_loader

BATCH_SIZE = 3
EPOCHS = 10
LEARNING_RATE = 0.001

def train_single_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)
        prediction = model(input)
        loss = loss_fn(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
    print(f" Loss = {epoch_loss / len(data_loader) }")
    return epoch_loss / len(data_loader)

def validate_single_epoch(model, data_loader, loss_fn, device):
    model.eval()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)
        with torch.no_grad():
            prediction = model(input)
        loss = loss_fn(prediction, target)
        epoch_loss+=loss.item()
    print(f" Loss = {epoch_loss / len(data_loader)}")
    return epoch_loss / len(data_loader)

def train(model, data_loader, loss_fn, optimizer, device, epochs):
    train_losses = []
    val_losses = []
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_loss = train_single_epoch(model, data_loader, loss_fn, optimizer, device)
        val_loss = validate_single_epoch(model, data_loader, loss_fn, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print("---------------------------")
    print("Finished training")
    return train_losses, val_losses

class CNNNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256,10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        logits = self.fc2(F.relu(x))
        predictions = self.softmax(logits)
        return predictions

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    csv_train = "Data/splits/train.csv"
    csv_val = "Data/splits/val.csv"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = SpectrogramDataset(csv_file=csv_train)
    val_dataset = SpectrogramDataset(csv_file=csv_val)
    train_loader = create_data_loader(train_dataset, BATCH_SIZE)
    val_loader = create_data_loader(val_dataset, BATCH_SIZE)
    cnn = CNNNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(),
                             lr = LEARNING_RATE)
    train_losses, val_losses = train(cnn, train_loader, loss_fn, optimizer, device, EPOCHS)
    print(train_losses)
    print(val_losses)
    torch.save(cnn, 'cnn.pth')

def plot_losses(train_losses, val_losses, epochs):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('Loss Plot')
    plt.show()

plot_losses(train_losses, val_losses, EPOCHS)
