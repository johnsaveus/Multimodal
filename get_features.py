import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
import torch.nn as nn
from torchsummary import summary

class SpectrogramDataset(Dataset):

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.wav_path = self.df['Path']
        self.label = self.df['Label']
        self._constants()
        self._transformation()
        self._mapping()

    def _constants(self):
        self.SAMPLE_RATE = 22050
        self.N_FFT = 1024
        self.HOP_LENGTH = self.N_FFT // 4
        self.N_MELS = 128
        self.MAX_LEN = 300
        
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
        signal , sr = torchaudio.load(self.wav_path[index])
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self.transformation(signal)
        signal = self._pad_or_truncate(signal)
        target = self.label_map[self.label[index]]
        return signal, target
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
            signal = resampler(signal)
        return signal
        
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal , dim = 0, keep_dim = True) 
        return signal
    
    def _transformation(self):
        self.transformation = torchaudio.transforms.MelSpectrogram(
            sample_rate = self.SAMPLE_RATE,
            n_fft = self.N_FFT,
            hop_length = self.HOP_LENGTH,
            n_mels = self.N_MELS
        )

    def _pad_or_truncate(self, signal):
        if signal.shape[2] > self.MAX_LEN:
            signal = signal[:, :, :self.MAX_LEN]
        elif signal.shape[2] < self.MAX_LEN:
            padding = self.MAX_LEN - signal.shape[2]
            signal = torch.nn.functional.pad(signal, (0, padding))
        return signal


class CNNNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
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
                stride=1,
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
                stride=1,
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
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(23040, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions
    

def create_data_loader(train_data, batch_size):
    train_loader = DataLoader(train_data, batch_size = batch_size)
    return train_loader

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

def train_single_epoch(model, data_loader, loss_fn, optimizer, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)
        prediction = model(input)
        loss = loss_fn(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"loss: {loss.item()}")

def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimizer, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":

    csv_file = "Data/splits/train.csv"
    device = 'cpu'
    dataset = SpectrogramDataset(csv_file=csv_file)
    train_loader = create_data_loader(dataset, BATCH_SIZE)
    cnn = CNNNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(),
                             lr = LEARNING_RATE)
    
    train(cnn, train_loader, loss_fn, optimizer, device, EPOCHS)

    
    # csv_file = "Data/splits/train.csv"
    # ds = SpectrogramDataset(csv_file=csv_file)

    # print(f"There are {len(ds)} samples")
    # signal, label = ds[3]
    # print(signal.shape)
    # print(label)


    

        




# def create_spectrogram(split_path, save_path, split):
#     df = pd.read_csv(split_path)
#     path = df['Path']
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     save_path = save_path + '/' + split + '/'
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     image_paths = []
#     for ix, song_path in enumerate(path):
#         y, sr = librosa.load(song_path, sr = None)
#         S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
#         S_dB = librosa.power_to_db(S, ref=np.max)
#         plt.figure(figsize=(10, 4))
#         librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, fmax=8000)
#         plt.axis('off')
#         plt.tight_layout(pad=0)
#         image_path = save_path + str(ix) + 'mel_spectrogram.png'
#         plt.savefig(image_path , bbox_inches='tight', pad_inches=0)
#         plt.close()
#         image_paths.append(image_path)
#     df['Image_path'] = image_paths
#     df.to_csv(split_path, index = False)

#create_spectrogram('Data/splits/train.csv', 'Data/images', 'train')
#create_spectrogram('Data/splits/val.csv', 'Data/images', 'val')
#create_spectrogram('Data/splits/test.csv', 'Data/images', 'test')
