import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import librosa
import numpy as np
import torch.nn.functional as F
from cnn_model import Constants
import os
import shutil
from cnn_model import CNNNetwork
import torch
import torch.nn.functional as F

def preproccess_mp3_path(root_dir):

    for genre_dir in os.listdir(root_dir):
        genre_path = os.path.join(root_dir, genre_dir)
        if not os.path.isdir(genre_path):
            continue
        for filename in os.listdir(genre_path):
            if filename.startswith(f"{genre_dir}-"):
                numeric_part = filename.split(f"{genre_dir}-")[1]
                new_filename = f"{numeric_part}"
                old_filepath = os.path.join(genre_path, filename)
                new_filepath = os.path.join(genre_path, new_filename)
                shutil.move(old_filepath, new_filepath)

def move_mp3_external(root_dir, external_folder):

    os.makedirs(external_folder, exist_ok=True)
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for root, _, files in os.walk(subdir_path):
            for file in files:
                source_file = os.path.join(root, file)
                destination_file = os.path.join(external_folder, file)
                shutil.move(source_file, destination_file)

def remove_ids(folder_path, csv_file):
    df = pd.read_csv(csv_file)
    rows_to_remove = []
    for index , row in df.iterrows():
        song_path = row['track_id']
        file_path = os.path.join(folder_path, song_path)
        if not os.path.exists(file_path + '.mp3'):
            print(f"File not found: {file_path}")
            rows_to_remove.append(index)
        else:
            print(f"File exists: {file_path}")
    df_cleaned = df.drop(rows_to_remove)
    df_cleaned.to_csv(csv_file, index=False)

def split_df(csv_file):

    df = pd.read_csv(csv_file)
    select_1480 = df.sample(n = 1480, random_state = 42)
    remaining_20 = df.drop(select_1480.index)

    select_1480.to_csv('Data/Train_emb.csv')
    remaining_20.to_csv('Data/Eval_emb.csv')

class SpectrogramDataset(Dataset):

    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        #df = df.iloc[:5]
        self.track_id = df['track_id']

    def __len__(self):
        return len(self.track_id)
    
    def __getitem__(self, index):
        signal = self._get_spectrogram(self.track_id[index])
        return signal
    
    def _get_spectrogram(self, song_path):
        
        full_path = 'Data/Proccessed/' + song_path + '.mp3'
        y, _  = librosa.load(full_path, sr = Constants.sample_rate)
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

csv_file = 'Data/Music Info.csv'
#preproccess_mp3_path('Data/MP3-Example')
#move_mp3_external('Data/MP3-Example', 'Data/Proccessed')
#remove_ids(folder_path = 'Data/Proccessed', csv_file = 'Data/Music Info.csv')
#split_df(csv_file)

def main():
    #preproccess_mp3_path('Data/MP3-Example')
    #move_mp3_external('Data/MP3-Example', 'Data/Proccessed')
    #remove_ids(folder_path = 'Data/Proccessed', csv_file = 'Data/Music Info.csv')
    #split_df(csv_file)
    csv_test = "Data/Train_emb.csv"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = SpectrogramDataset(csv_file=csv_test)
    loader = DataLoader(dataset= dataset, batch_size=1, shuffle = False)
    model = CNNNetwork().to(device)
    model_path = 'cnn.pth'
    model = torch.load(model_path, map_location = torch.device('cpu'))
    model.eval()
    model = model.to(device)
    embeddings = []   
    for ix, input in enumerate(loader):
        print(f"Infering {ix}")
        with torch.no_grad():
            fc1_output = F.relu(model.fc1(model.flatten(model.conv4(model.conv3(model.conv2(model.conv1(input)))))))
        embeddings.append(fc1_output)
    embeddings = torch.cat(embeddings, dim=0)
    df = pd.read_csv('Data/Train_emb.csv')
    df = df.drop(['spotify_preview_url', 'spotify_id', 'tags', 'genre'], axis = 1)
    emb_names  = {}
    for i in range(256):
        column_name = f'emb{i+1}'
        emb_names[column_name] = embeddings[:, i]
    embeddings_csv = pd.DataFrame(emb_names)
    total_csv = pd.concat([embeddings_csv, df], axis = 1)
    print(total_csv.head())
    total_csv.to_csv('Data/Autoencoder.csv', index = False)
main()
# csv_file = 'Data/Music Info.csv'
# dataframe = pd.read_csv(csv_file)
# print(dataframe.head())
