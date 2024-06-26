import pandas as pd
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os

def create_spectrogram(split_path, save_path, split):
    df = pd.read_csv(split_path)
    path = df['Path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = save_path + '/' + split + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_paths = []
    for ix, song_path in enumerate(path):
        y, sr = librosa.load(song_path, sr = None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, fmax=8000)
        plt.axis('off')
        plt.tight_layout(pad=0)
        image_path = save_path + str(ix) + 'mel_spectrogram.png'
        plt.savefig(image_path , bbox_inches='tight', pad_inches=0)
        plt.close()
        image_paths.append(image_path)
    df['Image_path'] = image_paths
    df.to_csv(split_path, index = False)

#create_spectrogram('Data/splits/train.csv', 'Data/images', 'train')
#create_spectrogram('Data/splits/val.csv', 'Data/images', 'val')
#create_spectrogram('Data/splits/test.csv', 'Data/images', 'test')
