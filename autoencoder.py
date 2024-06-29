import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Embedding_dataset(Dataset):

    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.data = df.drop(['Unnamed: 0', 'track_id', 'name', 'artist'], axis = 1)
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return torch.tensor(self.data[index])

    def feature_len(self):
        return self.data.shape[1]

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid() 
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_latent_vector(self, x):
        return self.encoder(x)

if __name__ == '__main__':
    dataset = Embedding_dataset(csv_file = 'Data/Autoencoder.csv')
    dataloader = DataLoader(dataset, batch_size=248, shuffle=True)
    
    model = Autoencoder(input_dim = 270, latent_dim = 32)
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 500
    model.train()

    for epoch in range(num_epochs):
        print(epoch)
        losses = 0
        for features in dataloader:
            features = features.to(torch.float32)  # Ensure the right data type
            outputs = model(features)
            loss = criterion(outputs, features)
            optimizer.zero_grad()
            losses+=loss.item()
            loss.backward()
            optimizer.step()
        print(losses / len(dataloader))
    torch.save(model, 'autoencoder_model.pth')