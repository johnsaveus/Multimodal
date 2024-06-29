from autoencoder import Autoencoder ,Embedding_dataset
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model_path = 'autoencoder_model.pth'
model = torch.load(model_path, map_location = torch.device('cpu'))

dataset = Embedding_dataset(csv_file = 'Data/Autoencoder.csv')
dataloader = DataLoader(dataset = dataset, batch_size = 1, shuffle = False)

embeddings = []
for ix, features in enumerate(dataloader):
    print(ix)
    features = features.to(torch.float32)
    outputs = model.get_latent_vector(features).squeeze()
    embeddings.append(outputs.detach().numpy())

columns = [f'emb{i+1}' for i in range(64)]
embeddings_csv = pd.DataFrame(embeddings, columns = columns)
df_for_names = pd.read_csv('Data/Train_emb.csv') ###########
names = df_for_names[['name']]
artist = df_for_names[['artist']]
embeddings_csv.insert(0, 'song_name', names)
np.random.seed(42)
torch.manual_seed(42)

random_20 = embeddings_csv.sample(n = 20, random_state = 42)
remaining_songs = embeddings_csv.drop(random_20.index)

similarities = cosine_similarity(random_20.iloc[:, 1:], remaining_songs.iloc[:, 1:])

top_similar = {}
cosine_values = {}

for i, random_song in enumerate(random_20['song_name']):
    sim_scores = similarities[i]
    top_indices = np.argsort(sim_scores)[-5:][::-1]  
    top_similar[random_song] = remaining_songs.iloc[top_indices]['song_name'].values
    cosine_values[random_song] = sim_scores[top_indices]

top_similar_songs_df = pd.DataFrame(top_similar)
top_similar_songs_df.index = [f'Top {i+1}' for i in range(5)]
top_similar_songs_df.to_csv('Top_10_songs.csv', index = False)

# cosine_values_df = pd.DataFrame(cosine_values)
# cosine_values_df.index = [f'Cosine Similarity {i+1}' for i in range(5)]
#cosine_values_df.to_csv('Top_10_similarities.csv', index = False)
# Display the DataFrames
print("Top Similar Songs:")
print(top_similar_songs_df)

# print("\nCosine Similarity Values:")
# print(cosine_values_df)
