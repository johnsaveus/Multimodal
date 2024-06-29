from cnn_model import CNNNetwork, SpectrogramDataset
import torch
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
def main():
    csv_test = "Data/splits/test.csv"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_dataset = SpectrogramDataset(csv_file=csv_test)
    test_loader = DataLoader(dataset= test_dataset, batch_size=1, shuffle = False)
    model = CNNNetwork().to(device)
    model_path = 'cnn.pth'
    model = torch.load(model_path, map_location = torch.device('cpu'))
    model.eval()
    model = model.to(device)   
    for input, _ in test_loader:
        with torch.no_grad():
            fc1_output = F.relu(model.fc1(model.flatten(model.conv4(model.conv3(model.conv2(model.conv1(input)))))))
        print(fc1_output.shape)
main()