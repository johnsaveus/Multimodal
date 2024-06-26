import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.transforms as transforms

class CNNModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, activation='relu')
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, activation='relu')
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding='same')
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, activation='relu')
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding='same')
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.3)
        
        # Calculate the size of the flattened feature vector after the convolutional and pooling layers
        self._to_linear = None
        self.convs(torch.randn(1, *input_shape))
        
        self.fc1 = nn.Linear(self._to_linear, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def convs(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.bn1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.bn2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.dropout(x)
        
        if self._to_linear is None:
            self._to_linear = x.view(x.size(0), -1).size(1)
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x
    
class CustomDataset(Dataset):
    def __init__(self, path_to_csv, transform=None):
        super(CustomDataset).__init__()
        self.dataframe = pd.read_csv(path_to_csv, index_col = False)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['Image_path']
        image = Image.open(img_path).convert("RGB")
        transform = transforms.ToTensor()
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add a batch dimension
        label = self.dataframe.iloc[idx]['Label']
        if self.transform:
            image = self.transform(image)
        return image_tensor, label
    
dataset = CustomDataset(path_to_csv = 'Data/splits/val.csv')

image = dataset[0][0]
print(image)