from cnn_model import CNNNetwork, SpectrogramDataset
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

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
    true_labels = []
    predicted_labels  = []
    for input, target in test_loader:
        with torch.no_grad():
            output = model(input)
        probs = torch.softmax(output, dim = 1)
        preds = torch.argmax(probs, dim = 1)
        true_labels.append(target)
        predicted_labels.append(preds)
    classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    print(classification_report(true_labels, predicted_labels, target_names = classes))
    print(accuracy_score(true_labels, predicted_labels))
    conf = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('Mel_cf.png')
    plt.show()
main()