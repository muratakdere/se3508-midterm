import sys
import os
import time     
# add the parent directory to the system path to allow relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score
from models.custom_cnn import CustomCNN

# hyperparameters
batch_size = 32
lr = 0.0005
weight_decay = 1e-4
epochs = 20

    
def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, epochs, device):
    best_f1 = 0.0
    best_model_wts = None
    model.to(device)
    
    start_time = time.time()  # for training period

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)

        model.eval()
        correct_val = 0
        total_val = 0
        running_val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() 
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
                all_preds.extend(preds.cpu())
                all_labels.extend(labels.cpu())

        val_acc = correct_val / total_val
        val_loss = running_val_loss / len(val_loader)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f'Epoch {epoch}/{epochs}')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

        # for best model finding
        if f1 > best_f1:
            best_f1 = f1
            best_model_wts = model.state_dict().copy()
            print(f"New best model found. (F1-score: {best_f1:.4f})")

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    if best_model_wts is not None:
        # save the best model weights to a file if they exist
        os.makedirs('saved_models', exist_ok=True)
        torch.save(best_model_wts, 'saved_models/custom_cnn_best.pth')
        print("The best model is saved in saved_models.")

    if best_model_wts:
        # reload the best model weights into the model
        model.load_state_dict(best_model_wts)

    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # split the dataset into training and validation (80% - 20%)
    dataset = datasets.ImageFolder(root='data/train', transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CustomCNN()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    model = train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, epochs=epochs, device=device)

if __name__ == '__main__':
    main()

    

    

