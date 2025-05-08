import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from models.vgg16_feature_extractor import get_vgg16


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
batch_size = 32
lr = 0.001
epochs = 10
num_classes = 5
patience = 3

# for best model saving
best_val_f1 = 0.0
epochs_without_improvement = 0

# data transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),              
    transforms.RandomResizedCrop(224),           
    transforms.RandomHorizontalFlip(p=0.5),     
    transforms.RandomRotation(15),     
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# # split the dataset into training and validation (80% - 20%)
dataset = datasets.ImageFolder(root='data/train', transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)


vgg16 = get_vgg16(num_classes).to(device)

# loss and optimization
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg16.classifier.parameters(), lr=lr)


os.makedirs("saved_models", exist_ok=True)

start_time = time.time()

for epoch in range(epochs):
    vgg16.train()
    running_loss = 0.0
    train_preds, train_labels = [], []

    for images, labels in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = vgg16(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    average_train_loss = running_loss / len(train_loader)
    acc = accuracy_score(train_labels, train_preds)
    prec = precision_score(train_labels, train_preds, average='macro')
    rec = recall_score(train_labels, train_preds, average='macro')
    f1 = f1_score(train_labels, train_preds, average='macro')

    vgg16.eval()
    val_preds, val_labels = [], []
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = vgg16(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            val_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    val_prec = precision_score(val_labels, val_preds, average='macro')
    val_rec = recall_score(val_labels, val_preds, average='macro')
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    val_loss = val_loss / len(val_loader)

    # save model if F1 improves, else apply early stopping.
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        epochs_without_improvement = 0
        torch.save(vgg16.state_dict(), "saved_models/vgg16_feature_extractor.pth")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}.")
            break

    print(f"\nEpoch [{epoch+1}/{epochs}]")
    print(f"  Train -> Loss: {average_train_loss:.4f}, Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
    print(f"  Val   -> Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")

total_time = time.time() - start_time 
print(f"\nTotal training time: {total_time:.2f} seconds")  

