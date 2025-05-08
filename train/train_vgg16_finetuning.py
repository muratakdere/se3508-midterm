import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from models.vgg16_finetuning import get_vgg16_finetuning  

# set the device to GPU, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
batch_size = 32
lr = 0.0001
epochs = 10
num_classes = 5


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# split the dataset into training and validation (80% - 20%)
dataset = datasets.ImageFolder(root='data/train', transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)


vgg16 = get_vgg16_finetuning(num_classes)
vgg16 = vgg16.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, vgg16.parameters()), lr=lr)

# track the best validation accuracy and set up early stopping
best_val_acc = 0.0
patience = 3
patience_counter = 0


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

    average_loss = running_loss / len(train_loader)
    acc = accuracy_score(train_labels, train_preds)
    prec = precision_score(train_labels, train_preds, average='macro')
    rec = recall_score(train_labels, train_preds, average='macro')
    f1 = f1_score(train_labels, train_preds, average='macro')

    # validation
    vgg16.eval()
    val_preds, val_labels = [], []
    val_running_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = vgg16(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            val_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss = val_running_loss / len(val_loader)
    val_acc = accuracy_score(val_labels, val_preds)
    val_prec = precision_score(val_labels, val_preds, average='macro')
    val_rec = recall_score(val_labels, val_preds, average='macro')
    val_f1 = f1_score(val_labels, val_preds, average='macro')

    print(f"\nEpoch [{epoch+1}/{epochs}]")
    print(f"  Train -> Loss: {average_loss:.4f}, Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
    print(f"  Val   -> Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")

    # early stopping and save
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        os.makedirs("saved_models", exist_ok=True)
        torch.save(vgg16.state_dict(), "saved_models/vgg16_finetuning.pth")
        print(f"Model improved, saving to saved_models/vgg16_finetuning.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break


end_time = time.time()  
total_duration = end_time - start_time

print(f"Total training time: {total_duration:.2f} seconds")
