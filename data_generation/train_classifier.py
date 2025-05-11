import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from medmnist import INFO, BloodMNIST
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader
from torchvision import models

from classifier import ClassifierModel


def load_data(data_flag, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    ds_train = BloodMNIST(split='train', transform=transform, download=True, size=64)
    ds_val = BloodMNIST(split='val', transform=transform, download=True, size=64)
    return DataLoader(ds_train, batch_size=batch_size, shuffle=True), DataLoader(ds_val, batch_size=batch_size, shuffle=False)

def train(model, train_loader, device, optimizer, criterion):
    model.train()
    running_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.flatten().to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
    return running_loss / len(train_loader.dataset)

def evaluate(model, loader, device, criterion):
    model.eval()
    loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.flatten().to(device)
            outputs = model(imgs)
            loss += criterion(outputs, labels).item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    avg_loss = loss / len(loader.dataset)
    acc = (all_preds == all_labels).float().mean().item()
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    return avg_loss, acc, precision, recall, f1, cm

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data_flag', type=str, default='bloodmnist')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--save_path', type=str, default='model.pth')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = load_data(args.data_flag, args.batch_size)
    num_labels = len(INFO[args.data_flag]['label'])
    model = ClassifierModel(num_labels)
    model.to(device)
    for param in model.model.parameters():
        param.requires_grad = False
    for param in model.model.fc.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.model.fc.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, device, optimizer, criterion)
        val_loss, val_acc, val_prec, val_rec, val_f1, cm = evaluate(model, val_loader, device, criterion)
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f" Train Loss: {train_loss:.4f}")
        print(f" Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}")
        print(cm)

    torch.save(model.state_dict(), args.save_path)

