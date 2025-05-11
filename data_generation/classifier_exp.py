import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from medmnist import INFO, BloodMNIST
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset

from classifier import ClassifierModel
from generator import Generator


def collate_tensor_pair(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack([torch.as_tensor(img) for img in imgs], dim=0)
    label_tensors = []
    for l in labels:
        if isinstance(l, torch.Tensor):
            label_tensors.append(l.view(1))
        elif isinstance(l, np.ndarray):
            label_tensors.append(torch.from_numpy(l).view(1))
        else:
            label_tensors.append(torch.tensor([l], dtype=torch.long))
    labels = torch.cat(label_tensors, dim=0)
    return imgs, labels


def load_real_datasets(data_flag, size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    train_ds = BloodMNIST(split="train", transform=transform, download=True, size=size)
    test_ds = BloodMNIST(split="test", transform=transform, download=True, size=size)
    return train_ds, test_ds


def make_synthetic_dataset(gen, n_samples, latent_dim, n_classes, device):
    zs = torch.randn(n_samples, latent_dim, device=device)
    ys = torch.randint(0, n_classes, (n_samples,), device=device)
    with torch.no_grad():
        imgs = gen(zs, ys).cpu()
    return TensorDataset(imgs, ys.cpu())


def run_experiment(train_ds, test_ds, n_classes, device, lr, epochs, batch_size):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_tensor_pair)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_tensor_pair)

    model = ClassifierModel(n_classes).to(device)
    for p in model.model.parameters():
        p.requires_grad = False
    for p in model.model.fc.parameters():
        p.requires_grad = True

    opt = optim.Adam(model.model.fc.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        losses = []
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).flatten()
            loss_value = loss_fn(model(x), y)
            opt.zero_grad()
            loss_value.backward()
            opt.step()
            losses.append(loss_value.item())
        avg_loss = sum(losses) / len(losses)
        print(f"Epoch {_ + 1}/{epochs}, Loss: {avg_loss:.4f}")


    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device).flatten()
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    acc = (y_true == y_pred).mean()
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1, "confusion_matrix": cm}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_flag", type=str, default="bloodmnist")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--generator_path", type=str, default="generator.pth")
    p.add_argument("--latent_dim", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_real, test_real = load_real_datasets(args.data_flag, args.size)
    n_classes = len(INFO[args.data_flag]["label"])
    gen = Generator(n_classes, args.latent_dim, 64, 3).to(device)
    gen.load_state_dict(torch.load(args.generator_path, map_location=device))
    gen.eval()

    print(f"Generator loaded from {args.generator_path}")

    syn_train = make_synthetic_dataset(gen, len(train_real), args.latent_dim, n_classes, device)
    print(f"Generated {len(syn_train)} synthetic training samples")
    syn_test = make_synthetic_dataset(gen, len(test_real), args.latent_dim, n_classes, device)
    print(f"Generated {len(syn_test)} synthetic test samples")

    num_real = len(train_real)
    half = num_real // 2
    train_real_sub = Subset(train_real, torch.randperm(num_real)[:half].tolist())
    train_syn_sub = Subset(syn_train, torch.randperm(num_real)[: num_real - half].tolist())
    train_combined = ConcatDataset([train_real_sub, train_syn_sub])

    test_num_real = len(test_real)
    test_half = test_num_real // 2
    test_real_sub = Subset(test_real, torch.randperm(test_num_real)[:test_half].tolist())
    test_syn_sub = Subset(syn_test, torch.randperm(test_num_real)[: test_num_real - test_half].tolist())
    test_combined = ConcatDataset([test_real_sub, test_syn_sub])

    metrics1 = run_experiment(train_real, syn_test, n_classes, device, args.lr, args.epochs, args.batch_size)
    print("Real → Syn:")
    print(
        f"  Acc: {metrics1['accuracy']:.4f},  Prec: {metrics1['precision']:.4f},  Rec: {metrics1['recall']:.4f},  F1: {metrics1['f1_score']:.4f}"
    )
    print("  Confusion matrix:")
    print(metrics1["confusion_matrix"])

    metrics2 = run_experiment(syn_train, test_real, n_classes, device, args.lr, args.epochs, args.batch_size)
    print("Syn → Real:")
    print(
        f"  Acc: {metrics2['accuracy']:.4f},  Prec: {metrics2['precision']:.4f},  Rec: {metrics2['recall']:.4f},  F1: {metrics2['f1_score']:.4f}"
    )
    print("  Confusion matrix:")
    print(metrics2["confusion_matrix"])

    metrics3 = run_experiment(train_combined, test_combined, n_classes, device, args.lr, args.epochs, args.batch_size)
    print("Combined 50/50:")
    print(
        f"  Acc: {metrics3['accuracy']:.4f},  Prec: {metrics3['precision']:.4f},  Rec: {metrics3['recall']:.4f},  F1: {metrics3['f1_score']:.4f}"
    )
    print("  Confusion matrix:")
    print(metrics3["confusion_matrix"])


if __name__ == "__main__":
    main()
