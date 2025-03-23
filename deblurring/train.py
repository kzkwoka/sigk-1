import argparse
import os
from typing import Callable, Tuple

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, Subset

from deblurring.dataset import DeblurringDataset
from deblurring.model import CNN, AutoEncoder


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module, dataloader: DataLoader, loss_fn: Callable[[Tensor, Tensor], Tensor], device: torch.device
) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    optimizer: Optimizer,
    device: torch.device,
    epochs: int,
    save_path: str,
) -> None:
    best_valid_loss = float("inf")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")


def split_dataset(dataset: Dataset, train_ratio: float = 0.8) -> Tuple[Subset, Subset]:
    train_size = int(train_ratio * len(dataset))
    valid_size = len(dataset) - train_size
    train_indices, valid_indices = train_test_split(
        range(len(dataset)), train_size=train_size, test_size=valid_size, random_state=42
    )
    return Subset(dataset, train_indices), Subset(dataset, valid_indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train deblurring model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training images")
    parser.add_argument("--kernel_size", type=int, default=5, help="Kernel size for Gaussian blur")
    parser.add_argument("--img_size", type=int, default=256, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model", type=str, choices=["cnn", "ae"], default="cnn", help="Model type to use")
    parser.add_argument("--save_path", type=str, default="model.pth", help="Path to save trained model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DeblurringDataset(args.data_path, img_size=args.img_size, kernel_size=args.kernel_size)
    train_dataset, valid_dataset = split_dataset(dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    model: nn.Module = CNN() if args.model == "cnn" else AutoEncoder()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    train_model(model, train_loader, valid_loader, criterion, optimizer, device, args.epochs, args.save_path)


if __name__ == "__main__":
    main()
