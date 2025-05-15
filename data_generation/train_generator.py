import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from medmnist import INFO, BloodMNIST
from torch.utils.data import DataLoader

from generator import Discriminator, Generator


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(data_flag: str, batch_size: int):
    info = INFO[data_flag]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    ds_train = BloodMNIST(split="train", transform=transform, download=True, size=64)
    ds_val = BloodMNIST(split="val", transform=transform, download=True, size=64)
    return DataLoader(ds_train, batch_size=batch_size, shuffle=True), DataLoader(
        ds_val, batch_size=batch_size, shuffle=False
    )


def train_one_epoch(gen, dis, loader, optim_G, optim_D, criterion, latent_dim, n_classes, device):
    gen.train()
    dis.train()
    g_losses = []
    d_losses = []
    for real_imgs, labels in loader:
        b = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        labels = labels.to(device)
        real_target = torch.ones(b, 1, device=device)
        fake_target = torch.zeros(b, 1, device=device)
        for _ in range(2):
            optim_G.zero_grad()
            noise = torch.randn(b, latent_dim, device=device)
            gl = torch.randint(0, n_classes, (b,), device=device)
            gen_imgs = gen(noise, gl)
            g_loss = criterion(dis(gen_imgs, gl), real_target)
            g_loss.backward()
            optim_G.step()
            g_losses.append(g_loss.item())
        optim_D.zero_grad()
        real_loss = criterion(dis(real_imgs, labels), real_target)
        noise = torch.randn(b, latent_dim, device=device)
        gl = torch.randint(0, n_classes, (b,), device=device)
        fake_imgs = gen(noise, gl).detach()
        fake_loss = criterion(dis(fake_imgs, gl), fake_target)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optim_D.step()
        d_losses.append(d_loss.item())
    return sum(g_losses) / len(g_losses), sum(d_losses) / len(d_losses)


def evaluate(gen, dis, loader, criterion, latent_dim, n_classes, device, seed=42):
    gen.eval()
    dis.eval()
    g_losses = []
    d_losses = []

    torch.manual_seed(seed)
    fixed_gen = torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad():
        for real_imgs, labels in loader:
            b = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            real_target = torch.ones(b, 1, device=device)
            fake_target = torch.zeros(b, 1, device=device)

            noise = torch.randn(b, latent_dim, generator=fixed_gen, device=device)
            gl = torch.randint(0, n_classes, (b,), generator=fixed_gen, device=device)

            gen_imgs = gen(noise, gl)
            g_loss = criterion(dis(gen_imgs, gl), real_target)
            g_losses.append(g_loss.item())

            real_loss = criterion(dis(real_imgs, labels), real_target)
            fake_loss = criterion(dis(gen_imgs, gl), fake_target)
            d_losses.append(((real_loss + fake_loss) / 2).item())

    return sum(g_losses) / len(g_losses), sum(d_losses) / len(d_losses)


def train_model(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = load_data(args.data_flag, args.batch_size)
    gen = Generator(len(INFO[args.data_flag]["label"]), args.latent_dim, 64, 3).to(device)
    dis = Discriminator(len(INFO[args.data_flag]["label"]), 64, 3).to(device)
    optim_G = optim.Adam(gen.parameters(), lr=args.lr_G, betas=(args.beta1, args.beta2))
    optim_D = optim.Adam(dis.parameters(), lr=args.lr_D, betas=(args.beta1, args.beta2))
    criterion = nn.BCEWithLogitsLoss()
    n_classes = len(INFO[args.data_flag]["label"])
    for epoch in range(args.epochs):
        g_tr, d_tr = train_one_epoch(
            gen, dis, train_loader, optim_G, optim_D, criterion, args.latent_dim, n_classes, device
        )
        g_val, d_val = evaluate(gen, dis, val_loader, criterion, args.latent_dim, n_classes, device)
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"D_train: {d_tr:.4f} | G_train: {g_tr:.4f} | "
            f"D_val: {d_val:.4f} | G_val: {g_val:.4f}"
        )
    torch.save(gen.state_dict(), f"{args.save_path}/generator.pth")
    torch.save(dis.state_dict(), f"{args.save_path}/discriminator.pth")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_flag", type=str, default="bloodmnist")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--latent_dim", type=int, default=100)
    p.add_argument("--lr_G", type=float, default=2e-4)
    p.add_argument("--lr_D", type=float, default=4e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.9)
    p.add_argument("--save_path", type=str, default=".")
    args = p.parse_args()
    train_model(args)
