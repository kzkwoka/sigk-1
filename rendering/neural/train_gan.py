import argparse
import torch
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import Subset

from models import GAN
from dataset import RenderDataset


device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for SRNet")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--channels", type=int, default=4, help="Number of channels in the input images")
    parser.add_argument("--z_dim", type=int, default=1000, help="Dimensionality of the latent space")
    parser.add_argument("--c_dim", type=int, default=10, help="Dimensionality of the condition space")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    model = GAN(channels=args.channels, z_dim=args.z_dim, c_dim=args.c_dim, lr=args.lr, ndf=16, ngf=64)
    dataset = RenderDataset("dataset_normal_max/")

    train_set, valid_set = torch.utils.data.random_split(dataset, [0.8, 0.2], torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    logger = WandbLogger(log_model=True, project="SIGK-4", save_dir="logs")
    logger.watch(model, log="all")
    logger.log_hyperparams(vars(args))
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = L.Trainer(accelerator=device,
                        max_epochs=args.epochs,
                        logger=logger,
                        callbacks=[lr_monitor],
                        log_every_n_steps=1)

    trainer.fit(model, train_loader, val_loader)
