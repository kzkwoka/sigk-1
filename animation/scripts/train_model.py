import argparse
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from src.dataset import AnimationTripletDataset
from src.model import AnimationUNet
import torch

# Enable float32 matmul precision optimization for better tensor core performance
torch.set_float32_matmul_precision('high')
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="frames_merged/train", help="Path to training data")
    parser.add_argument("--val_dir", type=str, default="frames_merged/valid", help="Path to validation data")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--channels", type=int, default=3, help="Input/output image channels")
    parser.add_argument("--optical_flow", action="store_true", help="Use optical flow as input")
    parser.add_argument("--project", type=str, default="animation-interpolation", help="WandB project name")
    parser.add_argument("--run_name", type=str, default="unet", help="WandB run name")
    return parser.parse_args()


def main(args):
    # Datasets and loaders
    train_dataset = AnimationTripletDataset(args.train_dir, args.optical_flow)
    val_dataset = AnimationTripletDataset(args.val_dir, args.optical_flow)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = AnimationUNet(channels=args.channels, optical_flow=args.optical_flow)

    # Logger
    wandb_logger = WandbLogger(project=args.project, name=args.run_name)

    # Trainer
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        verbose=True,
        dirpath="checkpoints",
        filename="{epoch:02d}-{val_loss:.4f}",
    )

    trainer = L.Trainer(
        accelerator="auto",  # Use GPU if available
        devices=1,  # Use 1 GPU
        precision=16,  # Use mixed precision
        strategy="auto",  # Use DDP if multiple GPUs are available
        max_epochs=args.epochs, logger=wandb_logger, log_every_n_steps=10, callbacks=[checkpoint_callback]
    )

    # Fit
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    args = parse_args()
    main(args)
