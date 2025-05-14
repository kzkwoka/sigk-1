import argparse
import torch
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import Subset

from models import GAN
from dataset import RenderDataset


device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for GAN")
    parser.add_argument("--channels", type=int, default=3, help="Number of channels in the input images")
    parser.add_argument("--c_dim", type=int, default=10, help="Dimensionality of the condition space")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate for the generator optimizer")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train")
    parser.add_argument("--warmup_epochs", type=int, default=0, help="Number of warmup epochs")
    parser.add_argument("--ngf", type=int, default=64, help="Number of generator filters")
    parser.add_argument("--ndf", type=int, default=64, help="Number of discriminator filters")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    model = GAN(channels=args.channels, condition_dim=args.c_dim, lr=args.lr,
                ngf=args.ngf, ndf=args.ndf,
                warmup_epochs=args.warmup_epochs,
                bg_weight=1, lambda_l1=100)
    dataset = RenderDataset("dataset_normal_max/")

    train_set, valid_set, _ = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1], torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=15)
    val_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size*8, shuffle=False, num_workers=15)

    logger = WandbLogger(log_model=True, project="SIGK-4", save_dir="logs")
    logger.watch(model, log="all")
    logger.log_hyperparams(vars(args))
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"ckpts/{logger.experiment.name}/",
        monitor="val/l1_loss",
        mode="min",
        save_top_k=1,
        every_n_epochs=1,
        filename="best-model",
        save_weights_only=True
    )

    trainer = L.Trainer(accelerator=device,
                        max_epochs=args.epochs,
                        logger=logger,
                        callbacks=[lr_monitor, checkpoint_callback],
                        log_every_n_steps=1)

    trainer.fit(model, train_loader, val_loader)
