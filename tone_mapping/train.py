import argparse
import torch
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from dataset import TMDataset
from model import TMNet

device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for SRNet")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    model = TMNet(3)
    dataset_train = TMDataset('tone_mapping/sihdr/split/train')
    dataset_valid = TMDataset('tone_mapping/sihdr/split/valid')

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=15)
    val_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=15)

    logger = WandbLogger(log_model=True, project="SIGK-2", save_dir="logs")
    logger.watch(model, log="all")
    logger.log_hyperparams(vars(args))
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = L.Trainer(accelerator=device,
                        max_epochs=args.epochs,
                        logger=logger,
                        callbacks=[lr_monitor],
                        log_every_n_steps=1)

    trainer.fit(model, train_loader, val_loader)
