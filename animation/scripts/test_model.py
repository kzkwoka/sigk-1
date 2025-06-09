import argparse
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from animation.src.dataset import AnimationTripletDataset
from animation.src.model import AnimationUNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for test loader")
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels (usually 3)")
    parser.add_argument("--optical_flow", action="store_true", help="Use optical flow as input")
    parser.add_argument("--project", type=str, default="animation-interpolation", help="WandB project name")
    parser.add_argument("--run_name", type=str, default="test_unet_128_last", help="WandB run name")
    return parser.parse_args()


def main(args):
    test_dataset = AnimationTripletDataset(args.test_dir, args.optical_flow)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model = AnimationUNet.load_from_checkpoint(checkpoint_path=args.ckpt_path, channels=args.channels)

    wandb_logger = WandbLogger(project=args.project, name=args.run_name)

    trainer = L.Trainer(logger=wandb_logger)
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    args = parse_args()
    main(args)
