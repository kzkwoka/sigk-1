import argparse
import torch
import lightning as L
from torchvision.transforms import InterpolationMode
from dataset import ImagePairDataset
from torchvision import transforms
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from model import SRNet, espcn, srcnn, vdsr
from utils import rgb_to_ycbcr

device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for SRNet")
    parser.add_argument("--model", type=str, default="espcn", help="Model to train")
    parser.add_argument("--use_bicubic", action="store_true", help="Use bicubic for upsize")
    parser.add_argument("--color_space", type=str, default="rgb", help="Used color space. Otherwise RGB used")
    parser.add_argument("--channels", type=int, default=1, help="Number of channels in the image (1 for only Y)")
    parser.add_argument("--input_size", type=int, default=32, help="Resolution for LR images")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--step_size", type=int, default=100, help="Step size for learning rate scheduler")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for learning rate scheduler")
    parser.add_argument("--gradient_clip_val", type=float, default=None, help="Gradient clipping value")
    return parser.parse_args()


def prepare_data(args):
    transform = transforms.Compose([
        # scale the LR image to (shorter side)
        transforms.Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.input_size),

        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC) if args.use_bicubic else transforms.Lambda(lambda x: x),
        transforms.Lambda(rgb_to_ycbcr) if args.color_space == "ycbcr" else transforms.Lambda(lambda x: x),
        transforms.ToTensor()
    ])

    original_transform = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),  # scale the HR image to 256 (shorter side)
        transforms.CenterCrop(256),  # crop the HR image to 256x256
        transforms.Lambda(rgb_to_ycbcr) if args.color_space == "ycbcr" else transforms.Lambda(lambda x: x),
        transforms.ToTensor()
    ])
    dataset = ImagePairDataset(image_dir='div2k/DIV2K_train_HR', input_transform=transform,
                               target_transform=original_transform)
    valid_dataset = ImagePairDataset(image_dir='div2k/DIV2K_valid_HR', input_transform=transform,
                                     target_transform=original_transform)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=15)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=15)
    return train_loader, valid_loader


if __name__ == '__main__':
    args = parse_args()
    net = eval(args.model)
    model = SRNet(net,
                  color_space=args.color_space,
                  model_kwargs={'channels': args.channels, 'upscale_factor': 256 // args.input_size},
                  lr_kwargs={'lr': args.lr, 'step_size': args.step_size, 'gamma': args.gamma})
    train_loader, valid_loader = prepare_data(args)

    logger = WandbLogger(log_model=True, project="SIGK", save_dir="logs")
    logger.watch(model, log="all")
    logger.log_hyperparams(vars(args))
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = L.Trainer(accelerator=device, max_epochs=args.epochs, logger=logger,
                        callbacks=[lr_monitor], log_every_n_steps=1,
                        gradient_clip_val=args.gradient_clip_val)
    trainer.fit(model, train_loader, valid_loader)
