import torch
import lightning as L
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
import wandb
from torchmetrics.functional.image import learned_perceptual_image_patch_similarity as lpips, \
    structural_similarity_index_measure as ssim, peak_signal_noise_ratio as psnr

from utils import ycrcb_to_rgb

torch.set_float32_matmul_precision('high')


def espcn(model_kwargs):
    channels = model_kwargs.get("channels", 1)
    upscale_factor = model_kwargs.get("upscale_factor", 4)
    return nn.Sequential(
        nn.Conv2d(channels, 64, (5, 5), (1, 1), (2, 2)),
        nn.Tanh(),
        nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1)),
        nn.Tanh(),
        nn.Conv2d(32, channels * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1)),
        nn.PixelShuffle(upscale_factor)
    )


def srcnn(model_kwargs):
    channels = model_kwargs.get("channels", 1)
    return nn.Sequential(
        nn.Conv2d(channels, 64, kernel_size=9, stride=1, padding=4),
        nn.ReLU(),
        nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.Conv2d(32, channels, kernel_size=5, stride=1, padding=2)
    )


def vdsr(model_kwargs):
    channels = model_kwargs.get("channels", 1)
    blocks = model_kwargs.get("blocks", 18)
    return nn.Sequential(
        nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1),  # Input layer
        nn.ReLU(),
        *[nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # Hidden layers
            nn.ReLU()
        ) for _ in range(blocks)],
        nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1)  # Output layer
    )


class SRNet(L.LightningModule):
    def __init__(self, model, color_space=None, model_kwargs=None, lr_kwargs=None):
        super().__init__()
        self.model_name = model.__name__
        self.model = model(model_kwargs)
        self.color_space = color_space
        self.model_kwargs = model_kwargs
        self.lr_kwargs = dict(
            lr=1e-2,
            step_size=10,
            gamma=0.5)
        if lr_kwargs:
            self.lr_kwargs.update(lr_kwargs)

    def forward(self, x):
        x = self.model(x)
        # clamp to 0-1
        x = torch.clamp(x, 0.0, 1.0)
        return x

    def _split_layers(self):
        if self.model_name == "espcn":
            n_last = 2
        elif self.model_name == "srcnn":
            n_last = 1
        return self.model[:-n_last], self.model[-n_last:]

    def configure_optimizers(self):
        main, last = self._split_layers()
        param_groups = [
            {"params": main.parameters(), "lr": self.lr_kwargs["lr"]},
            {"params": last.parameters(), "lr": self.lr_kwargs["lr"] * 0.1}
        ]
        optimizer = optim.Adam(param_groups, lr=self.lr_kwargs["lr"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_kwargs["step_size"],
                                              gamma=self.lr_kwargs["gamma"])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "name": "step_lr"
            }}

    def _step(self, inputs, targets):
        # get just y channel
        y_in = inputs[:, 0, :, :].unsqueeze(1) if self.model_kwargs["channels"] == 1 else inputs
        y_target = targets[:, 0, :, :].unsqueeze(1) if self.model_kwargs["channels"] == 1 else targets

        # forward pass
        y_out = self.forward(y_in)

        # reconstruct image
        reconstructed = y_out
        if self.color_space == "ycbcr":
            if self.model_kwargs["channels"] == 1:
                cr = targets[:, 1, :, :]
                cb = targets[:, 2, :, :]
                reconstructed = ycrcb_to_rgb(y_out, cr, cb)
                targets = ycrcb_to_rgb(y_target, cr, cb)
            elif self.model_kwargs["channels"] == 3:
                reconstructed = ycrcb_to_rgb(y_out[:, 0, :, :], y_out[:, 1, :, :], y_out[:, 2, :, :])
                targets = ycrcb_to_rgb(y_target[:, 0, :, :], y_target[:, 1, :, :], y_target[:, 2, :, :])

        return y_out, y_target, reconstructed, targets

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        y_out, y_target, reconstructed, targets = self._step(inputs, targets)
        loss = nn.functional.mse_loss(y_out, y_target)

        self.log("train/loss", loss)
        self.log("train/lpips", lpips(reconstructed, targets, normalize=True))
        self.log("train/psnr", psnr(y_out, y_target))
        self.log("train/ssim", ssim(y_out, y_target))
        grid = vutils.make_grid([reconstructed[0], targets[0]])
        wandb.log({"train/images": [wandb.Image(grid)]})
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        #TODO: use bicubic in validation to simulate testing?
        y_out, y_target, reconstructed, targets = self._step(inputs, targets)
        loss = nn.functional.mse_loss(y_out, y_target)

        self.log("val/loss", loss)
        self.log("val/lpips", lpips(reconstructed, targets, normalize=True))
        self.log("val/psnr", psnr(y_out, y_target))
        self.log("val/ssim", ssim(y_out, y_target))
        grid = vutils.make_grid([reconstructed[0], targets[0]])
        wandb.log({"val/images": [wandb.Image(grid)]})
        return loss
