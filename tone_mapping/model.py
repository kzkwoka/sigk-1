import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.models import vgg19


def gaussian_kernel_cdf(filter_size: int = 13, sigma: float = 2.0, channels: int = 64) -> torch.Tensor:
    import numpy as np
    import scipy.stats as st

    interval = (2 * sigma + 1.) / filter_size
    ll = np.linspace(-sigma - interval / 2., sigma + interval / 2., filter_size + 1)
    kern1d = np.diff(st.norm.cdf(ll))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    kernel = torch.from_numpy(kernel).float()
    kernel = kernel.view(1, 1, filter_size, filter_size)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel


def local_mean_std(x: Tensor, sigma: float = 2.0, kernel_size: int = 13) -> tuple[Tensor, Tensor, Tensor]:
    B, C, H, W = x.shape
    weight = gaussian_kernel_cdf(kernel_size, sigma, C).to(x.device)
    mean = F.conv2d(x, weight, padding=kernel_size // 2, groups=C)
    mean_sq = F.conv2d(x ** 2, weight, padding=kernel_size // 2, groups=C)
    std = (mean_sq - mean ** 2).clamp(min=1e-6).sqrt()
    return mean, std, mean


def sign_num_den(x: Tensor, gamma: float, beta: float, sigma: float = 2.0, kernel_size: int = 13) -> tuple[Tensor, Tensor]:
    mean, std, mean_box = local_mean_std(x, sigma, kernel_size)
    norm = (x - mean) / (mean.abs() + 1e-6)
    sign = torch.sign(norm)
    norm = torch.pow(norm.abs() + 1e-6, gamma)
    num = sign * norm
    den = torch.pow(std / (mean_box.abs() + 1e-6), beta)
    return num, 1 + den


def feature_contrast_masking(x: Tensor, gamma: float, beta: float, sigma: float = 2.0, kernel_size: int = 13) -> Tensor:
    num, den = sign_num_den(x, gamma, beta, sigma, kernel_size)
    return num / den


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features.eval()
        self.vgg11 = nn.Sequential(*vgg[:2])       # conv1_1
        self.vgg21 = nn.Sequential(*vgg[2:7])       # conv2_1
        self.vgg31 = nn.Sequential(*vgg[7:12])      # conv3_1
        self.vgg41 = nn.Sequential(*vgg[12:21])     # conv4_1
        self.vgg51 = nn.Sequential(*vgg[21:30])     # conv5_1
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: Tensor) -> list[Tensor]:
        x1 = self.vgg11(x)
        x2 = self.vgg21(x1)
        x3 = self.vgg31(x2)
        x4 = self.vgg41(x3)
        x5 = self.vgg51(x4)
        return [x1, x2, x3] #, x4, x5] # return only the first 3 layers


def FCM_loss(x: Tensor, target: Tensor, feat_ext: FeatureExtractor, gamma=0.5, beta=0.5) -> Tensor:
    x_feats = feat_ext(x)
    target_feats = feat_ext(target)
    loss = 0
    for xf, tf in zip(x_feats, target_feats):
        mask_x = feature_contrast_masking(xf, 1.0, beta)
        mask_t = feature_contrast_masking(tf, gamma, beta)
        loss += F.l1_loss(mask_x, mask_t)
    return loss / len(x_feats)


class TMNet(L.LightningModule):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=0)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1),
        )
        self.sigmoid = nn.Sigmoid()
        self.feat_ext = FeatureExtractor()

    def forward(self, x_low: Tensor, x_mid: Tensor, x_high: Tensor) -> Tensor:
        # encode
        e_low = self.encoder(x_low)
        e_mid = self.encoder(x_mid)
        e_high = self.encoder(x_high)
        # fusion
        x_concat = torch.cat([e_low, e_mid, e_high], dim=1)
        x_gated = self.fusion_gate(x_concat)
        x_res = self.decoder(x_gated)
        # residual
        x = x_res + x_low + x_mid + x_high
        x = self.sigmoid(x)
        return x

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 10
            }
        }

    def training_step(self, batch, batch_idx):
        x_low, x_mid, x_high, x_mu, _ = batch
        x = self(x_low, x_mid, x_high)
        loss = FCM_loss(x, x_mu, self.feat_ext, gamma=0.5, beta=0.5)
        self.log('train/loss', loss)
        return loss


if __name__ == '__main__':
    model = TMNet(3)
    model.train()
    x_low = torch.randn(2, 3, 256, 174)
    x_mid = torch.randn(2, 3, 256, 174)
    x_high = torch.randn(2, 3, 256, 174)
    x_mu = torch.randn(2, 3, 256, 174)
    batch = (x_low, x_mid, x_high, x_mu)
    loss = model.training_step(batch, 0)
    print(f"Output loss: {loss.item()}")
