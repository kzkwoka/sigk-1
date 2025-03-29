import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from torch import nn, Tensor


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

        self.vgg = None

    def forward(self, x_low: Tensor, x_mid: Tensor, x_high: Tensor) -> Tensor:
        # encode
        e_low = self.encoder(x_low)
        e_mid = self.encoder(x_mid)
        e_high = self.encoder(x_high)
        # fusion
        x_concat = torch.cat([e_low, e_mid, e_high], dim=1)
        x_gated = self.fusion_gate(x_concat)
        # decode
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
                "interval": "epoch",  # Step every epoch
                "frequency": 10  # Apply decay every 10 epochs
            }
        }

    def training_step(self, batch, batch_idx):
        x_low, x_mid, x_high, x_mu = batch
        x = self(x_low, x_mid, x_high)
        loss = nn.functional.mse_loss(x, x_mu)  #TODO: add feature contrast masking loss (using vgg)
        self.log('train_loss', loss)
        return loss


if __name__ == '__main__':
    model = TMNet(3)
    # print(model)
    x_low = torch.randn(2, 3, 256, 174)
    x_mid = torch.randn(2, 3, 256, 174)
    x_high = torch.randn(2, 3, 256, 174)
    out = model(x_low, x_mid, x_high)
    print(out.shape)
