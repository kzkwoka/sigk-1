import lightning as L
import torch

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


if __name__ == '__main__':
    model = TMNet(3)
    # print(model)
    x_low = torch.randn(2, 3, 256, 174)
    x_mid = torch.randn(2, 3, 256, 174)
    x_high = torch.randn(2, 3, 256, 174)
    out = model(x_low, x_mid, x_high)
    print(out.shape)

