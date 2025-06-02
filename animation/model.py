import torch
import torch.nn as nn
import lightning as L
import torchvision
import wandb

from torchmetrics.image import LearnedPerceptualImagePatchSimilarity as LPIPS, PeakSignalNoiseRatio as PSNR, \
    StructuralSimilarityIndexMeasure as SSIM


class UNet(nn.Module):
    def __init__(self, channels=1):
        super(UNet, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(2 * channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.out_conv = nn.Conv2d(16, channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))

        dec2 = self.decoder2(torch.cat([self.up2(bottleneck), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([self.up1(dec2), enc1], dim=1))
        out = torch.sigmoid(self.out_conv(dec1))
        return out


class AnimationUNet(L.LightningModule):
    def __init__(self, channels=1):
        super(AnimationUNet, self).__init__()
        self.model = UNet(channels=channels)
        # self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()
        self.ssim = SSIM()
        self.psnr = PSNR()

    def forward(self, x):
        return self.model(x)

    def log_eval_metrics(self, output, target, prefix=''):
        ssim = self.ssim(output, target)
        psnr = self.psnr(output, target)
        self.log(f'{prefix}/ssim', ssim, on_epoch=True)
        self.log(f'{prefix}/psnr', psnr, on_epoch=True)

    def log_examples(self, output, target, prefix='', n=8, mode='RGB'):
        sample_imgs = torch.cat((output[:n], target[:n]), dim=0)
        grid = torchvision.transforms.functional.to_pil_image(torchvision.utils.make_grid(sample_imgs,
                                                                                          nrow=n,
                                                                                          pad_value=0,
                                                                                          padding=5
                                                                                          ), mode=mode)
        wandb.log({f'{prefix}/images': wandb.Image(grid, mode=mode)})

    def training_step(self, batch, batch_idx):
        frame0, frame1, frame2 = batch
        input_pair = torch.cat([frame0, frame2], dim=1)
        output = self(input_pair)
        loss = self.criterion(output, frame1)
        self.log('train/loss', loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        frame0, frame1, frame2 = batch
        input_pair = torch.cat([frame0, frame2], dim=1)
        output = self(input_pair)
        loss = self.criterion(output, frame1)
        self.log('val/loss', loss, on_epoch=True)
        self.log_eval_metrics(output, frame1, prefix='val')
        if batch_idx == 0:
            self.log_examples(output, frame1, prefix='val')
        return loss

    def test_step(self, batch, batch_idx):
        frame0, frame1, frame2 = batch
        input_pair = torch.cat([frame0, frame2], dim=1)
        output = self(input_pair)
        loss = self.criterion(output, frame1)
        self.log('test/loss', loss, on_epoch=True)
        self.log_eval_metrics(output, frame1, prefix='test')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
