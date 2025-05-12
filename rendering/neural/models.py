import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torchvision
import wandb

from utils import weights_init

from flip_evaluator import evaluate as flip


class Generator(nn.Module):
    def __init__(self, ngf=64, nc=4, c_dim=10):
        super(Generator, self).__init__()
        self.ngf = ngf
        self.fc = nn.Sequential(
            nn.Linear(c_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, ngf * 4 * 8 * 8),
            nn.ReLU(True)
        )
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 8 → 16
            nn.Conv2d(ngf * 4, ngf * 2, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(ngf * 2),

            nn.Upsample(scale_factor=2),  # 16 → 32
            nn.Conv2d(ngf * 2, ngf, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(ngf),

            nn.Upsample(scale_factor=2),  # 32 → 64
            nn.Conv2d(ngf, ngf // 2, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(ngf // 2),

            nn.Upsample(scale_factor=2),  # 64 → 128
            nn.Conv2d(ngf // 2, nc, 3, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )

        self.apply(weights_init)

    def forward(self, c):
        x = self.fc(c)
        x = x.view(-1, self.ngf * 4, 8, 8)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=4):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, x):
        return self.model(x)


class GAN(L.LightningModule):
    def __init__(
            self,
            channels,
            condition_dim: int = 10,
            lr: float = 0.0002,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.generator = Generator(c_dim=condition_dim, nc=channels, ngf=self.hparams.ngf)
        self.discriminator = Discriminator(nc=channels, ndf=self.hparams.ndf)

    def forward(self, c):
        return self.generator(c)

    def adversarial_loss(self, y_hat, y, reduction='mean'):
        return F.mse_loss(y_hat, y, reduction=reduction)
        # return F.binary_cross_entropy(y_hat, y, reduction=reduction)

    def on_train_epoch_start(self):
        total_epochs = self.trainer.max_epochs
        progress = self.current_epoch / total_epochs

        self.lambda_l1 = self.hparams.lambda_l1
        self.lambda_adv = 1

        # self.lambda_l1 = self.hparams.lambda_l1 * (1.0 - progress)  # from 100 → 0
        # self.lambda_adv = 1.0 * progress  # from 0 → 1.0

        self.log("train/lambda_l1", self.lambda_l1)
        self.log("train/lambda_adv", self.lambda_adv)

    def _train_generator(self, imgs, optimizer_g, warmup=False):
        # generate images
        self.toggle_optimizer(optimizer_g)

        valid = torch.full((imgs.size(0), 1), 0.9)  # Smooth labels
        valid = valid.type_as(imgs)
        target = 2 * imgs[:, :self.hparams.channels] - 1  # Using tanh() so [-1, 1]
        alpha = imgs[:, self.hparams.channels]

        # Weighted L1 loss
        l1_perpixel = F.l1_loss(self.generated_imgs, target, reduction='none').mean(dim=1)
        fg_loss = (l1_perpixel * alpha).sum() / (alpha.sum() + 1e-6)
        bg_loss = (l1_perpixel * (1 - alpha)).sum() / ((1 - alpha).sum() + 1e-6)
        l1_loss = fg_loss + self.hparams.bg_weight * bg_loss
        self.log("train/g_l1_loss", l1_loss)

        if not warmup:
            # Adversarial loss
            g_loss = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)
            self.log("train/g_adv_loss", g_loss)

            # Total loss
            g_loss = self.lambda_adv * g_loss + self.lambda_l1 * l1_loss
        else:
            g_loss = self.lambda_l1 * l1_loss
        self.log("train/g_loss", g_loss, prog_bar=True)

        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()

        self.untoggle_optimizer(optimizer_g)

    def _train_discriminator(self, imgs, optimizer_d):
        self.toggle_optimizer(optimizer_d)

        # Real images loss
        valid = torch.full((imgs.size(0), 1), 0.9)  # Smooth labels
        valid = valid.type_as(imgs)
        real_loss = self.adversarial_loss(self.discriminator(imgs[:, :self.hparams.channels]), valid)
        self.log("train/d_real_loss", real_loss)
        # Fake images loss
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)
        fake_loss = self.adversarial_loss(self.discriminator(self.generated_imgs.detach()), fake)
        self.log("train/d_fake_loss", fake_loss)
        # Total loss
        d_loss = (real_loss + fake_loss) / 2
        self.log("train/d_loss", d_loss, prog_bar=True)

        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()

        self.untoggle_optimizer(optimizer_d)

    def training_step(self, batch, batch_idx):
        imgs, conds = batch
        conds = conds.type_as(imgs)

        optimizer_g, optimizer_d = self.optimizers()

        # conds = conds.view(conds.size(0), -1, 1, 1)
        self.generated_imgs = self(conds)

        if self.current_epoch < self.hparams.warmup_epochs:
            # Warmup generator
            self._train_generator(imgs, optimizer_g, warmup=True)
        else:
            # train discriminator
            self._train_discriminator(imgs, optimizer_d)

            # train generator
            self._train_generator(imgs, optimizer_g)

        # log sampled images
        if batch_idx % 10 == 0:
            self._plot_batch(imgs, "train/images")

    def validation_step(self, batch, batch_idx):
        imgs, conds = batch
        conds = conds.type_as(imgs)

        self.generated_imgs = self(conds)

        l1_loss = F.l1_loss(self.generated_imgs, 2 * imgs[:, :self.hparams.channels] - 1)
        self.log("val/l1_loss", l1_loss, prog_bar=True, on_epoch=True)

        flip_scores = []
        for pred, target in zip(self.generated_imgs, imgs):
            # FLIP expects numpy RGB, HWC in [0,1], float32
            pred = (pred + 1) / 2  # to [0,1]
            pred_np = pred[:3].permute(1, 2, 0).cpu().numpy().clip(0, 1).astype('float32')
            target_np = target[:3].permute(1, 2, 0).cpu().numpy().clip(0, 1).astype('float32')
            flip_scores.append(flip(target_np, pred_np, 'LDR')[1])

        mean_flip = sum(flip_scores) / len(flip_scores)
        self.log("val/flip", mean_flip, prog_bar=True, on_epoch=True)

        # log sampled images
        if batch_idx % 10 == 0:
            self._plot_batch(imgs, "val/images")

    def _plot_batch(self, imgs, name, n=6):
        gen_imgs = (self.generated_imgs[:n] + 1)/2
        if self.hparams.channels == 3:
            mode = 'RGB'
            sample_imgs = torch.cat((gen_imgs, imgs[:n, :3]), dim=0)
        else:
            mode = 'RGBA'
            sample_imgs = torch.cat((gen_imgs, imgs[:n]), dim=0)
        grid = torchvision.transforms.functional.to_pil_image(torchvision.utils.make_grid(sample_imgs,
                                                                                          nrow=n,
                                                                                          pad_value=1,
                                                                                          padding=5
                                                                                          ), mode=mode)
        wandb.log({name: wandb.Image(grid, mode=mode)})

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = 0.5
        b2 = 0.999

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr / 10, betas=(b1, b2))
        return [opt_g, opt_d], []
