import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torchvision
import wandb

from utils import weights_init

from flip_evaluator import evaluate as flip


class Generator(nn.Module):
    def __init__(self, ngf=64, nc=4, z_dim=100, c_dim=10):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim + c_dim, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=4, c_dim=10):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(nc + c_dim, ndf, 4, 2, 1, bias=False),
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
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, x, c):
        x = torch.cat([x, c], dim=1)
        return self.model(x)


class GAN(L.LightningModule):
    def __init__(
            self,
            channels,
            # width,
            # height,
            latent_dim: int = 100,
            condition_dim: int = 10,
            lr: float = 0.0002,
            b1: float = 0.5,
            b2: float = 0.999,
            batch_size: int = 64,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.generator = Generator(z_dim=self.hparams.latent_dim, c_dim=self.hparams.condition_dim,
                                   nc=self.hparams.channels, ngf=self.hparams.ngf)
        self.discriminator = Discriminator(c_dim=self.hparams.condition_dim, nc=self.hparams.channels,
                                           ndf=self.hparams.ndf)

        # self.validation_z = torch.randn(8, self.hparams.latent_dim)

        # self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, z, c):
        return self.generator(z, c)

    def adversarial_loss(self, y_hat, y, reduction='mean'):
        return F.binary_cross_entropy(y_hat, y, reduction=reduction)

    def _train_generator(self, z, imgs, conds, optimizer_g, lambda_l1=100):
        # generate images
        self.toggle_optimizer(optimizer_g)
        conds = conds.view(conds.size(0), -1, 1, 1)
        # visibility_mask = torch.where(
        #     imgs.sum() > 0,
        #     torch.full((imgs.size(0),), 2.0, device=imgs.device),
        #     torch.full((imgs.size(0),), 0.2, device=imgs.device)
        # )
        visibility_mask = 1
        self.generated_imgs = self(z, conds)

        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        conds = conds.expand(-1, -1, imgs.size(2), imgs.size(3))
        g_loss = self.adversarial_loss(self.discriminator(self.generated_imgs, conds).squeeze(-1).squeeze(-1), valid,
                                       'none')

        self.log("train/g_loss", g_loss.mean(), prog_bar=True)

        l1_loss = F.l1_loss(self.generated_imgs, imgs[:, :self.hparams.channels], reduction='none').mean(dim=[1, 2, 3])
        self.log("train/g_l1_loss", l1_loss.mean(), prog_bar=True)

        g_loss = (g_loss * visibility_mask).mean() + lambda_l1 * (l1_loss * visibility_mask).mean()
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()

        self.untoggle_optimizer(optimizer_g)

    def _train_discriminator(self, imgs, conds, optimizer_d):
        self.toggle_optimizer(optimizer_d)

        conds = conds.view(conds.size(0), conds.size(1), 1, 1).repeat(1, 1, imgs.size(2), imgs.size(3))

        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)
        real_loss = self.adversarial_loss(
            self.discriminator(imgs[:, :self.hparams.channels], conds).squeeze(-1).squeeze(-1), valid)
        self.log("train/d_real_loss", real_loss)

        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)
        fake_loss = self.adversarial_loss(
            self.discriminator(self.generated_imgs.detach(), conds).squeeze(-1).squeeze(-1), fake)
        self.log("train/d_fake_loss", fake_loss)

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

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim, 1, 1)
        z = z.type_as(imgs)

        # train generator
        self._train_generator(z, imgs, conds, optimizer_g)

        # train discriminator
        self._train_discriminator(imgs, conds, optimizer_d)

        # log sampled images
        if batch_idx % 10 == 0:
            if self.hparams.channels == 3:
                mode = 'RGB'
                sample_imgs = torch.cat((self.generated_imgs[:6], imgs[:6, :3]), dim=0)
            else:
                mode = 'RGBA'
                sample_imgs = torch.cat((self.generated_imgs[:6], imgs[:6]), dim=0)
            grid = torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(sample_imgs, nrow=6))
            wandb.log({"train/images": wandb.Image(grid, mode=mode)})

    def validation_step(self, batch, batch_idx):
        imgs, conds = batch
        conds = conds.type_as(imgs)

        z = torch.randn(imgs.shape[0], self.hparams.latent_dim, 1, 1)
        z = z.type_as(imgs)

        conds = conds.view(conds.size(0), -1, 1, 1)

        self.generated_imgs = self(z, conds)

        l1_loss = F.l1_loss(self.generated_imgs, imgs[:, :self.hparams.channels])
        self.log("val/l1_loss", l1_loss, prog_bar=True, on_epoch=True)

        flip_scores = []
        for pred, target in zip(self.generated_imgs, imgs):
            # FLIP expects numpy RGB, HWC in [0,1], float32
            pred_np = pred[:3].permute(1, 2, 0).cpu().numpy().clip(0, 1).astype('float32')
            target_np = target[:3].permute(1, 2, 0).cpu().numpy().clip(0, 1).astype('float32')
            flip_scores.append(flip(target_np, pred_np, 'LDR')[1])

        mean_flip = sum(flip_scores) / len(flip_scores)
        self.log("val/flip", mean_flip, prog_bar=True, on_epoch=True)

        # log sampled images
        if batch_idx % 10 == 0:
            if self.hparams.channels == 3:
                mode = 'RGB'
                sample_imgs = torch.cat((self.generated_imgs[:6], imgs[:6, :3]), dim=0)
            else:
                mode = 'RGBA'
                sample_imgs = torch.cat((self.generated_imgs[:6], imgs[:6]), dim=0)
            grid = torchvision.utils.make_grid(sample_imgs, nrow=6)
            wandb.log({"val/images": wandb.Image(grid, mode=mode)})

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr / 10, betas=(b1, b2))
        return [opt_g, opt_d], []

    # def on_validation_epoch_end(self):
    #     z = self.validation_z.type_as(self.generator.model[0].weight)

    #     # log sampled images
    #     sample_imgs = self(z, params)
    #     grid = torchvision.utils.make_grid(sample_imgs)
    #     self.logger.experiment.add_image("validation/generated_images", grid, self.current_epoch)
