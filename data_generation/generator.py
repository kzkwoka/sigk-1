import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, n_classes, latent_dim, img_size, channels):
        super(Generator,self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.init_size = img_size // 16
        self.fc = nn.Linear(latent_dim + n_classes, 512 * self.init_size * self.init_size)
        self.deconv_blocks = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1),
            nn.Tanh()
        )
    def forward(self, noise, labels):
        labels = labels.view(labels.size(0))
        emb = self.label_emb(labels)
        x = torch.cat((noise, emb), dim=1)
        x = self.fc(x)
        x = x.view(x.size(0),512,self.init_size,self.init_size)
        img = self.deconv_blocks(x)
        return img

class Discriminator(nn.Module):
    def __init__(self, n_classes, img_size, channels):
        super(Discriminator,self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.init_size = img_size // 16
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )
        self.adv_layer = nn.Linear(512 * self.init_size * self.init_size + n_classes, 1)
    def forward(self, img, labels):
        labels = labels.view(labels.size(0))
        out = self.conv_blocks(img)
        out = out.view(out.size(0), -1)
        emb = self.label_emb(labels)
        d_in = torch.cat((out, emb), dim=1)
        validity = self.adv_layer(d_in)
        return validity
