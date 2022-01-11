import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Identity
import torchvision
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, n_dim_latent):
        super(Decoder, self).__init__()

        self.bn3 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn1 = nn.BatchNorm2d(128)

        self.n_dim_latent = n_dim_latent
        self.transconv1 = torch.nn.ConvTranspose2d(
            in_channels=128, kernel_size=(3, 3), stride=2, padding=1, output_padding=1, out_channels=128)

        self.transconv2 = torch.nn.ConvTranspose2d(
            in_channels=128, kernel_size=(3, 3), stride=2, padding=1, output_padding=1, out_channels=64)

        self.transconv3 = torch.nn.ConvTranspose2d(
            in_channels=64, kernel_size=(3, 3), stride=2, padding=1, output_padding=1, out_channels=32)

        self.transconv4 = torch.nn.ConvTranspose2d(
            in_channels=32, kernel_size=(3, 3), stride=2, padding=1, output_padding=1, out_channels=3)

        self.Relu = nn.ReLU()
        self.sig = nn.Sigmoid()

        self.linear = nn.Linear(in_features=n_dim_latent, out_features=1024)
        self.linear2 = nn.Linear(in_features=1024, out_features=4608)

    def forward(self, x):
        x = self.Relu(self.linear(x))
        x = self.Relu(self.linear2(x))
        x = x.view(x.size(0), 128, 6, 6)

        x = self.Relu(self.bn1(self.transconv1(x)))
        x = self.Relu(self.bn2(self.transconv2(x)))
        x = self.Relu(self.bn3(self.transconv3(x)))
        x = self.sig(self.transconv4(x))

        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.resnet18 = torchvision.models.resnet18(pretrained=False)
        self.resnet18.fc = Identity()

    def forward(self, x):
        x = self.resnet18(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, n_dim_latent):
        super(AutoEncoder, self).__init__()
        self.n_dim_latent = n_dim_latent
        self.encoder = Encoder(n_dim_latent)
        self.decoder = Decoder(n_dim_latent)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
