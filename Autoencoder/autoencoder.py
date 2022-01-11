import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
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
    def __init__(self, n_dim_latent):
        super(Encoder, self).__init__()
        self.n_dim_latent = n_dim_latent

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.Relu = nn.ReLU()
        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv1 = torch.nn.Conv2d(
            in_channels=3, kernel_size=(5, 5), out_channels=32, padding='same')

        self.conv2 = torch.nn.Conv2d(
            in_channels=32, kernel_size=(4, 4), out_channels=64, padding='same')

        self.conv3 = torch.nn.Conv2d(
            in_channels=64, kernel_size=(4, 4), out_channels=128, padding='same')

        self.conv4 = torch.nn.Conv2d(
            in_channels=128, kernel_size=(3, 3), out_channels=128, padding='same')

        self.linear1 = nn.Linear(in_features=4608, out_features=1024)
        self.linear2 = nn.Linear(in_features=1024, out_features=n_dim_latent)

    def forward(self, x):
        x = self.maxPool(self.Relu(self.bn1(self.conv1(x))))
        x = self.maxPool(self.Relu(self.bn2(self.conv2(x))))
        x = self.maxPool(self.Relu(self.bn3(self.conv3(x))))
        x = self.maxPool(self.Relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.Relu(self.linear1(x))
        x = self.linear2(x)
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
