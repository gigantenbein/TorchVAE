import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(3072, 400)
        self.fc21 = nn.Linear(400, 10)
        self.fc22 = nn.Linear(400, 10)
        self.fc3 = nn.Linear(10, 400)
        self.fc4 = nn.Linear(400, 3072)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    # key part of VAE
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 3072))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

"""
conv1
relu
max_pool
conv2
relu
max_pool
conv3
relu
max_pool
conv_transpose_1
relu
conv_transpose_2
relu
conv_transpose_3
sigmoid
"""


class ConvolutionalVAE(nn.Module):
    def __init__(self):
        super(ConvolutionalVAE, self).__init__()

        self.encoder_layers = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2)
        )
        self.mu_fc = nn.Linear(4096, 128)
        self.logvar_fc = nn.Linear(4096, 128)

        self.latent_fc = nn.Linear(128, 8 * 8 * 1024)
        self.decoder_layers = nn.Sequential(
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 3, kernel_size=4, stride=1, padding=2),
        )
        # self.latent_fc = nn.Linear(128, 8 * 8 * 64)
        # self.decoder_layers = nn.Sequential(
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(16, 3, kernel_size=4, stride=1, padding=2),
        # )

    def encode(self, x):
        h1 = self.encoder_layers(x)
        h1 = h1.view(-1, 4096)
        #(mu_, logvar_) = h1.split(1536, 1)
        return self.mu_fc(h1), self.logvar_fc(h1)

    # key part of VAE
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = self.latent_fc(z)
        h3 = self.decoder_layers(h3.view(-1, 1024, 8, 8))
        return torch.sigmoid(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z).view(-1, 3072), mu, logvar