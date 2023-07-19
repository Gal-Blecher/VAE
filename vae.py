import torch
import torch.nn as nn
from torchvision.models import resnet18

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        # Encoder (ResNet-18)
        self.encoder = resnet18(pretrained=True)
        self.encoder.fc = nn.Linear(512, latent_dim * 2)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * 256 * 256),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu, logvar = torch.chunk(x, 2, dim=1)
        return mu, logvar

    def decode(self, z):
        output = self.decoder(z)
        output = output.view(-1, 3, 256, 256)
        return output

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar
