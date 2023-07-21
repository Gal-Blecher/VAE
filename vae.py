import torch
import torch.nn as nn
from torchvision.models import resnet18

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        # Encoder (ResNet-18)
        self.encoder = resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])

        # Decoder
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)
        self.tanh = nn.Tanh()

    def decoder(self, z):
        z = self.fc(z)
        z = z.view(z.size(0), 256, 8, 8)  # Reshape to (batch_size, 256, 8, 8)
        z = self.deconv1(z)
        z = self.bn1(z)
        z = torch.relu(z)
        z = self.deconv2(z)
        z = self.bn2(z)
        z = torch.relu(z)
        z = self.deconv3(z)
        # z = self.tanh(z)
        return z

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu, logvar = torch.chunk(x, 2, dim=1)
        return mu, logvar

    def decode(self, z):
        output = self.decoder(z)
        return output

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std) * 0
        z = mu + epsilon * std
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar
