import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau


def plot_images(original_images, reconstructed_images):
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 5))
    plt.subplots_adjust(wspace=0.1)

    for i in range(5):
        # Plot original images
        axes[0, i].imshow(np.transpose(original_images[i].detach().numpy(), (1, 2, 0)))
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')

        # Plot reconstructed images
        axes[1, i].imshow(np.transpose(reconstructed_images[i].detach().numpy(), (1, 2, 0)))
        axes[1, i].axis('off')
        axes[1, i].set_title('Reconstructed')

    plt.tight_layout()
    plt.show()

def kl_loss(mu, logvar):
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_loss


def train_vae(vae, train_loader, num_epochs, save_path, setup_dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae.to(device)

    recon_criterion = nn.MSELoss()
    optimizer = optim.Adam(vae.parameters(), lr=setup_dict['lr'])
    scheduler = ReduceLROnPlateau(optimizer, patience=30, factor=0.1, verbose=True)


    min_loss = float('inf')
    best_model = None

    for epoch in range(num_epochs):
        running_recon_loss = 0.0
        running_kl_loss = 0.0

        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()

            reconstructed_images, mu, logvar = vae(images)
            recon_loss = recon_criterion(reconstructed_images, images)
            kl = kl_loss(mu, logvar)

            loss = recon_loss + setup_dict['kl_coeff'] * kl

            loss.backward()
            optimizer.step()

            running_recon_loss += recon_loss.item()
            running_kl_loss += kl.item()

        epoch_recon_loss = running_recon_loss / len(train_loader)
        epoch_kl_loss = running_kl_loss / len(train_loader)
        total_loss = epoch_recon_loss + epoch_kl_loss

        print(
            f"Epoch [{epoch + 1}/{num_epochs}]: Recon Loss: {epoch_recon_loss}, KL Loss: {epoch_kl_loss}, Total Loss: {total_loss}")
        scheduler.step(total_loss)

        if total_loss < min_loss:
            vae = vae.to('cpu')
            min_loss = total_loss
            best_model = vae.state_dict()
            torch.save(vae.state_dict(), save_path)
            vae = vae.to(device)

    vae.load_state_dict(best_model)
