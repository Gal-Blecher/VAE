import pandas as pd
import torch
import torchvision.models as models
import vae
import train
import data
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt


def load_model(setup_dict):
    # Create an instance of the model
    model = vae.VAE(setup_dict['latent_dim'])
    state_dict = torch.load(setup_dict['load_path'])
    model.load_state_dict(state_dict)
    model.eval()

    train_dataloader = data.get_dataloader(setup_dict['data_dir_path'])
    batch = next(iter(train_dataloader))[0]
    mu, logvar = model.encode(batch)
    z = model.reparameterize(mu, logvar)
    output = model.decode(z)

    return model


def two_dims_from_z(setup_dict, model):
    dataloader = data.get_dataloader(setup_dict['data_dir_path'])
    df = pd.DataFrame(columns=['x', 'y', 'label'])
    all_z = []  # To store all the z values
    all_labels = []  # To store all the labels

    with torch.no_grad():  # Disable gradient computation
        for images, labels in dataloader:
            mu, logvar = model.encode(images)
            z = model.reparameterize(mu, logvar)

            all_z.append(z)
            all_labels += labels.tolist()

    # Concatenate all z values and labels (if not empty)
    if len(all_labels) > 0:
        all_z = torch.cat(all_z, dim=0)
        all_labels = np.array(all_labels)

        # Reduce dimensionality of z to 2 using t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        reduced_z = tsne.fit_transform(all_z)

        # Create the DataFrame with reduced z values and labels
        df['x'] = reduced_z[:, 0]
        df['y'] = reduced_z[:, 1]
        df['label'] = all_labels

    return df

def plot_scatter_with_labels(df):
    plt.figure(figsize=(8, 6))
    unique_labels = df['label'].unique()
    for label in unique_labels:
        plt.scatter(
            df[df['label'] == label]['x'],
            df[df['label'] == label]['y'],
            label=f'Label {label}',
            alpha=0.7
        )
    plt.title('t-SNE Plot of Latent Space (z)')
    plt.xlabel('Dimension 1 (x)')
    plt.ylabel('Dimension 2 (y)')
    plt.legend()
    plt.grid(True)
    plt.show()






if __name__ == '__main__':
    setup_dict = {
        'test_batch': True,
        'latent_dim': 256,
        'n_epochs': 50,
        'save_path': 'models/vae_1.pkl',
        'data_dir_path': '/Users/galblecher/Desktop/private/inter/imagene/DS_dataset/test',
        'kl_coeff': 1e-5,
        'lr': 0.001,
        'load_path': '/Users/galblecher/Desktop/private/inter/VAE/models/auto_encoder_vae_like.pkl'
    }
    model = load_model(setup_dict)
    low_dim_data = two_dims_from_z(setup_dict, model)
    plot_scatter_with_labels(low_dim_data)
    t=1
