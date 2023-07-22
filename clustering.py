import pandas as pd
import torch
import torchvision.models as models
import vae
import train
import data
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


def load_model(setup_dict):
    model = vae.VAE(setup_dict['latent_dim'])
    state_dict = torch.load(setup_dict['load_path'])
    model.load_state_dict(state_dict)
    model.eval()

    train_dataloader = data.get_dataloader(setup_dict['data_dir_path'])
    batch = next(iter(train_dataloader))[0]
    mu, logvar = model.encode(batch)
    z = model.reparameterize(mu, logvar)
    output = model.decode(z)
    train.plot_images(batch, output)

    return model


def two_dims_from_z(setup_dict, model):
    dataloader = data.get_dataloader(setup_dict['data_dir_path'])
    df = pd.DataFrame(columns=['x', 'y', 'label'])
    all_z = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            mu, logvar = model.encode(images)
            z = model.reparameterize(mu, logvar)

            all_z.append(z)
            all_labels += labels.tolist()

    if len(all_labels) > 0:
        all_z = torch.cat(all_z, dim=0)
        all_labels = np.array(all_labels)

        tsne = TSNE(n_components=2, random_state=42, perplexity=50, n_iter=10_000)
        reduced_z = tsne.fit_transform(all_z)

        df['x'] = reduced_z[:, 0]
        df['y'] = reduced_z[:, 1]
        df['label'] = all_labels

    return df, all_z, reduced_z

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


def cluster(z):
    data = z.cpu().numpy()

    n_clusters = 5
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(data)
    cluster_labels = gmm.predict(data)

    return cluster_labels

if __name__ == '__main__':
    setup_dict = {
        'test_batch': True,
        'latent_dim': 256,
        'n_epochs': 50,
        'save_path': 'models/vae_1.pkl',
        'data_dir_path': '/Users/galblecher/Desktop/private/inter/imagene/DS_dataset/test',
        'kl_coeff': 1e-5,
        'lr': 0.001,
        'load_path': '/Users/galblecher/Desktop/private/inter/VAE/models/vae_10_256.pkl'
    }
    model = load_model(setup_dict)
    df, all_z, reduced_z = two_dims_from_z(setup_dict, model)
    plot_scatter_with_labels(df)
    preds = cluster(torch.tensor(reduced_z))
    df['label'] = preds
    plot_scatter_with_labels(df)
