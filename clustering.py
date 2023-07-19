import torch
import torchvision.models as models
import vae
import train
import data


def load_model(setup_dict):
    # Create an instance of the model
    model = vae.VAE(setup_dict['latent_dim'])
    state_dict = torch.load(setup_dict['load_path'])

    # Load the state_dict into the model
    model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    model.eval()

    train_dataloader = data.get_dataloader(setup_dict['data_dir_path'])
    batch = next(iter(train_dataloader))[0]
    mu, logvar = model.encode(batch)
    z = model.reparameterize(mu, logvar)
    output = model.decode(z)

    return model


if __name__ == '__main__':
    setup_dict = {
        'test_batch': True,
        'latent_dim': 128,
        'n_epochs': 50,
        'save_path': 'models/vae_2.pkl',
        'data_dir_path': '/Users/galblecher/Desktop/private/inter/imagene/DS_dataset',
        'kl_coeff': 1e-6,
        'lr': 0.001,
        'load_path': 'models/vae_1.pkl'
    }
    model = load_model(setup_dict)
