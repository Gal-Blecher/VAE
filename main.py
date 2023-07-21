import data
import vae
import train

def run(setup_dict):

    train_dataloader = data.get_dataloader(setup_dict['data_dir_path'])

    model = vae.VAE(latent_dim=setup_dict['latent_dim'])

    if setup_dict['test_batch']==True:
        batch = next(iter(train_dataloader))[0]
        mu, logvar = model.encode(batch)
        z = model.reparameterize(mu, logvar)
        output = model.decode(z)

    train.train_vae(model, train_dataloader, setup_dict['n_epochs'], setup_dict['save_path'], setup_dict)


if __name__ == '__main__':
    setup_dict = {
        'test_batch': True,
        'latent_dim': 16,
        'n_epochs': 300,
        'save_path': 'models/vae_8.pkl',
        'data_dir_path': '/home/gal/DS_dataset/train',
        # 'data_dir_path': '/Users/galblecher/Desktop/private/inter/imagene/DS_dataset/train',
        'kl_coeff': 1e-5,
        'lr': 0.001
    }
    run(setup_dict)

