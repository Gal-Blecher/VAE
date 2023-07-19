import data
import vae
import train

def run(setup_dict):

    train_dataloader = data.get_dataloader(setup_dict['data_dir_path'])

    model = vae.VAE(latent_dim=setup_dict['latent_dim'])

    if setup_dict['test_batch']==True:
        batch = next(iter(train_dataloader))
        mu, logvar = model.encode(batch[0])
        z = model.reparameterize(mu, logvar)
        output = model.decode(z)

    model = train.train_vae(model, train_dataloader, setup_dict['n_epochs'], setup_dict['save_path'], setup_dict)


if __name__ == '__main__':
    setup_dict = {
        'test_batch': True,
        'latent_dim': 128,
        'n_epochs': 50,
        'save_path': './models/vae_1.pkl',
        'data_dir_path': '/home/gal/DS_dataset/train',
        'kl_coeff': 1e-6,
        'lr': 0.001
    }
    run(setup_dict)

