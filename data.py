import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

def calculate_mean_std(image_folder_path):
    # Define the transformation to calculate mean and standard deviation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to a consistent size
        transforms.ToTensor()  # Convert images to tensors
    ])

    # Load the image dataset without labels
    dataset = ImageFolder(root=image_folder_path, transform=transform)

    # Calculate mean and standard deviation
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    mean = 0.
    std = 0.
    total_samples = 0

    for data, _ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples

    return mean, std

def get_dataloader(path_to_images_folder):
    # mean, std = calculate_mean_std(path_to_images_folder)

    transform = transforms.Compose([
        transforms.CenterCrop((128, 128)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.2438, 0.2180, 0.2218), std=(0.4609, 0.4237, 0.3003))
    ])

    # Load the image dataset with the desired transforms
    dataset = ImageFolder(root=path_to_images_folder, transform=transform)

    # Create a data loader for the dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    return dataloader
