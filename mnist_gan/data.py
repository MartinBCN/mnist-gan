import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

device = torch.device("cuda" if (torch.cuda.is_available() and os.environ.get('USE_GPU')) else "cpu")


def get_data(batch_size: int = 4):
    """

    Parameters
    ----------
    batch_size

    Returns
    -------

    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (2*x) - 1),
        transforms.Lambda(lambda x: x.reshape(28*28))
        ])

    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = DataLoader(dataset1, batch_size=batch_size)
    test_loader = DataLoader(dataset2, batch_size=batch_size)

    return train_loader, test_loader


if __name__ == '__main__':
    train, test = get_data(4)

    for i, (data, target) in enumerate(train):
        print(data.shape)
        print(target.shape)
        for j in range(4):
            print(torch.min(data[j]))
            print(torch.max(data[j]))

        if i == 0:
            break
