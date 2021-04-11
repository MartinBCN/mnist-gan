import os
from pathlib import Path
from typing import Union, Tuple

from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def get_data(data_dir: Union[Path, str], batch_size: int = 10) -> Tuple[DataLoader, DataLoader]:
    """
    Wrapper for the built-in MNIST dataset in TorchVision

    Parameters
    ----------
    data_dir: Union[Path, str]
        Base data directory, mnist data set will be downloaded here if necessary
    batch_size: int, default = 10
        Batch size should be chosen depending on available memory

    Returns
    -------
    train_loader: DataLoader
        PyTorch DataLoader with MNIST dataset and transforms (Tensor, Normalize)
    test_loader: DataLoader
        PyTorch DataLoader with MNIST dataset and transforms (Tensor, Normalize)
    """
    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
    ])

    train_data = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


if __name__ == "__main__":
    data_directory = os.environ.get('DATA_DIR', Path(__file__).parents[1] / 'data')
    train, test = get_data(data_directory, 10)

    print(len(train), len(test))

    for (batch, labels) in train:
        print(batch.shape)
        print(labels.shape)
        break
