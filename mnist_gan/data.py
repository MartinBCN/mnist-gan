import os
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def get_data(batch_size: int = 10):
    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,),(0.5,)),
    ])
    # to_pil_image = transforms.ToPILImage()

    data_dir = os.environ.get('DATADIR', Path(__file__).parents[1] / 'data')

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
    train, test = get_data(10)

    print(len(train), len(test))

    for (batch, labels) in train:
        print(batch.shape)
        print(labels.shape)
        break
