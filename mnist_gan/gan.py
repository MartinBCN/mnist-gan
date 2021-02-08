import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn

from mnist_gan.discriminator import Discriminator
from mnist_gan.generator import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MnistGan(object):

    def __init__(self, batch_size: int = 4,
                 latent_dimension: int = 100):
        self.batch_size = batch_size
        self.discriminator = Discriminator()
        self.generator = Generator(latent_dimension=latent_dimension)
        self.latent_dimension = latent_dimension

    def train(self, data: DataLoader, epochs: int = 10):
        for i in range(epochs):
            ones = np.ones(self.batch_size)
            zeros = np.zeros(self.batch_size)

            # Real Images:

            self.discriminator.train()
            disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002)
            disc_optimizer.zero_grad()

            for (real_images, _) in data:
                real_images = real_images.to(device)
                outputs = self.discriminator(real_images)
                criterion = nn.CrossEntropyLoss()

                loss = criterion(outputs, ones)
                loss.backward()
                disc_optimizer.step()
