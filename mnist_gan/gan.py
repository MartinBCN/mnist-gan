import os

import numpy as np
import torch
from sklearn import metrics
from torch import optim, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

from mnist_gan.discriminator import Discriminator
from mnist_gan.generator import Generator

device = torch.device("cuda" if (torch.cuda.is_available() and os.environ.get('USE_GPU')) else "cpu")


class MnistGan(object):

    def __init__(self, batch_size: int = 4,
                 latent_dimension: int = 100):
        self.batch_size = batch_size
        self.discriminator = Discriminator()
        self.generator = Generator(latent_dimension=latent_dimension)
        self.criterion = nn.BCELoss()
        self.latent_dimension = latent_dimension
        self.bookkeeping = {'loss': {'discriminator': [], 'generator': []},
                            'accuracy': {'discriminator': [], 'generator': []}}

        self.visualisation_batch = torch.rand(5, self.latent_dimension)

    def fake_batch(self, target: Tensor, optimizer: Optimizer):
        noise = torch.rand(self.batch_size, self.latent_dimension)
        fake_images = self.generator(noise)

        outputs = self.discriminator(fake_images)
        accuracy_fake = metrics.accuracy_score(target, outputs > 0.5)
        loss_fake = self.criterion(outputs, target)
        loss_fake.backward()
        optimizer.step()

        return loss_fake, accuracy_fake

    def visualise(self, epoch: int):

        #
        fig_dir = os.environ.get('FIG_DIR', 'figures')
        images = self.generator(self.visualisation_batch)

        # Rescale images 0 - 1
        images = 0.5 * images + 0.5
        images = images.detach().numpy()

        cols = images.shape[0]
        fig, axs = plt.subplots(1, cols)
        idx = 0
        for i in range(cols):
            axs[i].imshow(images[idx].reshape(28, 28), cmap='gray')
            axs[i].axis('off')
            idx += 1
        fig.savefig(f"{fig_dir}/gan_images_{epoch}.png")
        plt.close()

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(self.bookkeeping['loss']['discriminator'], label='Discriminator Loss')
        axs[0, 0].legend()
        axs[0, 1].plot(self.bookkeeping['accuracy']['discriminator'], label='Discriminator Accuracy')
        axs[0, 1].legend()
        axs[1, 0].plot(self.bookkeeping['loss']['generator'], label='Generator Loss')
        axs[1, 0].legend()
        axs[1, 1].plot(self.bookkeeping['accuracy']['generator'], label='Generator Accuracy')
        axs[1, 1].legend()

        fig.savefig(f"{fig_dir}/losses.png")
        plt.close()

    def train(self, data: DataLoader, epochs: int = 10):
        disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002)
        gen_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002)

        ones = torch.ones(self.batch_size, 1)
        zeros = torch.ones(self.batch_size, 1)
        n_batch = len(data)

        for epoch in range(epochs):

            print(f'=== Epoch {epoch} / {epochs}')

            epoch_loss_discriminator = 0
            epoch_accuracy_discriminator = []
            epoch_loss_generator = 0
            epoch_accuracy_generator = []

            # --- Train Discriminator ---

            self.discriminator.train()
            self.generator.eval()
            disc_optimizer.zero_grad()

            for i, (real_images, _) in enumerate(data):

                if (i % 50) == 0:
                    print(f'Train Discriminator Batch {i}/{n_batch}')

                # --- Real Images ---
                real_images = real_images.to(device)
                outputs = self.discriminator(real_images)
                accuracy_real = metrics.accuracy_score(ones, outputs > 0.5)
                loss_real = self.criterion(outputs, ones)
                loss_real.backward()
                disc_optimizer.step()

                # --- Fake Images ---
                loss_fake, accuracy_fake = self.fake_batch(target=zeros, optimizer=disc_optimizer)
                epoch_loss_discriminator += (loss_fake + loss_real) / 2
                epoch_accuracy_discriminator.append((accuracy_fake + accuracy_real) / 2)

            self.bookkeeping['loss']['discriminator'].append(epoch_loss_discriminator)
            self.bookkeeping['accuracy']['discriminator'].append(np.mean(epoch_accuracy_discriminator))

            # --- Train Generator ---
            self.generator.train()
            self.discriminator.eval()
            gen_optimizer.zero_grad()

            for i in range(2 * n_batch):

                if (i % 50) == 0:
                    print(f'Train Generator Batch {i}/{2 * n_batch}')

                loss, accuracy = self.fake_batch(target=ones, optimizer=gen_optimizer)
                epoch_loss_generator += loss
                epoch_accuracy_generator.append(accuracy)

            self.bookkeeping['loss']['generator'].append(epoch_loss_generator)
            self.bookkeeping['accuracy']['generator'].append(np.mean(epoch_accuracy_generator))

            self.visualise(epoch)
