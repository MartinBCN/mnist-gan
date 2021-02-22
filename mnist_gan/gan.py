import os
import torch
from sklearn import metrics
from torch import optim, nn, Tensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mnist_gan.discriminator import Discriminator
from mnist_gan.generator import Generator

DEVICE = torch.device("cuda" if (torch.cuda.is_available() and os.environ.get('USE_GPU')) else "cpu")


class GAN:

    def __init__(self, latent_dimension: int = 100):

        self.latent_dimension = latent_dimension
        self.generator = Generator(latent_dimension).to(DEVICE)
        self.discriminator = Discriminator().to(DEVICE)

        # This is fixed, we want to see how this improves
        self.visualisation_noise = self.create_noise(5, latent_dimension)

        # optimizers
        self.optimiser_generator = optim.Adam(self.generator.parameters(), lr=0.0002)
        self.optimiser_discriminator = optim.Adam(self.discriminator.parameters(), lr=0.0002)

        # loss function
        self.criterion = nn.BCELoss()

        self.losses = {'discriminator': [], 'generator': []}
        self.accuracies = {'discriminator': [], 'generator': []}

    @staticmethod
    def create_noise(sample_size, nz):
        """
        Function to create the noise vector

        Parameters
        ----------
        sample_size
        nz

        Returns
        -------

        """
        return torch.randn(sample_size, nz).to(DEVICE)

    @staticmethod
    def label_real(size):
        """
        to create real labels (1s)

        Parameters
        ----------
        size

        Returns
        -------

        """

        data = torch.ones(size, 1)
        return data.to(DEVICE)

    @staticmethod
    def label_fake(size):
        """
        to create fake labels (0s)

        Returns
        -------

        """
        data = torch.zeros(size, 1)
        return data.to(DEVICE)

    def visualise(self, epoch: int):

        #
        fig_dir = os.environ.get('FIG_DIR', 'figures')
        self.generator.eval()
        images = self.generator(self.visualisation_noise)

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

        fig, axs = plt.subplots(2)
        axs[0].plot(self.losses['discriminator'], label='Discriminator')
        axs[0].plot(self.losses['generator'], label='Generator')
        axs[0].legend(title='Loss')
        axs[1].plot(self.accuracies['discriminator'], label='Discriminator')
        axs[1].plot(self.accuracies['generator'], label='Generator')
        axs[1].legend(title='Accuracy')

        fig.savefig(f"{fig_dir}/losses.png")
        plt.close()

    def train_discriminator(self, data_real: Tensor, data_fake: Tensor) -> (float, float):

        batch_size = data_real.size(0)
        real_label = self.label_real(batch_size)
        fake_label = self.label_fake(batch_size)

        self.optimiser_discriminator.zero_grad()
        output_real = self.discriminator(data_real)
        loss_real = self.criterion(output_real, real_label)
        accuracy_real = metrics.accuracy_score(real_label, output_real > 0.5)

        output_fake = self.discriminator(data_fake)
        loss_fake = self.criterion(output_fake, fake_label)
        accuracy_fake = metrics.accuracy_score(fake_label, output_fake > 0.5)

        loss_real.backward()
        loss_fake.backward()
        self.optimiser_discriminator.step()

        return loss_real + loss_fake, (accuracy_real + accuracy_fake) / 2

    # function to train the generator network
    def train_generator(self, data_fake: Tensor):

        b_size = data_fake.size(0)

        real_label = self.label_real(b_size)
        self.optimiser_generator.zero_grad()
        output = self.discriminator(data_fake)

        accuracy = metrics.accuracy_score(real_label, output > 0.5)
        loss = self.criterion(output, real_label)
        loss.backward()
        self.optimiser_generator.step()

        return loss, accuracy

    def train(self, train_loader: DataLoader, epochs: int = 10):

        for epoch in range(epochs):

            #
            self.generator.train()
            self.discriminator.train()

            loss_g = 0.0
            loss_d = 0.0
            accuracy_generator = 0
            accuracy_discriminator = 0

            for bi, data in enumerate(train_loader):
                image, _ = data
                image = image.to(DEVICE)
                b_size = len(image)

                data_fake = self.generator(self.create_noise(b_size, self.latent_dimension)).detach()
                data_real = image
                # train the discriminator network
                loss_batch, acc_batch = self.train_discriminator(data_real, data_fake)
                loss_d += loss_batch
                accuracy_discriminator += acc_batch

                # train the generator network
                data_fake = self.generator(self.create_noise(b_size, self.latent_dimension))
                loss_batch, acc_batch = self.train_generator(data_fake)
                loss_g += loss_batch
                accuracy_generator += acc_batch

            epoch_loss_g = loss_g / len(train_loader)  # total generator loss for the epoch
            epoch_loss_d = loss_d / len(train_loader)  # total discriminator loss for the epoch
            self.losses['generator'].append(epoch_loss_g)
            self.losses['discriminator'].append(epoch_loss_d)

            self.accuracies['generator'].append(accuracy_generator / len(train_loader))
            self.accuracies['discriminator'].append(accuracy_discriminator / len(train_loader))

            print(f"Epoch {epoch} of {epochs}")
            print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")

            self.visualise(epoch)
