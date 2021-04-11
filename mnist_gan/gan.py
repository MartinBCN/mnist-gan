import os
from pathlib import Path
from typing import Union, Tuple

import torch
from sklearn import metrics
from torch import optim, nn, Tensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mnist_gan.discriminator import Discriminator
from mnist_gan.generator import Generator
plt.style.use('ggplot')

DEVICE = torch.device("cuda" if (torch.cuda.is_available() and os.environ.get('USE_GPU')) else "cpu")


class GAN:
    """
    GAN

    Define and train both the Generator and Discriminator networks simultaneously

    Parameters
    ----------
    latent_dimension: int, default = 100
        Size of the latent dimension
    learning_rate: float, default = 0.0002
        Learning rate, for simplicity we use the same LR for both optimizer. A more elaborate example than MNIST
        in all likelihood requires a more sophisticated choice here
    """
    def __init__(self, latent_dimension: int = 100, learning_rate: float = 0.0002):

        self.latent_dimension = latent_dimension
        self.generator = Generator(latent_dimension).to(DEVICE)
        self.discriminator = Discriminator().to(DEVICE)

        # This is fixed, we want to see how this improves
        self.visualisation_noise = self.create_noise(5)

        # We use Adam with a given learning rate in both cases
        self.optimiser_generator = optim.Adam(self.generator.parameters(), lr=learning_rate)
        self.optimiser_discriminator = optim.Adam(self.discriminator.parameters(), lr=learning_rate)

        # Binary Cross Entropy as Loss Function
        self.criterion = nn.BCELoss()

        # Track losses and accuracies
        self.losses = {'discriminator': [], 'generator': []}
        self.accuracies = {'discriminator': [], 'generator': []}

    def create_noise(self, sample_size: int) -> Tensor:
        """
        Function to create the noise vector

        Parameters
        ----------
        sample_size: int
            Number of fake images we want to create

        Returns
        -------
        Tensor
            Noise vector from embedded dimension, shape [sample_size, embedded_dimension]
        """
        return torch.randn(sample_size, self.latent_dimension).to(DEVICE)

    @staticmethod
    def label_real(batch_size: int) -> Tensor:
        """
        Helper function to create real labels (ones)

        Parameters
        ----------
        batch_size: int

        Returns
        -------
        Tensor
            Fixed labels for the case of real images -> all ones
            Shape [batch_size, 1]
        """

        data = torch.ones(batch_size, 1)
        return data.to(DEVICE)

    @staticmethod
    def label_fake(batch_size: int) -> Tensor:
        """
        Helper function to create fake labels (zeros)

        Parameters
        ----------
        batch_size: int

        Returns
        -------
        Tensor
            Fixed labels for the case of fake images -> all zeros
            Shape [batch_size, 1]
        """
        data = torch.zeros(batch_size, 1)
        return data.to(DEVICE)

    def visualise(self, epoch: int) -> None:
        """
        Create two plots:
        1) Sample of what the generator creates from the fixed noise sample. In time this should look more and more
            like the familiar MNIST numbers
        2) Loss and accuracies vs. epoch. Note that this will not like like a regular training because ideally
            both the discriminator and the generator become better at what they do

        Parameters
        ----------
        epoch: int

        Returns
        -------
        None
        """

        fig_dir = os.environ.get('FIG_DIR', 'figures')

        # Create sample images from fixed noise batch
        with torch.no_grad():
            self.generator.eval()
            images = self.generator(self.visualisation_noise)

        # Rescale images 0 - 1
        images = 0.5 * images + 0.5
        images = images.detach().cpu().numpy()

        cols = images.shape[0]
        fig, axs = plt.subplots(1, cols)
        for i in range(cols):
            axs[i].imshow(images[i].reshape(28, 28), cmap='gray')
            axs[i].axis('off')
        fig.savefig(f"{fig_dir}/gan_images_{epoch}.png")
        plt.close()

        # --- Loss/Accuracy ---
        fig, axs = plt.subplots(2, figsize=(12, 8))
        axs[0].plot(self.losses['discriminator'], label='Discriminator')
        axs[0].plot(self.losses['generator'], label='Generator')
        axs[0].legend(title='Loss')
        axs[1].plot(self.accuracies['discriminator'], label='Discriminator')
        axs[1].plot(self.accuracies['generator'], label='Generator')
        axs[1].legend(title='Accuracy')

        fig.savefig(f"{fig_dir}/losses.png")
        plt.close()

    def train_discriminator(self, data_real: Tensor, data_fake: Tensor) -> Tuple[float, float]:
        """
        Training the Discriminator. Here we feed both a batch of real and a batch of fake images with fixed targets
        (ones for real, zeros for fake, respectively). The loss is calculated as binary cross entropy in both cases.

        Parameters
        ----------
        data_real: Tensor
            Real images, shape [batch_size, 1, 28, 28]
        data_fake
            Fake images, shape [batch_size, 1, 28, 28]

        Returns
        -------
        loss: float
            Sum of losses for fake and real image detection
        accuracy: float
            Mean of accuracy for real/fake image detection
        """

        # Create one set of fake and one set of real labels
        batch_size = data_real.shape[0]
        real_label = self.label_real(batch_size)
        fake_label = self.label_fake(batch_size)

        # Training Step Discriminator
        self.optimiser_discriminator.zero_grad()

        # 1) Detect real images
        output_real = self.discriminator(data_real)
        loss_real = self.criterion(output_real, real_label)
        accuracy_real = metrics.accuracy_score(real_label, output_real > 0.5)
        loss_real.backward()

        # 2) Detect fake images
        output_fake = self.discriminator(data_fake)
        loss_fake = self.criterion(output_fake, fake_label)
        accuracy_fake = metrics.accuracy_score(fake_label, output_fake > 0.5)
        loss_fake.backward()

        self.optimiser_discriminator.step()

        # Book-keeping
        loss = loss_real.detach().cpu() + loss_fake.detach().cpu()
        accuracy = (accuracy_real + accuracy_fake) / 2
        return loss, accuracy

    def train_generator(self, data_fake: Tensor) -> Tuple[float, float]:
        """
        Function to train the Generator part of the GAN

        Parameters
        ----------
        data_fake: Tensor
            Fake image data, shape [batch_size, 1, 28, 28]

        Returns
        -------
        loss: float
        accuracy: float
        """

        # We use FAKE data and REAL as label as we want the generator to produce fake images that appear real
        b_size = data_fake.shape[0]
        real_label = self.label_real(b_size)

        # Training step for Generator
        self.optimiser_generator.zero_grad()
        output = self.discriminator(data_fake)
        loss = self.criterion(output, real_label)
        loss.backward()
        self.optimiser_generator.step()

        # Book-keeping
        loss = loss.detach().cpu()
        accuracy = metrics.accuracy_score(real_label, output > 0.5)

        return loss, accuracy

    def train(self, train_loader: DataLoader, epochs: int = 10) -> None:
        """
        Training function for GAN

        Notice that contrary to regular NN training we do not define an early exit strategy here. Since both adversary
        networks are supposed to keep improving there is no obvious convergence in the classical sense.

        Parameters
        ----------
        train_loader: DataLoader
            PyTorch DataLoader with training data
        epochs: int
            Number of epochs

        Returns
        -------
        None
        """

        for epoch in range(epochs):

            # Visualisation at the end of the epoch is done in eval -> back to train()
            self.generator.train()
            self.discriminator.train()

            loss_g = 0.0
            loss_d = 0.0
            accuracy_generator = 0
            accuracy_discriminator = 0

            for data in train_loader:

                # Data batches
                image, _ = data
                image = image.to(DEVICE)
                b_size = len(image)

                data_fake = self.generator(self.create_noise(b_size)).detach()
                data_real = image

                # train the discriminator network
                loss_batch, acc_batch = self.train_discriminator(data_real, data_fake)
                loss_d += loss_batch
                accuracy_discriminator += acc_batch

                # train the generator network
                data_fake = self.generator(self.create_noise(b_size))
                loss_batch, acc_batch = self.train_generator(data_fake)
                loss_g += loss_batch
                accuracy_generator += acc_batch

            # --- Book-keeping ---
            epoch_loss_g = loss_g / len(train_loader)  # total generator loss for the epoch
            epoch_loss_d = loss_d / len(train_loader)  # total discriminator loss for the epoch
            self.losses['generator'].append(epoch_loss_g)
            self.losses['discriminator'].append(epoch_loss_d)

            self.accuracies['generator'].append(accuracy_generator / len(train_loader))
            self.accuracies['discriminator'].append(accuracy_discriminator / len(train_loader))

            print(f"Epoch {epoch + 1} of {epochs}")
            print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")

            # Visualise the state after each epoch to track the progress
            self.visualise(epoch)

    def save_generator(self, fn: Union[str, Path]) -> None:
        """
        Save the Generator for future purposes

        Parameters
        ----------
        fn: Union[str, Path]

        Returns
        -------
        None
        """
        fn = Path(fn)
        fn.parents[0].mkdir(parents=True, exist_ok=True)
        torch.save(self.generator.state_dict(), fn)
