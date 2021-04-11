import torch
import torch.nn as nn
from torch import Tensor


class Generator(nn.Module):
    """
    Generator part of the GAN
    Takes a random tensor from the latent dimension and creates a MNIST-like tensor.
    Architecture is fixed as follows:
    * Three hidden layers [256, 512, 1024]
    * Hidden-layer activation LeakyReLU
    * Final activation TanH to mimic normalized BW images

    Parameters
    ----------
    latent_dimension: int
        Size of the latent dimension
    """

    def __init__(self, latent_dimension: int):
        super(Generator, self).__init__()
        self.embedded_dimension = latent_dimension
        self.main = nn.Sequential(

            nn.Linear(self.embedded_dimension, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),

            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward step via Sequential

        Parameters
        ----------
        x: Tensor
            Input from embedded dimension, shape [batch_size, embedded_dimension]

        Returns
        -------
        Tensor
            Fake MNIST batch, shape [batch_size, 1, 28, 28]
        """
        return self.main(x).view(-1, 1, 28, 28)


if __name__ == '__main__':
    batch = torch.rand(10, 1000)
    gen = Generator(1000)
    print(batch.shape)
    generated_batch = gen(batch)
    print(generated_batch.shape)
    print(generated_batch)
