import torch
import torch.nn as nn
from torch import Tensor


class Discriminator(nn.Module):
    """
    Discriminator part of the GAN

    The architecture is fixed as follows:
    * Three hidden layers of sizes [1024, 512, 256]
    * LeakyReLU as hidden-layer activation
    * Regularization through dropout layer
    * Final activation by sigmoid for binary discrimination

    Class Attributes
    ----------------
    n_input: int
        Flattened MNIST image -> 28x28 = 784
    """

    n_input = 784

    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(

            nn.Linear(self.n_input, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Define the forward step for the discriminator

        Parameters
        ----------
        x: Tensor
            Input of shape [batch_size, 1, 28, 28]

        Returns
        -------
        Tensor
            Output of shape [batch_size, 1] with discrete values 0/1
            1: Real Image
            0: Fake Image
        """
        x = x.view(-1, self.n_input)
        return self.main(x)


if __name__ == '__main__':
    batch = torch.rand(10, 1, 28, 28)
    disc = Discriminator()
    print(batch.shape)
    generated_batch = disc(batch)
    print(generated_batch.shape)
