from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Generator(nn.Module):
    def __init__(self, latent_dimension: int = 100, output_dim: int = 784):
        super(Generator, self).__init__()

        self.dense1 = nn.Linear(latent_dimension, 256)
        self.dense2 = nn.Linear(256, 512)
        self.dense3 = nn.Linear(512, 1024)
        self.dense4 = nn.Linear(1024, output_dim)

        """
          i = Input(shape=(latent_dim,))
          x = Dense(256, activation=LeakyReLU(alpha=0.2))(i)
          x = BatchNormalization(momentum=0.7)(x)
          x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
          x = BatchNormalization(momentum=0.7)(x)
          x = Dense(1024, activation=LeakyReLU(alpha=0.2))(x)
          x = BatchNormalization(momentum=0.7)(x)
          x = Dense(D, activation='tanh')(x)

          model = Model(i, x)
        """

    def forward(self, x):
        x = F.leaky_relu(self.dense1(x), negative_slope=0.2)
        x = F.leaky_relu(self.dense2(x), negative_slope=0.2)
        x = F.leaky_relu(self.dense3(x), negative_slope=0.2)
        x = torch.tanh(self.dense4(x))
        return x


if __name__ == '__main__':
    noise = torch.rand(100)
    generator = Generator(100)
    print(generator(noise).shape)
