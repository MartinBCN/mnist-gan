import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, image_size: int = 784):
        super(Discriminator, self).__init__()

        self.dense1 = nn.Linear(image_size, 512)
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, 1)

        """
          i = Input(shape=(img_size,))
          x = Dense(512, activation=LeakyReLU(alpha=0.2))(i)
          x = Dense(256, activation=LeakyReLU(alpha=0.2))(x)
          x = Dense(1, activation='sigmoid')(x)
        """

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = torch.sigmoid(self.dense3(x))
        return x


if __name__ == '__main__':
    test_batch = torch.rand(4, 784)
    discriminator = Discriminator()
    print(test_batch.shape)
    result = discriminator(test_batch)
    print(result)
    print(result.shape)

    # torch.Size([4])