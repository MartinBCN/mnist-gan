import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            nn.Linear(self.nz, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)


if __name__ == '__main__':
    batch = torch.rand(10, 1000)
    gen = Generator(1000)
    print(batch.shape)
    generated_batch = gen(batch)
    print(generated_batch.shape)
