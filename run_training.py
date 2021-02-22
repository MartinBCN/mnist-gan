from mnist_gan.data import get_data


# learning parameters
from mnist_gan.gan import GAN

batch_size = 512
train, test = get_data(64)

gan = GAN(latent_dimension=128)

gan.train(train, epochs=20)
