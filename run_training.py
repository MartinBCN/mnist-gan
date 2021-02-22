import os

from mnist_gan.data import get_data


# learning parameters
from mnist_gan.gan import GAN

batch_size = 512
train, test = get_data(64)

gan = GAN(latent_dimension=128)

gan.train(test, epochs=2)

model_dir = os.environ.get('MODELS', 'models')
gan.save_generator(f'{model_dir}/test.pth')
