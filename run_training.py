from mnist_gan.data import get_data
from mnist_gan.gan import MnistGan

# The split is not really useful here except for one thing: the test set can be used as a reduced-size sample
# to execute a full epoch faster

train_loader, test_loader = get_data()

model = MnistGan()

model.train(test_loader)
