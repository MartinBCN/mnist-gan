from mnist_gan.data import get_data
from mnist_gan.gan import MnistGan

# The split is not really useful here except for one thing: the test set can be used as a reduced-size sample
# to execute a full epoch faster

batch_size = 50
train_loader, test_loader = get_data(batch_size=batch_size)

model = MnistGan(batch_size=batch_size)

model.train(train_loader)
