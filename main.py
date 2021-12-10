import train
import test
from torchvision.datasets import EMNIST
from torchvision.transforms import ToTensor
from mnist_net import MNISTNet
import torch
import os

train_digits_data = EMNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True,
    split="digits"
)

test_digits_data = EMNIST(
    root='data',
    train=False,
    transform=ToTensor(),
    download=True,
    split="digits"
)

machine_location = models/machine.h5
model = MNISTNet()
if os.path.isfile(machine_location):
    model = torch.load("models/cnn_trans.h5")
else:
    model = train.train(model, train_digits_data)
torch.save(model, 'models/machine.h5')
test.test(model, test_digits_data)