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

machine_location = "models/machine.h5"
model = MNISTNet()
if not os.path.isfile(machine_location):
    model = train.train(model, train_digits_data)
    torch.save(model, machine_location)

model = torch.load(machine_location)
test.test(model, test_digits_data)