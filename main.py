import train
import sal
from torchvision.datasets import EMNIST
from torchvision.transforms import ToTensor
from mnist_net import MNISTNet
import torch
import os
from torchvision import datasets, transforms

transform = transforms.ToTensor()
train_digits_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

test_digits_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
machine_location = "models/machine.h5"
model = MNISTNet()
if not os.path.isfile(machine_location):
    model = train.train(model, train_digits_data)
    torch.save(model, machine_location)

model = torch.load(machine_location)
sal.test(model, test_digits_data)