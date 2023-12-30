import os
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader


dataset = MNIST("./../../Datasets/", download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset)
