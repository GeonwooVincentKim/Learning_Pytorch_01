"""
    Generative Adversarial Network
    - 1. GAN is a generative model.
    - 2. GAN trains adversarial.
    - 3. GAN is a network model.

    A methodology in which two opposing models
    compete to learn.

    GAN Model consist two key-modules,
    Generator and discriminator.

    -> The GAN Model is a model designed to complement
    security-impaired Maching Learning models
    such as ResNet and CNN.
"""
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from torchvision import transforms, datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt


# Hyper-Parameter
EPOCHS = 500
BATCH_SIZE = 100
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("Use the following devices. : ", DEVICE)


# Prepare DataSets that needs for 'Model Training".
# Fashion MNIST DataSet
trainset = datasets.FashionMNIST(
    './.data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ),)
    ])
)

train_loader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# Implement Generator and Discriminator
# Result values are Equal to images dimensions,
# 784 dimensions which has same dimension
# with Fashion MNIST.


# Generator
G = nn.Sequential(
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 784),
    nn.Tanh()
)

# Discriminator
D = nn.Sequential(
    nn.Linear(784, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1),
    nn.Sigmoid()
)


# Send Weight Models to specified Model.
D = D.to(DEVICE)
G = G.to(DEVICE)

# Error Function of Binary-Crossing Entropy Function and
# the Adam Module which optimize Generator and Discriminator.
criterion = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = optim.Adam(G.parameters(), lr=0.0002)

total_step = len(train_loader)
for epoch in range(EPOCHS):
    for i, (images, _) in enumerate(train_loader):
        images = images.reshape(BATCH_SIZE, -1).to(DEVICE)
