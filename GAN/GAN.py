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
import numpy as np

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
        transforms.Normalize((0.5,), (0.5,), )
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

        # Generate 'real' and 'fake' Label.
        real_labels = torch.ones(BATCH_SIZE, 1).to(DEVICE)
        fake_labels = torch.zeros(BATCH_SIZE, 1).to(DEVICE)

        """
            Discriminator Recognition for Real-Images.
        """
        # Calculate Error that the Discriminator
        # recognize Real-Image for real.
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        # Generate 'fake' Image by Random-Tensor.
        z = torch.randn(BATCH_SIZE, 64).to(DEVICE)
        fake_images = G(z)

        """
            Discriminator Recognition for Fake-Images.
        """
        # Calculate Error that the Discriminator
        # recognize Fake-Image for fake.
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        """
            Calculation Discriminator Errors.
        """
        # Calculate Discriminator Error by adding error
        # which brings and get the answer from Real-Images and
        # Fake Images.
        d_loss = d_loss_real + d_loss_fake
        outputs = D(fake_images)
        g_loss = criterion(outputs, real_labels)

        # Proceed Generator Model training
        # by importing Back-Propagation Algorithm.
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        """
            Check for Learnign Progress. (Inner Epoch)
        """
        print(
            'Inner_Epoch [{}/{}], '
            'Inner_d_loss: {:.4f}, Inner_g_loss: {:.4f}, '
            'Inner_D(x): {:.2f}, Inner_D(G(z)): {:.2f}'
            .format(
                i, EPOCHS+100,
                d_loss.item(), g_loss.item(),
                real_score.mean().item(),
                fake_score.mean().item()
            )
        )
    """
        Check for Learning Progress. (Outer Epoch)
    """
    print(
        'Epoch [{}/{}], '
        'd_loss: {:.4f}, g_loss: {:.4f}, '
        'D(x): {:.2f}, D(G(z)): {:.2f}'
        .format(
           epoch, EPOCHS,
           d_loss.item(), g_loss.item(),
           real_score.mean().item(),
           fake_score.mean().item()
        )
    )
    print("\n\n\n")

# Visualization Image that Generator made.
z = torch.randn(BATCH_SIZE, 64).to(DEVICE)
fake_images = G(z)

for i in range(10):
    fake_images_img = np.reshape(fake_images.data.cpu().numpy()
                                 [i], (28, 28))
    plt.imshow(fake_images_img, cmap='gray')
    plt.show()
