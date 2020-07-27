import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from torchvision import transforms, datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

EPOCHS = 500
BATCH_SIZE = 100
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("Use the following devices : ", DEVICE)

trainset = datasets.FashionMNIST(
    './.data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ),)
    ]),
)

train_loader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 10)

        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.embed(labels)
        x = torch.cat([z, c], 1)
        return self.model(x)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        c = self.embed(labels)
        x = torch.cat([x, c], 1)
        return self.model(x)


# Create Model Instance and
# Send Weight Models to specified Model.
D = Discriminator().to(DEVICE)
G = Generator().to(DEVICE)

# Error Function of Binary-Crossing Entropy Function and
# the Adam Module which optimize Generator and Discriminator.
criterion = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = optim.Adam(G.parameters(), lr=0.0002)

total_step = len(train_loader)
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(BATCH_SIZE, -1).to(DEVICE)

        # Generate 'Real' Labels and 'Fake' Labels.
        real_labels = torch.ones(BATCH_SIZE, 1).to(DEVICE)
        fake_labels = torch.ones(BATCH_SIZE, 1).to(DEVICE)

        # Calculate Error that the Discriminator
        # recognize Real-Image for real.
        # (Input DataSet Label)
        labels = labels.to(DEVICE)
        outputs = D(images, labels)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        # Generate 'Fake-Image' by input Random-Tensor and
        # Random-Label into 'Generator'.
        z = torch.randn(BATCH_SIZE, 100).to(DEVICE)
        g_label = torch.randint(0, 10, (BATCH_SIZE,)).to(DEVICE)
        fake_images = G(z, g_label)

        """
            Discriminator Recognition for Real-Images.
        """
        # Calculate Error that the Discriminator
        # recognize Real-Image for real.
        outputs = D(fake_images, g_label)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        """
            Calculation Discriminator Errors.
        """
        # Calculate Discriminator Error by adding error
        # which brings and get the answer from Real-Images and
        # Fake Images.
        d_loss = d_loss_real + d_loss_fake

        # Process training procedure for Discriminator Errors
        # by importing Back-Propagation Algorithm.
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        """
            Generator Recognition for Real-Images.
        """
        # Calculate the error as to
        # whether the generator has cheated
        # the discriminator. (random label input)
        fake_images = G(z, g_label)
        outputs = D(fake_images, g_label)
        g_loss = criterion(outputs, real_labels)

        # Proceed training procedure for Generator Model
        # by importing Back-Propagation Algorithm.
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    print("Epoch [{} / {}] "
          "d_loss: {:.4f} - g_loss: {:.4f} "
          "D(x): {:.2f} - D(G(z)): {:.2f}"
          .format(epoch, EPOCHS,
                  d_loss.item(), g_loss.item(),
                  real_score.mean().item(),
                  fake_score.mean().item()))
