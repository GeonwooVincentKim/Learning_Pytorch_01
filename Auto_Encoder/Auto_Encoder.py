import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

EPOCH = 10
BATCH_SIZE = 64
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("Using Device : ", DEVICE)

# Load Fashion MNIST Dataset
# We need to observe the Images that Auto-Encoder generates,
# so we don't need to use test_loader.
trainset = datasets.FashionMNIST(
    root='./.data/',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# testset = datasets.FashionMNIST(
#     root='./.data/',
#     train=False,
#     download=True,
#     transform=transforms.ToTensor()
# )

train_loader = DataLoader(
    dataset=trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)


# Define and Generates Auto_Encoder Module.
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # It starts with 28 x 28, or 784 dimensions, and
        # gradually decreases. The Last output leaves only
        # three features for visualization in three dimensions.
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3),
        )

        # A Decoder recovers 784 Dimensions by receive
        # Latent Variable.
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

