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
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


"""
    Use Error-Function which named Mean_Squared_Loss to calculate 
    the values from Decoder and difference in original. 

    2. Named 'criterion' which means standard, and
       Instantiation this Object.
"""
"""
    Make sure you have to write down 
    if __name__ == "__main__":

    if you don't write down this code,
    the 
"""
if __name__ == "__main__":
    autoencoder = Autoencoder().to(DEVICE)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    # Add Random Noise into this function.
    # It gonna be needs for Inputs that go through a Model
    # when its training.
    def add_noise(img):
        noise = torch.randn(img.size()) * 0.2
        noisy_img = img + noise
        return noisy_img


    def train(autoencoder, train_loader):
        autoencoder.train()
        avg_loss = 0
        for step, (x, label) in enumerate(train_loader):
            x = add_noise(x)
            x = x.view(-1, 28 * 28).to(DEVICE)
            y = x.view(-1, 28 * 28).to(DEVICE)

            label = label.to(DEVICE)
            encoded, decoded = autoencoder(x)

            loss = criterion(decoded, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
        return avg_loss / len(train_loader)


    for epoch in range(1, EPOCH + 1):
        loss = train(autoencoder, train_loader)
        print("[Epoch : {}] - loss : {}".format(epoch, loss))
