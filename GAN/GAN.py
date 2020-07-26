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

from torchvision import transforms, datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt


# Hyper-Parameter
EPOCHS = 500
BATCH_SIZE = 100
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("Use the following devices. : ", DEVICE)


