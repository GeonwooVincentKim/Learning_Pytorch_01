import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as torch_dt
from torchvision import transforms, datasets

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

EPOCHS = 40
BATCH_SIZE = 64

train_loader = torch_dt.DataLoader(
    datasets.FashionMNIST(
        './.data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
    batch_size=BATCH_SIZE, shuffle=True),
test_loader = torch_dt.DataLoader(
    datasets.FashionMNIST(
        './.data',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3801,))
        ])
    ),
    batch_size=BATCH_SIZE, shuffle=True
)
"""
  CNN Basic
"""
