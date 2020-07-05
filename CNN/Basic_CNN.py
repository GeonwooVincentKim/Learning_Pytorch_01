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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    # Define Further Data flows in forward() function.
    # Send to GPU Memory to use GPU by settings "cuda".
    # It will proceed if you don't define or set anything.
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        # Dropout the result value which shows as Convolution Result.
        self.drop = nn.Dropout2d()

        # Now the image that passed Convolution Layer and Dropout
        # pass Normal Neural Network.
        # Makes sure reduce each values of Convolution Layer
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)


