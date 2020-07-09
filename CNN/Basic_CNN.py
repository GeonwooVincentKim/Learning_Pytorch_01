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


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        # Dropout the result value which shows as Convolution Result.
        self.drop = nn.Dropout2d()

        # Now the image that passed Convolution Layer and Dropout
        # pass Normal Neural Network.
        # Makes sure reduce each values of Convolution Layer
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # We defined Training Modules of Model, now we going to make path
    # that the data through Input and Output.
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        # The two-dimensional feature map cannot be inserted directly into the input,
        # so use the view() function to distribute it in one dimension.
        x = x.view(-1, 320)

        # The last layer is a neural network with 10 output values with 0 to 9 Label.
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = CNN().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx & 200 == 0:
            print("Train Epoch : {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}"
                  .format(epoch, batch_idx * len(data),
                          len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item()))


def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            # Sum-up Batch Error.
            test_loss += F.cross_entropy(
                output, target, reduction='sum'
            ).item()

            # The Index which have highest value is Prediction Value.
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuracy = 100. * correct / len(test_loader.dataset)
        return test_loss, test_accuracy


for epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer, epoch)
    test_loss, test_accuracy = evaluate(model, test_loader)

    print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'
          .format(epoch, test_loss, test_accuracy))
