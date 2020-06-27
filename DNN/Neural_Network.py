import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Set Whole Study Data as 30.
EPOCHS = 30

# Set Mini-Batch-Size as 64.
BATCH_SIZE = 64

"""
  1. Starting for Image classification.
  2. Implement ANN (Artificial Neural Network) for
     Image Classification.
  3. A 3-story ANN (Artificial Neural Network) with 
     3 Layers.
  4. Enter 784 pixel values to output 256 values 
     by multiplying the weights by the matrix 
     and adding bias.
  5. And in the same process, we're going to go through 
     fc2() and fc3() functions, and we're going to output 
     10 values at the end. Each of the 10 output values
     represents Class, and the class with the largest of 
     the 10 will be the predicted value of this model.
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


# Due to the DEVICE Variable, Send it to the GPU when using CUDA,
# or to the CPU if not.
model = Net().to(DEVICE)

# Use optim.SGD algorithm which is the Optimization Algorithm
# in Pytorch embedded modules.
optimizer = optim.SGD(model.paraemeters(), lr=0)


# Training occurs by repeating the tasks of viewing data
# and adjusting the weight of the model.
def train(model, train_loader, optimizer):
    # Change to Training Mode.
    model.train()

    # Now we occurred shape of data as
    # [batch_size, color, height, area].
    for batch_idx, (data, target) in enumerate(train_loader):
        # Send Training Data to the Memory of DEVICE.
        # After then, Stage of Training are similar to
        # the Foregoing Chapter.
        data, target = data.to(DEVICE), target.to(DEVICE)

        # The Gradient are calculate when execute backward() function in Error.
        # Step() Function modifies the weights to match the previously defined algorithm
        # for the calculated slope.
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


# This is the function which named as performance measurement model.
# This function will execute evaluation model while measure model.
def evaluate(model, test_loader):
    # Change to evaluation model.
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
