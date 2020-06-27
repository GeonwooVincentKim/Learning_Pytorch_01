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
  1. Convert Image to Tensor.
  2. The version of transform is a tool that 
  convert inputs.
"""
transform = transforms.Compose([
    transforms.ToTensor()
])

"""
  1. The Fashion MNIST Dataset is divided 
  into a training set for training and 
  a test set for performance evaluation.

  2. Check Data-sets which you define folder location
  as 'root', if Data-sets not existing, 
  it will save New Data-sets file automatically.
"""
trainset = datasets.FashionMNIST(
    root='./.data/',
    train=True,
    download=True,
    transform=transform
)

testset = datasets.FashionMNIST(
    root='./.data/',
    train=False,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=BATCH_SIZE
)

test_loader = torch.utils.data.DataLoader(
    dataset=testset,
    batch_size=BATCH_SIZE
)

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
# Make sure you have to set Learning Rate as 0.01.
# If you set Learning Rate as 0, The Model will not training by itself.
optimizer = optim.SGD(model.parameters(), lr=0.01)


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

    # Specify torch.no_grad as like this syntax.
    # Make sure you have to send as DEVICE, and then
    # Get output model predictive value.
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            # Add all of Error.
            # set 'reduction='sum'' through cross entropy for evaluate,
            # and get Mini_Batch's 'sum' instead of Mini_Batch's 'average'.
            test_loss += F.cross_entropy(output, target,
                                         reduction='sum').item()

            # The Class which have most big value is prediction of Model.
            # Compare prediction and Answer and then add 1 when both of them are match.cc
            # 'eq()' function outputs 1 when the Model Prediction Fashion Item and Label are match,
            # otherwise, it outputs 0 when it is not Match.
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Now we got Error of Entire DataSet and Sum of all values of Numbers.
    # Now we are going to get accuracy by multiply 100 to entire Answer Average.
    # Average = Values which are contains Entire of Model DataSet and All values of Numbers.
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


# Repeat the verification using Epoch, learning, and DataSet and print the results.
for epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer)
    test_loss, test_accuracy = evaluate(model, test_loader)

    print('[{}] Test Loss : {:.4f}, Accuracy: {:.2f}%'.format(epoch, test_loss, test_accuracy))
