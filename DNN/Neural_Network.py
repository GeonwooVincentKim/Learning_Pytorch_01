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
