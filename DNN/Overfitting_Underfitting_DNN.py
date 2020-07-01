import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, datasets

BATCH_SIZE = 64
"""
  OverFitting (Example are also included).
  1. Creating a Machine Learning Model can provide 
     good study efficiency, but sometimes it doesn't 
     perform well in test sets or in real life.
  2. For example, your school senior let you know that
     Example of test will be same, but the Professor or
     your school teacher realized that every students 
     average score have improved. so it have been changed,
     and then you couldn't clear an examination paper. 
     because you only just studied Test Data which forms
     are same, which mean is the Data-Set and Real-life Situation
     doesn't match, so you couldn't clear it.

  UnderFitting
  1. It is the opposite way of OverFitting.
  2. For example, you didn't studied to much, so your Score of Test 
     is bad. That is an example of UnderFitting.
     
  BATCH_SIZE 
  - Using a batch size of 64 (orange) achieves a test accuracy of 98% 
  while using a batch size of 1024 only achieves about 96%. 
  But by increasing the learning rate, using a batch size of 1024 
  also achieves test accuracy of 98%. 
  (
    From
    https://medium.com/mini-distill/effect-of-batch-size-on-training-dynamics-21c14f7a716e#:~:text=Using%20a%20batch%20size%20of,achieves%20test%20accuracy%20of%2098%25.
  ) 
"""
# Add Torch_Vision Data-Set in Data-Loader.
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./.data',
                   train=True,
                   download=True,
                   transform=transforms.Compose([
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True
)

"""
  Dropout
  1. It solves Model OverFitting Problem.
  2. Dropout does not use a part of Neural Network
     while Learning Process.
  3. For example, when Dropout is just only use 50%,
     It just use 50% of Neural Network in every stage of Learning Process.
  4. Use entire of Neurons when the Model are in Qualification Process.
  5. Disperse to Other Neurons not to excluded Neurons, and then 
     prevent Static Phenomenon each of Neurons. 
"""


class Net(nn.Module):
    def __init__(self, droptout_p=0.2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        # Probability of Dropout
        self.dropout_p = droptout_p

    # Dropout once again after
    # Passing through layer1 and once again
    # after passing through layer2.
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        # Add Dropout
        x = F.dropout(
            x, training=self.training,
            p=self.dropout_p
        )
        x = F.relu(self.fc2(x))
        # Add Dropout
        x = F.dropout(
            x, training=self.training,
            p=self.dropout_p
        )
        x = self.fc3(x)
        return x
