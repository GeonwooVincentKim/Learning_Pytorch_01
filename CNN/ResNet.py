"""
ResNet
1. ResNet are use Realistic Color Data.
2. This Model apply CNN Model.
3. ResNet is the model that won the ImageNet competition
   in 2015 where 10 million images were learned to compete
   for recognition with 150,000 images.
"""
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as torch_dt
from torchvision import transforms, datasets


EPOCHS = 300
BATCH_SIZE = 128

"""
  Now we are going to use "datasets.CIFAR10" instead of "datatsets.FashionMNIST".
  To prevent Overfitting, add a noise into Learning DataSet such as RandomCrop and RandomHorizontalFlip.
"""
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./.DATA',
                     train=True,
                     download=True,
                     transform=transforms.Compose([
                         transforms.RandomCrop(32, padding=4),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5),
                                              (0.5, 0.5, 0.5))
                     ])),
    batch_size=BATCH_SIZE, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./.DATA',
                     train=False,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5),
                                              (0.5, 0.5, 0.5))
                     ])),
    batch_size=BATCH_SIZE, shuffle=True
)

"""
    How to store CNN deeply
    
    The overlapping of multiple Neural Networks does not infinitely 
    improve the learning performance.
    Because the information of Initial Input Images are disappear
    through several levels of Neural Network.
    
    The key to ResNet is that it divides the network into smaller blocks, 
    Residual blocks.
    
    By adding x, which was input to the output of the residual block, 
    the model can be designed much deeper.
    
    Train model by learn Input and Output separately are
    better than Train Input and Output together.
"""


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    """
        Input X comes in and goes through the Convolution, Batch normalization, 
        and activation functions, and then again passes the input X through self.shortcut 
        to make it the same size and add it to the value passed through the activation function.
    """
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


"""
    Defined ResNet.
    The model receives the image, then convolution 
    and Batch normalization, then passes through 
    several layers of BasicBlock and outputs forecasts 
    through mean pulling and neural networks.
"""


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.linear = nn.Linear(64, num_classes)
