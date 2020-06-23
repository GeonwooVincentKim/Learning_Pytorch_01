from torchvision import datasets, transforms, utils
from torch.utils import data

"""
  1. torch.utils.data
  --> It's a module that have tools which is defines
  Dataset standard and get them by cutting and mixing together.
  
  2. torchvision.datasets
  --> Collection of Image Data Set that inherit 
  'torch.utils.data.Dataset'.
  
  3. torchvision.transform
  --> This module contains various conversion filters 
  that can be used for Image Dataset.
  
  4. torchvision.utils
  --> It's a module that have tools for save and
  visualization Image Data.
"""

import matplotlib.pyplot as plt
import numpy as np

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
