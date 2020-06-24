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

"""
  1. The meaning of batch_size is the Number of 
  Data processed at once.
  
  2. If batch_size is 16, it reads 16 image per
  iteration. 
  
  If you have enough free memory space on your computer, 
  you can do it bigger, or if you don't have enough, 
  you can do less.
  
  3. Put the loaded DataSet before the Data Loader parameter 
  and specify Batch_size.
"""
batch_size = 16

train_loader = data.DataLoader(
    dataset=trainset,
    batch_size=batch_size
)

test_loader = data.DataLoader(
    dataset=testset,
    batch_size=batch_size
)
