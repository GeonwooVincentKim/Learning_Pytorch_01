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
