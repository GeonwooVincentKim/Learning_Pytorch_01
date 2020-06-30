import torch
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

