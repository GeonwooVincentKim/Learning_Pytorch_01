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
  iteration. Which means is batch_size will executes
  Numbers of Data at once easily.
  
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

"""
  1. Now we prepared Data-Loader, so we are able to
  extract data convenient.
  
  2. Just extract one batch to know how does data 
  looks like.
  
  3. One batch of images and labels will contain 
  16 images and 16 labels desired, 
  for the batch size you set earlier.
"""
dataiter = iter(train_loader)
images, labels = next(dataiter)


"""
  1. You can make one image which collect all images by 
  using function, 'utils.make_grid()' from 'torchvision' module.
  
  2. At the same time, 'img' is a tensor of Pytorch, so 
  we should change to Numpy Matrix that can interchange
  with 'matplotlib' module.
  
  3. Matplotlib recognizes a different order of dimensions, 
  use the np.transpose() function to send the first (0th) dimension 
  to the back.
"""
img = utils.make_grid(images, padding=0)
npimg = img.numpy()
plt.figure(figsize=(10, 7))
plt.imshow(np.transpose(npimg, (1, 2, 0)))
plt.show()

print(labels)

"""
  Set Images Classes as
  1. T-Shirt/top
  2. Trouser
  3. Sweater
  4. Dress
  5. Coat
  6. Sandals
  7. Shirt
  8. Sneakers
  9. Backpacks
  10. Ankle Boots
  
  Those Data Set will be given Label as a Number
  instead of Name. 
"""
CLASSES = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle Boot'
}

"""
  Print Labels English Text.
"""
for label in labels:
    index = label.item()
    print(CLASSES[index])
