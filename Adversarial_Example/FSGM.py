import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
import json
import matplotlib.pyplot as plt


"""
    Now we are going to make model that predicts
    Panda images as Gibbon.
"""
model = models.resnet101(pretrained=True)
model.eval()

CLASSES = json.load(open('./imagenet_samples/imagenet_classes.json'))
idx2class = [CLASSES[str(i)] for i in range(1000)]

# Import Image
img = Image.open("imagenet_samples/corgie.jpg")

# Convert Image as Tensor.
"""
    BICUBIC
    -1. BICUBIC interpolation is an extension of cubic
    interpolation for interpolating data points on a 
    two-dimensional regular grid.
    
    -2. BICUBIC is also well used while Developer or User 
    proceed Image Scaling.
    
    -3. Meaning of Image Scaling are refers to the resizing
    of a digital image.
    
    -4. Whatever, BICUBIC Interpolation is a 2D system of using
    cubic splines or other polynomial technique and enlarging
    digital Images.
"""
img_transforms = transforms.Compose([
    transforms.Resize((224, 224), Image.BICUBIC),
    transforms.ToTensor(),
])

img_tensor = img_transforms(img)
img_tensor = img_tensor.unsqueeze(0)

print("Shape of Image Tensor: ", img_tensor.size())
