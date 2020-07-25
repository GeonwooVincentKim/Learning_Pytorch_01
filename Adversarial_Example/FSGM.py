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
