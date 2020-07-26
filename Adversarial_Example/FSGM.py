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

"""
    1. squeeze
    -> Returns a tensor with all the dimensions of 
    Input of size '1' removed.
    
    2. unsqueeze
    -> Returns a new tensor with a dimension of size 
    one inserted at the specified position.
"""
img_tensor = img_transforms(img)
img_tensor = img_tensor.unsqueeze(0)

print("Shape of Image Tensor: ", img_tensor.size())

# Convert Numpy Array to visualize Images.
# [1, 3, 244, 244] -> [3, 244, 244]
original_img_view = img_tensor.squeeze(0).detach()
original_img_view = original_img_view.transpose(0, 2).\
    transpose(0, 1).numpy()

# Visualize Tensor.
plt.imshow(original_img_view)

# Adversarial Attack
output = model(img_tensor)
prediction = output.max(1, keepdim=False)[1]

prediction_idx = prediction.item()
prediction_name = idx2class[prediction_idx]

print("Predicted Label Number", prediction_idx)
print("Label Name", prediction_name)


"""
    fgsm_attack()
    - 1. This is the function that Gets the 
    Original Images (or Image) and generate 
    Adversarial_Example.
    
    - 2. FGSM also extract Information of 
    Model Input Images Gradient, and distort it
    and add into Original Images(or Image). 
"""


def fgsm_attack(image, epsilon, gradient):
    # Find the sign value of the element of the Gradient Value.
    sign_gradient = gradient.sign()

    # Adjust each Pixel Value of the Image by Epsilon
    # in the Sign_gradient direction.
    perturbed_image = image + epsilon * sign_gradient

    # Adjust the values outside the range of [0, 1].
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


# Create Adversarial_Example
# Set to get Image Gradient Value.
img_tensor.requires_grad_(True)

# Through Images to the Model.
output = model(img_tensor)

# Get Error values(No.263 is Welsh Corgi).
loss = F.nll_loss(output, torch.tensor([263]))

# Get Gradient Value.
model.zero_grad()
loss.backward()

# Extract Image Gradient.
gradient = img_tensor.grad.data


# Create Adversarial_Example by FGSM Attack.
epsilon = 0.03
perturbed_data = fgsm_attack(img_tensor, epsilon, gradient)

# Through Created Adversarial_Example to the Model.
output = model(perturbed_data)

# Confirm Perturbed_prediction
perturbed_prediction = output.max(1, keepdim=True)[1]
perturbed_prediction_idx = perturbed_prediction.item()
perturbed_prediction_name = idx2class[perturbed_prediction_idx]

print("Predicted Label Number : ", perturbed_prediction_idx)
print("Label Name : ", perturbed_prediction_name)

