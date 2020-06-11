import torch
import pickle
import matplotlib.pyplot as plt

# Loading and visualization of contaminated images
# (images to be restored).
broken_image = torch.FloatTensor(
    pickle.load(
        open('./broken_image_t.p', 'rb'),
        encoding='latin1')
)
plt.imshow(broken_image.view(100, 100))


# This is a function that taint images.
# But we will going to recover images by using ML Platform.
def weird_function(x, n_iter=5):
    h = x
    filt = torch.tensor([-1. / 3, 1. / 3, -1. / 3])
    for i in range(n_iter):
        zero_tensor = torch.tensor([1.0 * 0])
        h_l = torch.cat((zero_tensor, h[:-1]), 0)
        h_r = torch.cat((h[1:], zero_tensor), 0)
        h = filt[0] * h + filt[2] * h_l + filt[1] * h_r
        if i % 2 == 0:
            h = torch.cat((h[h.shape[0] // 2:], h[:h.shape[0] // 2]), 0)
    return h


# Realize function that get difference between Hypothesis Tensor
# and tainted Images.
def distance_loss(hypothesis, broken_image):
    return torch.dist(hypothesis, broken_image)


# Generate Tensor with Random Values.
random_tensor = torch.randn(10000, dtype=torch.float)

# Set Learning Rate as 0.8
lr = 0.8

# Set requires_grad as True to differentiated
# Error Function as random_tensor.
for i in range(0, 20000):
    random_tensor.requires_grad_(True)

    # Get Hypothesis through weird_function()
    # to random_tensor.
    hypothesis = weird_function(random_tensor)
    loss = distance_loss(hypothesis, broken_image)
    loss.backward()

    # Pytorch are generates Variables Path through
    # in the Neural Network Model.
    with torch.no_grad():
        random_tensor = random_tensor - lr * random_tensor.grad

        if i % 1000 == 0:
            print("Loss at {} = {}".format(i, loss.item()))

    # Restored Image Visualization
    plt.imshow(random_tensor.view(100, 100).data)
