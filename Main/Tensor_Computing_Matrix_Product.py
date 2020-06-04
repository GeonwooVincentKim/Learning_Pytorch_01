import torch

# Create a tensor with a shape of 5x3 by passing 5 and 3
# as arguments to the randn() function, which creates a tensor
# by randomly picking values from Normal Distribution.
w = torch.randn(5, 3, dtype=torch.float)
# Define a tensor with a 3X3 shape by putting real elements directly.
x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
print("w size : ", w.size())
print("x size : ", x.size())
print("w : ", w)
print("x : ", x)
