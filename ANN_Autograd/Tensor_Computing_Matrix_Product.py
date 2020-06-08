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

# Define a Variable named "b"
# which is used in the Matrix Operation
# in addtion to the Matrix Product.
b = torch.randn(5, 2, dtype=torch.float)
print("b size : ", b.size())
print("b : ", b)

# Execute Matrix Product by using torch.mm() function.
# w's Column is 5, and x's Row is 2,
# Which mean shape is [5, 2]
wx = torch.mm(w, x)
print("ws Size : ", wx.size())
print("wx :", wx)

# Adding b Matrix Element to the wx Matrix Element.
result = wx + b
print("result size : ", result.size())
print("result : ", result)
