import torch

# Matrix
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(x)  # Two-Dimensional Space tensor

print("Size : ", x.size())
print("Shape : ", x.shape)
# A Number of Square bracket decides
# what Dimension Space does Program use.
print("Rank (Dimension) : ", x.ndimension())

# Increase Rank
# Add Dimension Space named 1 in the First Place, 0.
x = torch.unsqueeze(x, 0)
print(x)
print("Size (UnSqueeze) : ", x.size())
print("Shape : ", x.shape)
print("Rank (Dimension) : ", x.ndimension())

# Reduce Rank
x = torch.squeeze(x)
print(x)
print("Size (Squeeze) : ", x.size())
print("Shape : ", x.shape)
print("Rank (Dimension) : ", x.ndimension())

# View How much Tensor Size I have.
x = x.view(9)
print(x)
print("Size (View) : ", x.size())
print("Shape : ", x.shape)
print("Rank (Dimension) : ", x.ndimension())

# It will show Error because there is no
# Number of Element of Tensor.
try:
    x = x.view(2, 4)
except Exception as e:
    # But it will show that it is wrong that
    # Adding some Element which is not existing
    # into the Number of Element of Tensor.
    print(e)
