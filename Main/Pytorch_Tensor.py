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
print("Size : ", x.size())
print("Shape : ", x.shape)
print("Rank (Dimension) : ", x.ndimension())

# Reduce Rank
x = torch.squeeze(x)
print(x)
print("Size : ", x.size())
print("Shape : ", x.shape)
print("Rank (Dimension) : ", x.ndimension())
