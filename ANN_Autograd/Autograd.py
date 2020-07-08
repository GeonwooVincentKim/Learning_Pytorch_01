import torch
import matplotlib.pyplot as plb
# Define Scalar Tensor named "w"
# Meaning of requires_grad is Calculating Differential Variable.
w = torch.tensor(1.0, requires_grad=True)
a = w * 3


# More Complicated Calculate.
l = a**2

# By calling the backward() Function on L,
# w.grad returned the formula to which "w" belongs
# to the derivative of "w", 18.
l.backward()
print("A Value in which 'l' "
      "is Differential by 'w' is ({})".format(w.grad))
