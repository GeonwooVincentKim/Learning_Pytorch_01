# Learning Classification using Backpropagation_Algorithm

import torch
import numpy
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Number of Dimension
# Only Possible to input 2.

n_dim = int(input("Please Input Dimension No.2. (Only Number)"))

if n_dim is 2:
    # Now you make Clusters to find out where cluster location
    # which is included data .
    # Make Training Value which is defined as X and Y.
    # make_blobs() is a function which is imported from Scikit-Learn Library.
    x_train, y_train = make_blobs(
        n_samples=80, n_features=n_dim,
        centers=[[1, 1], [-1, -1], [1, -1], [-1, 1]],
        shuffle=True, cluster_std=0.3
    )

    # Make Testing Value which is defined as X and Y.
    x_test, y_test = make_blobs(
        n_samples=20, n_features=n_dim,
        centers=[[1, 1], [-1, -1], [1, -1], [-1, 1]],
        shuffle=True, cluster_std=0.3
    )

else:
    try:
        print(n_dim)

    except:
        TypeError("You might be not type No.2. "
                  "Please reboot program and input No.2 again.")


# Implement label_map() function to change all of the Data
# which have No.0 or No.1 as a Label to No.0 Label.
def label_map(y_, from_, to_):
    y = numpy.copy(y_)
    for f in from_:
        y[y_ == f] = to_
    return y


y_train = label_map(y_train, [0, 1], 0)
y_train = label_map(y_train, [2, 3], 1)
y_test = label_map(y_test, [0, 1], 0)
y_test = label_map(y_test, [2, 3], 1)


# Visualization Data which is Labeled well or not
# by using Matplotlib Library.
def vis_data(x, y=None, c='r'):
    if y is None:
        y = [None] * len(x)
    for x_, y_ in zip(x, y):
        # If Label Number is 0, it shows '.',
        # else Label Number is 1, it shows '+'.
        if y_ is None:
            plt.plot(x_[0], x_[1], '*', markerfacecolor='none',
                     markeredgecolor=c)
        else:
            plt.plot(x_[0], x_[1], c + 'o' if y_ == 0 else c + '+')


plt.figure()
vis_data(x_train, y_train, c='r')
plt.show()

# Convert the Numpy Vector format data you just created
# into Pytorch Tensor format.
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.FloatTensor(y_train)
y_tes = torch.FloatTensor(y_test)
