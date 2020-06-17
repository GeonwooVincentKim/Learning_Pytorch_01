# Learning Classification using Backpropagation_Algorithm

import torch
import numpy
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plot
import torch.nn.functional as F


# Number of Dimension
n_dim = 2

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
