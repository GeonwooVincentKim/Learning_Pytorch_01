import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets

BATCH_SIZE = 64
lr = 0.01
EPOCHS = 40
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

"""
    Convert loaded IMDB DataSet which are going to use 
    for Neural Network Learning.
"""
TEXT = data.Field(sequential=True, batch_first=True, lower=True)
LABEL = data.Field(sequential=False, batch_first=True)

# Make the DataSets which are inputs into the Model
# by using 'split()' function in 'datasets'.
trainset, testset = datasets.IMDB.splits(TEXT, LABEL)

# Make word vocabulary that needs for Word Embedding
# by using Created DataSet.
TEXT.build_vocab(trainset, min_freg=5)
LABEL.build_vocab(trainset)
