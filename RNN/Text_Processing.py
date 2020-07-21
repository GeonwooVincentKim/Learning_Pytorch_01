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

trainset, valset = trainset.splie(split_ratio=0.8)
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (trainset, valset, testset),
    batch_size=BATCH_SIZE, shuffle=True,
    repeat=False
)

vocab_size=len(TEXT.vocab)
n_classes = 2

print("[Learning-Set]: %d [Vertification-Set]: %d [Test-Set]: %d [Word-Count]: %d [Class]: %d"
      % (len(trainset), len(valset), len(testset), vocab_size, n_classes))
