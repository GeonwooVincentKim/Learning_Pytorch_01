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

vocab_size = len(TEXT.vocab)
n_classes = 2

print("[Learning-Set]: %d [Vertification-Set]: %d [Test-Set]: %d [Word-Count]: %d [Class]: %d"
      % (len(trainset), len(valset), len(testset), vocab_size, n_classes))


# Set n_layers as below the 2.
class BasicGRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
        super(BasicGRU, self).__init__()
        print("Building Basic GRU model...")
        self.n_layers = n_layers
        self.embed = nn.Embedding(n_vocab, embed_dim)

        # set hidden vector dimension and Dropout.
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_p)

        # Define a RNN Model.
        # nn.RNN sometimes forget the Front-Side Information,
        # because the Gradient became to small or big.
        self.gru = nn.GRU(embed_dim, self.hidden_dim,
                          num_layers=self.n_layers,
                          batch_first=True)
        # When gradient of RNN extremely bigger than usual,
        # it says 'gradient explosion',
        # otherwise, it says 'vanishing gradient'.
        # DL Developer upgraded RNN, and set the name as 'GRU'.

        self.out = nn.Linear(self.hidden_dim, n_classes)
