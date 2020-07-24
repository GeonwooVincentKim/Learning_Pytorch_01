"""
    Seq2Seq is the model which connected two RNN
    models each playing different roles.
"""
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt

vocab_size = 256   # Length of Entire ASCII Code.
x_ = list(map(ord, "hello"))   # Convert to ASCII Code.
y_ = list(map(ord, "hola"))    # Convert to ASCII Code.
x = torch.LongTensor(x_)
y = torch.LongTensor(y_)


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Seq2Seq, self).__init__()
        self.n_layers = 1
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.GRU(hidden_size, hidden_size)
        self.decoder = nn.GRU(hidden_size, hidden_size)
        self.project = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, targets):
        initial_state = self._init_state()
        embedding = self.embedding(inputs).unsqueeze(1)
        encoder_output, encoder_state = self.encoder(embedding, initial_state)
        decoder_state = encoder_state
        decoder_input = torch.LongTensor([0])

    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()

