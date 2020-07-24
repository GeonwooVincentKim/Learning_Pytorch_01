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
print("hello -> ", x_)
print("hello -> ", y_)
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

        # Predict 'o' Token by using 'h' Token in next Iterators
        # when you predicted "h" token from "hola" translated sentences
        # by using ASCII Number '0', the Initial placed of Sentences Token.
        outputs = []
        for i in range(targets.size()[0]):
            decoder_input = self.embedding(decoder_input).unsqueeze(1)
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)

            # Predict Next syllable by Output of Decoder.
            projection = self.project(decoder_output)
            outputs.append(projection)

            # Update Decoder Input by using 'Teacher-Forcing'.
            decoder_input = torch.LongTensor([targets[i]])
        outputs = torch.stack(outputs).squeeze()
        return outputs

    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_size).zero_()


seq2seq = Seq2Seq(vocab_size, 16)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(seq2seq.parameters(), lr=1e-3)

log = []
for i in range(1000):
    prediction = seq2seq(x, y)
    loss = criterion(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_val = loss.data
    log.append(loss_val)

    if i % 100 == 0:
        print("\n Iterates: %d, Error: %s" % (i, loss_val.item()))
        _, top1 = prediction.data.topk(1, 1)
        print([chr(c) for c in top1.squeeze().numpy().tolist()])
