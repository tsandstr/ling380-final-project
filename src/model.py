import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, vocab_size, input_size, hidden_size, num_layers,
                 dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.vocab_size = vocab_size
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, input_size)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           dropout=dropout)
        self.decoder = nn.Linear(hidden_size, vocab_size)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf
        # 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for
        # Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if hidden_size != input_size:
                raise ValueError('When using the tied flag, hidden_size must \
                be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.vocab_size)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.hidden_size),
                weight.new_zeros(self.num_layers, bsz, self.hidden_size))
