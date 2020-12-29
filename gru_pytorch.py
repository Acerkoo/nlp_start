import torch
import numpy as np

from torch import nn, optim

import d2l_pytorch as d2l

device = ('cuda' if torch.cuda.is_available() else 'cpu')

(corpus_indices, word2idx, idx2word, vocab_size) = d2l.load_data_jay_lyrics()
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

class GruModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(GruModel, self).__init__()

        self.rnn = nn.GRU(input_size=vocab_size, hidden_size=hidden_size)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.state = None
        self.dense = nn.Linear(hidden_size, vocab_size)

    def forward(self, X, state):
        X = d2l.to_onehot(X, self.vocab_size)
        Y, self.state = self.rnn(torch.stack(X), state)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state

num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e-2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

model = GruModel(vocab_size, num_hiddens).to(device)

d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                  corpus_indices, idx2word, word2idx,
                                  num_epochs, num_steps, lr, clipping_theta,
                                  batch_size, pred_period, pred_len, prefixes)