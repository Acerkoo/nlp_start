import math
import time

import torch
from torch import nn
from torch import optim

import d2l_pytorch as d2l

device = ('cuda' if torch.cuda.is_available() else 'cpu')

class RnnModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(RnnModel, self).__init__()

        self.rnn = nn.RNN(input_size=vocab_size, hidden_size=hidden_size)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.state = None
        self.dense = nn.Linear(hidden_size, vocab_size)

    def forward(self, X, state):
        X = d2l.to_onehot(X, self.vocab_size)
        Y, self.state = self.rnn(torch.stack(X), state)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state

def predict_rnn_pytorch(prefix, seq_len, model, device, word2idx, idx2word):
    state = None
    output = [word2idx[prefix[0]]]
    for t in range(seq_len + int(len(prefix)) - 1):
        X = torch.tensor([[output[-1]]], device=device).view(1, 1)
        Y, state = model(X, state)
        # print(type(state))
        if t < int(len(prefix)) - 1:
            output.append(word2idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx2word[idx] for idx in output])

def train_and_prefix_rnn_pytorch(prefixes, seq_len, model, device, word2idx, idx2word,
                                 num_epoch, batch_size, lr,
                                 corpus_indices, clipping_theta,
                                 pred_period, pred_len):
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    state = None
    for epoch in range(1):
        data_iter = d2l.data_iter_consecutive(corpus_indices, batch_size, seq_len, device)
        l_sum, n, start = 0.0, 0, time.time()

        for X, Y in data_iter:
            if state is not None:
                if isinstance(state, tuple):
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()
            # if state is not None:
            #     print(state.requires_grad)
            (output, state) = model(X, state)

            # output: [seq_len * batch_size, vocab_size]

            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(output, y.long())

            optimizer.zero_grad()
            l.backward()
            d2l.grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()

            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
            # n += 1

        # try:
        #     perplexity = math.exp(l_sum / n)
        # except OverflowError:
        #     perplexity = float('inf')
        # if (epoch + 1) % pred_period == 0:
        #     print('epoch %d, perplexity %f, time %.2f sec' % (epoch + 1, perplexity, time.time() - start))
        #     for prefix in prefixes:
        #         print(' -', predict_rnn_pytorch(prefix, pred_len, model, device, word2idx, idx2word))


if __name__ == '__main__':
    seq_len, hidden_size = 30, 256
    num_epochs, batch_size, lr, clipping_theta = 10, 32, 1e-3, 1e-2
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    (corpus_indices, word2idx, idx2word, vocab_size) = d2l.load_data_jay_lyrics()

    model = RnnModel(vocab_size, hidden_size).to(device)
    # result = predict_rnn_pytorch('是否', 10, model, device, word2idx, idx2word)

    train_and_prefix_rnn_pytorch(prefixes, seq_len, model, device, word2idx, idx2word,
                                 num_epochs, batch_size, lr, corpus_indices, clipping_theta,
                                 pred_period, pred_len)
