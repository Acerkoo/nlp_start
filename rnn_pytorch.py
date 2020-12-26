import math
import time

import torch
from torch import nn
from torch import optim
import sys

sys.path.append('..')
import d2l_pytorch as d2l

device = ('cuda' if torch.cuda.is_available() else 'cpu')


class RnnModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(RnnModel, self).__init__()

        self.rnn = nn.RNN(input_size=vocab_size, hidden_size=hidden_size)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dense = nn.Linear(hidden_size, vocab_size)
        self.state = None

    def forward(self, X, state):
        X = d2l.to_onehot(X, self.vocab_size)
        # print('type of x:', type(X))
        Y, self.state = self.rnn(torch.stack(X), state)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state


def predict_rnn_pytorch(prefix, sentence_len, model, vocab_size, device, idx2word, word2idx):
    state = None
    output = [word2idx[prefix[0]]]
    for t in range(sentence_len + int(len(prefix)) - 1):
        X = torch.tensor([[output[-1]]], device=device).view(1, 1)
        y, state = model(X, state)
        if t < int(len(prefix)) - 1:
            output.append(word2idx[prefix[t + 1]])
        else:
            output.append(int(y.argmax(dim=1).item()))
    return ''.join([idx2word[idx] for idx in output])

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()

def train_and_predict_rnn_pytorch(model, hidden_size, vocab_size, device,
                                  corpus_indices, idx2word, word2idx, num_epochs, num_steps,
                                  lr, clipping_theta, batch_size, pred_period, pred_len, prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = d2l.data_iter_consecutive(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            # if
            output, state = model(X, state)
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(output, y.long())
            optimizer.zero_grad()
            l.backward()

            d2l.grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

            try:
                perplexity = math.exp(l_sum / n)
            except OverflowError:
                perplexity = float('inf')
            if (epoch + 1) % pred_period == 0:
                print('epoch %d, perplexity %f, time %.2f sec' % (
                    epoch + 1, perplexity, time.time() - start))
                for prefix in prefixes:
                    print(' -', predict_rnn_pytorch(prefix, pred_len, model, vocab_size, device, idx_to_char,
                                                    char_to_idx))

hidden_size = 256
seq_len = 25

if __name__ == '__main__':
    model = RnnModel(vocab_size, hidden_size).to(device)
    # result = predict_rnn_pytorch('分开', 10, model, vocab_size, device, idx_to_char, char_to_idx)
    # print(result)

    num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2 # 注意这里的学习率设置
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']

    data_iter = d2l.data_iter_consecutive(corpus_indices, batch_size, seq_len, device)
    for X, Y in data_iter:
        print(Y.shape)
        print(torch.transpose(Y, 0, 1).shape)
        y = torch.transpose(Y, 0, 1).contiguous().view(-1)
        print(y.shape, len(y))
        break
    train_and_predict_rnn_pytorch(model, hidden_size, vocab_size, device,
                                      corpus_indices, idx_to_char, char_to_idx,
                                      num_epochs, seq_len, lr, clipping_theta,
                                      batch_size, pred_period, pred_len, prefixes)
