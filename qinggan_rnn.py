import collections
import math
import os
import random
import time
import torch
from torch import nn
from torch.utils import data as Data
from torchtext import vocab as Vocab
from tqdm import tqdm

import d2l_pytorch as d2l

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Data_Root = d2l.Data_Root_server


def read_imdb(tag, Data_root=Data_Root):
    Data_root += 'aclImdb'
    data = []
    cnt = 0
    for label in ['pos', 'neg']:
        folder_name = os.path.join(Data_root, tag, label)
        for file in tqdm(os.listdir(folder_name)):
            try:
                with open(os.path.join(folder_name, file), 'rb') as f:
                    review = f.read().decode('utf-8').replace('\n', ' ').replace('\r', ' ')
                    data.append([review, 1 if label == 'pos' else 0])
            except Exception as e:
                cnt += 1

    # print('cannot open file num', cnt)
    random.shuffle(data)
    return data


def get_tokenized_imdb(data):
    def tokenized(text):
        return [tk.lower() for tk in text.split(' ')]

    return [tokenized(review) for review, _ in data]


def get_vocab_imdb(train_data):
    tokenized_data = get_tokenized_imdb(train_data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return Vocab.Vocab(counter, min_freq=5)


def preprocess_imdb(data, vocab):
    max_l = 500

    def pad(x):
        return x[:max_l] if int(len(x)) >= max_l else x + [0] * (max_l - int(len(x)))

    tokenized_data = get_tokenized_imdb(data)
    features = torch.tensor([pad([vocab.stoi(tk) for tk in st]) for st in tokenized_data])
    labels = torch.tensor([score for _, score in data])
    return features, labels


class BiRNN(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # inputs的形状是(批量大小, 词数)，因为LSTM需要将序列长度(seq_len)作为第一维，所以将输入转置后
        # 再提取词特征，输出形状为(词数, 批量大小, 词向量维度)
        # print(inputs.shape)
        embeddings = self.embedding(inputs.permute(1, 0))
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(词数, 批量大小, 2 * 隐藏单元个数)
        outputs, _ = self.encoder(embeddings)  # output, (h, c)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。它的形状为
        # (批量大小, 4 * 隐藏单元个数)。
        # print(outputs.shape, outputs.size())
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        # print(encoding.shape, outputs.size())
        outs = self.decoder(encoding)
        return outs


def load_pretrained_embedding(words, pretrained_vocab):
    embed = torch.zeros(int(len(words)), pretrained_vocab.vectors[0].shape[0])
    oov_count = 0
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, :] = pretrained_vocab.vectors[idx]
        except KeyError:
            oov_count += 1
    if oov_count > 0:
        pass
        # print('There are %d oov words' % oov_count)
    return embed


def evaluate_accuracy(data_iter, net, device):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        X = X.to(device)
        y = y.to(device)
        y_hat = net(X)
        acc_sum += (y_hat.argmax(dim=1) == y).float().sum().cpu().item()
        n += y.shape[0]
    return acc_sum / n


def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.cuda()
    print("training on", device)

    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            if batch_count == 0:
                print(X.shape)
                print(y.shape)
            X = X.cuda()
            y = y.cuda()
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net, device)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


def predict_sentiment(net, vocab, sentence):
    device = list(net.parameters())[0].device
    sentence = torch.tensor([vocab.stoi[tk] for tk in sentence], device=device)
    label = torch.argmax(net(sentence.view((1, -1))), dim=1)
    return 'positive' if label.item() == 1 else 'negative'

if __name__ == '__main__':
    train_data, test_data = read_imdb('train', Data_Root), read_imdb('test', Data_Root)
    vocab = get_vocab_imdb(train_data)

    batch_size = 64
    train_set = Data.TensorDataset(*d2l.preprocess_imdb(train_data, vocab))
    test_set = Data.TensorDataset(*d2l.preprocess_imdb(test_data, vocab))
    train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
    test_iter = Data.DataLoader(test_set, batch_size)

    embed_size, num_hiddens, num_layers = 100, 100, 2
    net = BiRNN(vocab, embed_size, num_hiddens, num_layers)
    glove_vocab = Vocab.GloVe(name='6B', dim=100, cache=os.path.join(Data_Root, 'glove'))

    net.embedding.weight.data.copy_(load_pretrained_embedding(vocab.itos, glove_vocab))
    net.embedding.weight.requires_grad = False

    lr, num_epochs = 0.01, 5
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    loss = nn.CrossEntropyLoss()

    train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

    predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great'])  # positive

    predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'bad'])  # negative
