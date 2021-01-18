import collections
import os
import time
import random

import torch
from torch import nn
import torchtext.vocab as Vocab
import torch.utils.data as Data
import torch.nn.functional as F
from tqdm import tqdm

import d2l_pytorch as d2l

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Data_Root = d2l.Data_Root_server


def read_imdb(tag, data_root=Data_Root):
    data = []
    data_root += 'aclImdb'
    for label in ['pos', 'neg']:
        folder_name = os.path.join(data_root, tag, label)
        for file in tqdm(os.listdir(folder_name)):
            try:
                with open(os.path.join(folder_name, file), 'rb') as f:
                    review = f.read().decode('utf-8').replace('\n', ' ').replace('\r', ' ')
                    data.append([review, 1 if label == 'pos' else 0])
            except Exception as e:
                pass

    random.shuffle(data)
    return data


def get_tokenized_data(data):
    def tokenizer(text):
        return [tk.lower() for tk in text.split(' ')]

    return [tokenizer(review) for review, _ in data]


def get_vocab_imdb(data):
    tokenized_data = get_tokenized_data(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return Vocab.Vocab(counter, min_freq=5)


def preprocess_imdb(data, vocab):
    max_l = 500

    def pad(x):
        return x[:max_l] if int(len(x)) > 500 else x + [0] * (max_l - int(len(x)))

    tokenized_data = get_tokenized_data(data)
    features = torch.tensor([pad([vocab.stoi[tk] for tk in st]) for st in tokenized_data])
    labels = torch.tensor([tag for _, tag in data])
    return features, labels


def corr1d(X, K):
    w = K.shape[0]
    Y = torch.zeros(X.shape[0] - w + 1)
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y


def corr1d_multi_in(X, K):
    # print(([corr1d(x, k) for x, k in zip(X, K)]))
    return torch.stack([corr1d(x, k) for x, k in zip(X, K)]).sum(dim=0)


# X = torch.tensor([[0, 1, 2, 3, 4, 5, 6],
#               [1, 2, 3, 4, 5, 6, 7],
#               [2, 3, 4, 5, 6, 7, 8]])
# K = torch.tensor([[1, 2], [3, 4], [-1, -3]])
# Y = corr1d_multi_in(X, K)
# print(Y)

class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
        return F.max_pool1d(x, kernel_size=x.shape[2])


class TextCNN(nn.Module):
    def __init__(self, vocab, embed_size, kernel_sizes, num_channels):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # 不参与训练的嵌入层
        self.constant_embedding = nn.Embedding(len(vocab), embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # 时序最大池化层没有权重，所以可以共用一个实例
        self.pool = GlobalMaxPool1d()
        self.convs = nn.ModuleList()  # 创建多个一维卷积层
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels = 2*embed_size,
                                        out_channels = c,
                                        kernel_size = k))

    def forward(self, inputs):
        # 将两个形状是(批量大小, 词数, 词向量维度)的嵌入层的输出按词向量连结
        embeddings = torch.cat((
            self.embedding(inputs),
            self.constant_embedding(inputs)), dim=2) # (batch, seq_len, 2*embed_size)
        # 根据Conv1D要求的输入格式，将词向量维，即一维卷积层的通道维(即词向量那一维)，变换到前一维
        embeddings = embeddings.permute(0, 2, 1)
        # 对于每个一维卷积层，在时序最大池化后会得到一个形状为(批量大小, 通道大小, 1)的
        # Tensor。使用flatten函数去掉最后一维，然后在通道维上连结
        encoding = torch.cat([self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.convs], dim=1)
        # 应用丢弃法后使用全连接层得到输出
        outputs = self.decoder(self.dropout(encoding))
        return outputs


def evaluate_accuracy(test_iter, net, device):
    acc_sum, n = 0.0, 0
    for X, y in test_iter:
        X = X.to(device)
        y = y.to(device)
        y_hat = net(X)
        acc_sum += (y_hat.argmax(dim=1) == y).float().sum().cpu().item()
        n += y.shape[0]
    return acc_sum / n


if __name__ == '__main__':
    batch_size = 64
    train_data, test_data = read_imdb('train', data_root=Data_Root), read_imdb('test', data_root=Data_Root)
    vocab = get_vocab_imdb(train_data)
    train_set = Data.TensorDataset(*preprocess_imdb(train_data, vocab))
    test_set = Data.TensorDataset(*preprocess_imdb(test_data, vocab))

    train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
    test_iter = Data.DataLoader(test_set, batch_size)

    embed_size, kernel_sizes, num_channels = 100, [3, 4, 5], [100, 100, 100]
    net = TextCNN(vocab=vocab, embed_size=embed_size, kernel_sizes=kernel_sizes, num_channels=num_channels)

    glove_vocab = Vocab.GloVe(name='6B', dim=100, cache=os.path.join(Data_Root, 'glove'))
    net.embedding.weight.data.copy_(d2l.load_pretrained_embedding(vocab.itos, glove_vocab))
    net.constant_embedding.weight.data.copy_(d2l.load_pretrained_embedding(vocab.itos, glove_vocab))
    net.constant_embedding.weight.requires_grad = False

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    loss = nn.CrossEntropyLoss()

    net = net.to(device)
    print('training on ', device)

    batch_count = 0
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)

            y_hat = net(X)
            l = loss(y_hat, y).sum()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_loss_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().cpu().item()
            n += y.shape[0]

        test_acc = evaluate_accuracy(test_iter, net, device)
        print('epoch = %d, loss = %.4f, train_acc = %.4f, test_acc = %.4f, cost = %.4f'
              % (epoch + 1, train_loss_sum / n, train_acc_sum / n, test_acc, time.time() - start))
