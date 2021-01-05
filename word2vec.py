import collections
import math
import os
import random
import sys
import time

import torch
from torch import utils
from torch import nn
from torch.nn import functional as F
from torch.utils import data as Data

import d2l_pytorch as d2l

Data_root = d2l.Data_Root_server
file_name = os.path.join(Data_root, 'ptb', 'ptb.train.txt')
# print(file_name)
with open(file_name, 'r') as f:
    lines = f.readlines()
    raw_dataset = [st.split() for st in lines]

# print(len(raw_dataset))

# for st in raw_dataset[:5]:
#     print('len:', len(st), 'first 5 words:', st[:5])

counter = collections.Counter([tk for st in raw_dataset for tk in st])
counter = dict(filter(lambda x: x[1] >= 5, counter.items()))

idx2token = [tk for tk, _ in counter.items()]
token2idx = {tk: idx for idx, tk in enumerate(idx2token)}
dataset = [[token2idx[tk] for tk in st if tk in token2idx] for st in raw_dataset]
num_tokens = sum([int(len(st)) for st in dataset])

# print(num_tokens)

def discard(idx):
    return random.uniform(0, 1) < 1 - math.sqrt(1e-4 / counter[idx2token[idx]] * num_tokens)

sub_datasets = [[tk for tk in st if not discard(tk)] for st in dataset]
# print(sum([int(len(st)) for st in sub_datasets]))

def compare_counts(token):
    return '# %s: before=%d, after=%d' % (token,\
                                          int(sum([st.count(token2idx[token]) for st in dataset])),\
                                          int(sum([st.count(token2idx[token]) for st in sub_datasets])))

# print(compare_counts('the'))
# print(compare_counts('join'))

def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:
             continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i - window_size), min(int(len(st)), center_i + window_size + 1)))
            indices.remove(center_i)
            contexts.append([st[idx] for idx in indices])
    return centers, contexts

all_centers, all_contexts = get_centers_and_contexts(sub_datasets, 5)

def get_negatives(all_contexts, sampling_weights, K):
    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(int(len(sampling_weights))))
    for contexts in all_contexts:
        negative = []
        while int(len(negative)) < int(len(contexts)) * K:
            if i == int(len(neg_candidates)):
                i, neg_candidates = 0, random.choices(population, sampling_weights, k=int(1e5))

            neg, i = neg_candidates[i], i + 1
            if neg not in set(contexts):
                negative.append(neg)
        all_negatives.append(negative)
    return all_negatives


sampling_weights = [counter[w] ** 0.75 for w in idx2token]
all_negatives = get_negatives(all_contexts, sampling_weights, 5)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, centers, contexts, negatives):
        assert int(len(centers)) == int(len(contexts)) == int(len(negatives))
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, idx):
        return (self.centers[idx], self.contexts[idx], self.negatives[idx])

    def __len__(self):
        return len(self.centers)

def batchify(data):
    max_len = max(int(len(c)) + int(len(n)) for _, c, n in  data)
    centers, contexts_negatives,  masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = int(len(context)) + int(len(negative))
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * int(len(context)) + [0] * (max_len - int(len(context)))]
    return (torch.tensor(centers).view(-1, 1), torch.tensor(contexts_negatives),
            torch.tensor(masks), torch.tensor(labels))


batch_size = 512
num_workers = 0 if sys.platform.startswith('win32') else 4

dataset = MyDataset(all_centers,
                    all_contexts,
                    all_negatives)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True,
                            collate_fn=batchify, num_workers=num_workers)

for batch in data_iter:
    for name, data in zip(['centers', 'contexts_negatives', 'masks', 'labels'],
                          batch):
        print(name, 'shape:', data.shape)
    break

embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
# print(embed.weight)

def skip_gram(center, contexts_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred

class SigmoidBinaryCrossEntorpyLoss(nn.Module):
    def __init__(self):
        super(SigmoidBinaryCrossEntorpyLoss, self).__init__()

    def forward(self, inputs, targets, mask=None):
        '''
        :param inputs: Tensor shape: (batch_size, len)
        :param targets: Tensor of the same shape as input
        :param mask:
        :return:
        '''
        inputs, targets, mask = inputs.float(), targets.float(), mask.float()
        res = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', weight=mask)
        return res.mean(dim=1)

loss = SigmoidBinaryCrossEntorpyLoss()

embed_size = 100
net = nn.Sequential(
    nn.Embedding(num_embeddings=len(idx2token), embedding_dim=embed_size),
    nn.Embedding(num_embeddings=len(idx2token), embedding_dim=embed_size)
)

def train(net, lr, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('train on', device)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for batch in data_iter:
            center, context_negative, mask, label = [d.to(device) for d in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.view(label.shape), label, mask) * mask.shape[1] \
                 / mask.float().sum(dim=1)).mean()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.cpu().item()
            n += 1
        print('epoch %d, loss %.2f, time %.2fs'
              % (epoch + 1, l_sum / n, time.time() - start))


train(net, 0.01, 5)
