import collections
import math
import random
import sys
import time
import os
import numpy as np
import torch
from torch import nn
import torch.utils.data as Data

import d2l_pytorch as d2l
# print(torch.__version__)

# assert 'ptb.train.txt' in os.listdir('Data/data/ptb.train.txt')

with open('Data/data/ptb.train.txt', 'r') as f:
    lines = f.readlines()
    raw_dataset = [st.split() for st in lines]
print('# sentences: %d' % len(raw_dataset))