# Basic Imports
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

# PyTorch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Import Custom eGRU Modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.utils.class_weight import compute_class_weight
from eGRUModules import eGRUCell, QLinear


from scipy.signal import medfilt

# Audio Playback Modules
import librosa as lbr
from librosa.display import waveplot, specshow
from IPython.display import display,Image, Audio
from tqdm import tqdm, tqdm_notebook, trange

th.manual_seed(7);
device = 'cuda' if th.cuda.is_available() else 'cpu'

use_quant   = True         # Use Quantization
use_binary  = True         # Convert to Binary
use_cweight = True         # Use Class Weights (for Imbalanced Classes)

max_epoch  = 500
batch_size = 128

maxlen  = 24
dataset = 'coughdataset.pkl'

netname = 'net.pth'
logname = 'log.csv'


# Get Data
def get_data():
    df = pd.read_pickle(dataset)
    cmap = dict(enumerate(df.Class.cat.categories))
    print('')
    print('Entries in dataset: %d' % (len(df)))
    print('Class Map: ', cmap)

    # Data Preparations
    # Truncation
    pad = lambda a, n: a[:, 0: n] if a.shape[1] > n else np.hstack((a, np.zeros([a.shape[0], n - a.shape[1]])))
    Stft = df.Stft.apply(pad, args=(maxlen,))

    # Get X & Y Data
    x_data = np.dstack(Stft.values).transpose(2, 1, 0)  # size: (batch_size, timesteps, dim)
    y_data = df.Class.cat.codes.values.astype('int32')

    return x_data, y_data, cmap

# Get Event Selection
def get_selection(s,thresh=30,bg=0,fg=1,fwin=9):
    ss = np.sum(s,axis=-1)        # Sum along dimension
    ss = ss/ss.max()              # Normalize
    yt = np.percentile(ss,thresh) # Get threshold as percentile of sums
    ys = bg*np.ones_like(ss)      # New array filled with background, bg
    ys[ss>yt] = fg                # Highlight selections with foreground, fg
    ys = medfilt(ys,fwin)         # Median filter to smoothen with fwin window length
    return ys

def get_dataloader(x_data,y_data,batch_size=32):
    # Convert to Torch Tensors
    x = th.from_numpy(x_data).float()
    y = th.from_numpy(y_data).long()

    x = x.to(device)
    y = y.to(device)
    # TensorDataset & Loader
    dataset = TensorDataset(x,y)
    loader  = DataLoader(dataset,batch_size=batch_size,shuffle=True,drop_last=True)
    return loader

x_data,y_data,cmap = get_data()

# Get Finer Sequential labels
if use_binary:
    y_data = np.array([get_selection(x, fg=y, bg=1) if y == 0 else np.ones(24) for x, y in zip(x_data, y_data)])

# Binarize
if use_binary:
    y_data[y_data == 2] = 1
    cmap = {0: 'cough', 1: 'noise'}
    print('New Map:', cmap)

# Data Splitting Indices
indices = range(len(y_data))
itrain, itestn, _, _ = train_test_split(indices, y_data, test_size=0.3, random_state=32)
itrain, ivalid, _, _ = train_test_split(itrain, y_data[itrain], test_size=0.25, random_state=32)

# Dataloaders
trainloader = get_dataloader(x_data[itrain], y_data[itrain], batch_size)
validloader = get_dataloader(x_data[ivalid], y_data[ivalid], batch_size)

print ('Train size: %d'%len(itrain))
print ('Testn size: %d'%len(itestn))
print ('Valid size: %d'%len(ivalid))

def sample_labelplot():
    K   = random.choice(range(len(x_data)))
    sd = np.sum(x_data[K],axis=-1)
    sd = sd/sd.max()
    plt.figure(figsize=(4,4))
    plt.subplot(211);specshow(x_data[K].T,cmap='viridis');
    plt.subplot(212);plt.margins(0);
    plt.plot(sd);plt.plot(y_data[K],c='r'),plt.ylim([0,1.02])
    plt.subplots_adjust(hspace=0)

sample_labelplot()

if use_cweight:
    cweight = compute_class_weight('balanced',np.unique(y_data),y_data.flatten())
    print('Class weights: ',cweight)
    cweight = th.tensor(cweight,dtype=th.float).to(device)
else:
    cweight = None

# Plot Class Distribution
plt.figure(figsize=(4,2))
sns.distplot(y_data.flatten(),kde=False);
plt.title('Class Distribution');

# Network Architecture
# Dimensions
steps,fbins = x_data.shape[1:]
nclass      = len(np.unique(y_data))
L1,L2,L3    = 32,20,16


# Neural Network
class Network(nn.Module):
    def __init__(self, dev='cpu'):
        super(Network, self).__init__()
        self.dev = dev
        self.gru1 = eGRUCell(fbins, L1, use_quant)
        self.gru2 = eGRUCell(L1, L2, use_quant)
        self.fc3 = QLinear(L2, L3)
        self.fc4 = QLinear(L3, nclass)

    def forward(self, inputs):
        blen = inputs.size(0)
        tlen = inputs.size(1)

        h1 = th.Tensor(blen, L1).uniform_(-1.0, 1.0).to(self.dev)
        h2 = th.Tensor(blen, L2).uniform_(-1.0, 1.0).to(self.dev)
        h = []

        for t in range(tlen):
            x = inputs[:, t, :]
            h1 = self.gru1(x, h1)
            h2 = self.gru2(h1, h2)
            h3 = self.fc3(h2).clamp(min=0)
            h4 = self.fc4(h3)
            h.append(h4)
        y = th.stack(h).permute(1, 0, 2)
        return F.log_softmax(y, dim=-1)

net = Network(dev=device)
net = net.to(device)
opt = optim.Adam(net.parameters(),lr=0.001)

def crt(x,y):
    xp = x.permute(0,2,1)
    return F.nll_loss(xp,y,weight=cweight)

# Mode Prediction
def get_pred(outs):
    pred   = th.argmax(outs,-1)
    return pred


LOSS, ACCU = [], []


def do_training(max_epoch):
    for epoch in (range(max_epoch)):
        rloss = []
        acc = 0.0
        for i, data in enumerate(trainloader):
            x, y = data
            opt.zero_grad()

            outs = net(x)
            loss = crt(outs, y)
            loss.backward()
            opt.step()

            rloss.append(loss.item())

        total, correct = 0., 0.
        with th.no_grad():
            for data in validloader:
                x, y = data
                outs = net(x)
                pred = get_pred(outs)
                total += y.flatten().size(0)
                correct += (pred.flatten() == y.flatten()).sum().item()

        LOSS.append(np.mean(rloss))
        ACCU.append(correct / total)
        tqdm.write('Epoch %d/%d, Loss: %.4f, Accu: %.4f' % (epoch + 1, max_epoch, LOSS[-1], ACCU[-1]))
    return

do_training(max_epoch)

th.save(net.state_dict(), "./" + netname)

plt.figure(figsize=(8,3))
plt.subplot(121)
plt.plot(range(len(LOSS)),LOSS,'g'); plt.title('Loss');
plt.subplot(122)
plt.plot(range(len(ACCU)),ACCU,); plt.title('Accuracy');
