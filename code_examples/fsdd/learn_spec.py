# Basic imports
import glob
import matplotlib.pyplot as plt
import numpy as np
import signal
import os
import copy

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms
import torchaudio

# Dataset class and neural network class
from fsdd_dataset import MyCustomFSDD, my_collate

import models

# Learning parameters
batch_size = 64
learning_rate = 1e-3
L1, L2, L3, L4 = (500, 200, 80, 25)
max_epoch = 150
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Dataset variables
data_path = "/home/fatih/code/free-spoken-digit-dataset/recordings"

# Preprocessing variables
n_fft = 128
fbins = n_fft//2 + 1


# Create the Spectrogram transfo
spec_transform = transforms.Spectrogram(n_fft = n_fft, normalized=True)
transform = nn.Sequential(spec_transform, transforms.AmplitudeToDB())

# Output files
netname = "net"

# Flag for training and its function
run = True

# Function for interrupt
def signal_handler(signal, frame):
    global run
    run = False

signal.signal(signal.SIGINT, signal_handler)

# Initialize the dataset and dataloader (Train & Test)
traindataset = MyCustomFSDD(data_path = data_path, train = True, transform = transform)
trainloader = DataLoader(traindataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 0, collate_fn=my_collate)

testdataset = MyCustomFSDD(data_path = data_path, train = False, transform = transform)
testloader = DataLoader(testdataset, batch_size = 1, shuffle = True, pin_memory = True, num_workers = 0)

# Initialize the NN model
net = models.SpectrogramLearner(fbins = fbins, output_dim = 10, hidden_dims = (L1, L2, L3, L4), device=device)
net = net.to(device)

# Optimizer and Loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = learning_rate)

# Training
if __name__ == "__main__":
    for epoch in range(max_epoch):
        batchiter = 0
        for batch in trainloader:

            batchiter += 1

            spec = batch[0][0].squeeze(0).squeeze(0).to(device)
            label = batch[1].to(device)
            seq_length = batch[3].to(device)

  
            y_pred = net(spec[:, 0, :, :])
            
            optimizer.zero_grad()
            loss = criterion(y_pred, label)

            loss.backward()
            optimizer.step()
            
            print("TRAIN","Epoch:",epoch+1, "Data-Num:",batchiter, "Loss:",loss.item())
        if (epoch % 10 == 0):
            torch.save(net.state_dict(), "./saved_models/" + netname + "_epoch_%d"%(epoch) + ".pth")

        
        if run == False:
            print("SIGINT asserted exiting training loop")
            break
    print("Saving trained model...")
    torch.save(net.state_dict(), "./saved_models/" + netname + "_epoch_%d"%(epoch) + ".pth")
