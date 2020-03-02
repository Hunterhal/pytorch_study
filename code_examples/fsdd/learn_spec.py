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
batch_size = 32
learning_rate = 1e-3
L1, L2, L3 = (500, 200, 80)
max_epoch = 100
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
net = models.SpectrogramLearner(fbins = fbins, output_dim = 10, hidden_dims = (L1, L2, L3))
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
            #plt.imshow(spec[0].squeeze(0).cpu())
            #plt.show()
            max_length = torch.max(seq_length)

            indexes = list(range(0, batch_size))

            h1 = torch.zeros(batch_size, L1, requires_grad=True, device=device)
            h2 = torch.zeros(batch_size, L2, requires_grad=True, device=device)

            c1 = torch.zeros(batch_size, L1, requires_grad=True, device=device)
            c2 = torch.zeros(batch_size, L2, requires_grad=True, device=device)
            
            for i in range(max_length):

                state = [h1, c1 ,h2, c2]
                y_pred, last_state = net(spec[:, 0, :, i], state)

                h1 = torch.tensor(last_state[0]) 
                h2 = torch.tensor(last_state[2])
                c1 = torch.tensor(last_state[1])
                c2 = torch.tensor(last_state[3])

                indexes_buffer = copy.deepcopy(indexes)
                for j in indexes_buffer:
                    if i == seq_length[j]:
                        indexes.remove(j)

                if len(indexes) < 5:
                    break

                optimizer.zero_grad()
                
                loss = criterion(y_pred[indexes], label[indexes])

                loss.backward()
                optimizer.step()
            
            print("TRAIN","Epoch:",epoch+1, "Data-Num:",batchiter, "Time-Segment:", i, "Loss:",loss.item(), " index: ", indexes)

        if epoch % 5 == 0:
            #torch.save(net.state_dict(), "./saved_models/" + netname + "_epoch_%d"%(epoch) + ".pth")
            pass
        
        if run == False:
            print("SIGINT asserted exiting training loop")
            break
    print("Saving trained model...")
    torch.save(net.state_dict(), "./saved_models/" + netname + "_epoch_%d"%(epoch) + ".pth")
