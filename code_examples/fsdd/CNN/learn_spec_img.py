# Basic imports
import glob
import numpy as np
import signal
import os

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms
import torchaudio

# Dataset class and neural network class
from fsdd_dataset_img import MyCustomFSDD

import models
import cv2


# Learning parameters
batch_size = 32
learning_rate = 1e-3
max_epoch = 1000
error_rate = 1e-5
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Dataset variables
data_path = "/home/mehmet/Desktop/bitirme/codes/fsdd/recordings"
all_pngs_in_path = glob.glob(data_path + "/*.png")

# Output files
netname = "net"

# Flag for training and its function
run = True

def signal_handler(signal, frame):
    global run
    run = False

signal.signal(signal.SIGINT, signal_handler)

# Initialize the dataset and dataloader
traindataset = MyCustomFSDD(data_path = data_path, train = True)
trainloader = DataLoader(traindataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 0)

testdataset = MyCustomFSDD(data_path = data_path, train = False)
testloader = DataLoader(testdataset, batch_size = 1, shuffle = True, pin_memory = True, num_workers = 0)

# Initialize the NN model
net = models.Net()
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
            spec = np.transpose(batch[0], (0, 3, 1, 2)) 
            spec = spec.to(device)
            label = batch[1].to(device)
            y_pred = net(spec[:, 0, :, :].unsqueeze(1))   
            optimizer.zero_grad()    
            loss = criterion(y_pred, label)
            loss.backward()
            optimizer.step()
            
            print("TRAIN","Epoch:",epoch+1, "Data-Num:",batchiter, "Loss:",loss.item(), " label: ", label.tolist())

        if epoch % 2 == 0:
            torch.save(net.state_dict(), "./saved_models/" + netname + "_epoch_%d"%(epoch) + ".pth")
        
        if run == False or ((loss.item() < error_rate) and (loss.item() != 0.0)):
            print("SIGINT asserted exiting training loop")
            break
    print("Saving trained model...")
    torch.save(net.state_dict(), "./saved_models/" + netname + "_epoch_%d"%(epoch) + ".pth")
