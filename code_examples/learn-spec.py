# Basic imports
import glob
import matplotlib.pyplot as plt
import numpy as np

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms

# Dataset class and neural network class
from fsdd_dataset import MyCustomFSDD
import models

# Learning parameters
batch_size = 1
learning_rate = 1e-3
L1, L2, L3 = (250, 100, 40)
max_epoch = 5
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Dataset variables
folder_data = glob.glob("/home/fatih/free-spoken-digit-dataset/recordings/*.wav")
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# Preprocessing variables
n_fft = 1024
fbins = n_fft//2 + 1

# Output files
netname = "net.pth"

# Create the Spectrogram transform
spec_transform = transforms.Spectrogram(n_fft=n_fft)

# Initialize the dataset and dataloader
traindataset = MyCustomFSDD(folder_data, transfrom=spec_transform)
trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)

# Initialize the NN model
net = models.SpectrogramLearner(fbins=fbins, output_dim=10, hidden_dims=(L1, L2, L3), batch_size=batch_size)
net = net.to(device)

# Optimizer and Loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), learning_rate, momentum=0.9)

# Training
for epoch in range(max_epoch):
    dataiter = 0
    
    for batch in trainloader:
        dataiter += 1
        spec = batch[0][0].squeeze(0).squeeze(0).to(device)
        label = batch[1].to(device)

        h1 = torch.zeros(batch_size, L1, requires_grad=True, device=device)
        h2 = torch.zeros(batch_size, L2, requires_grad=True, device=device)

        c1 = torch.zeros(batch_size, L1, requires_grad=True, device=device)
        c2 = torch.zeros(batch_size, L2, requires_grad=True, device=device)

        
        for i in range(spec.size()[1]):

            state = [h1, c1 ,h2, c2]
            y_pred, last_state = net(spec[:, i].unsqueeze(0), state)

            h1 = torch.tensor(last_state[0])
            h2 = torch.tensor(last_state[2])
            c1 = torch.tensor(last_state[1])
            c2 = torch.tensor(last_state[3])

            optimizer.zero_grad()
            loss = criterion(y_pred, label)
            
            loss.backward()
            optimizer.step()
            
            print("Epoch:",epoch+1, "Data-Num:",dataiter, "Time-Segment:", i, "Loss:",loss.item())

print("Saving trained model...")
torch.save(net.state_dict(), "./" + netname)