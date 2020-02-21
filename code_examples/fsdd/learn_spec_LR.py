# Basic imports
import glob
import matplotlib.pyplot as plt
import numpy as np
import signal

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa

# Dataset class and neural network class
from fsdd_dataset import MyCustomFSDD
import models


# Learning parameters
batch_size = 16
learning_rate = 1e-3
L1, L2, L3 = (250, 100, 40)
max_epoch = 600
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Dataset variables
data_path = "C:\\Users\\Mehmet\\Desktop\\bitirme\\codes\\fsdd\\free-spoken-digit-dataset-master\\recordings"

# Preprocessing variables
n_fft = 256
fbins = n_fft//2 + 1

# Output files
netname = "net1.pth"

# Flag for training and its function
run = True

def signal_handler(signal, frame):
    global run
    run = False

signal.signal(signal.SIGINT, signal_handler)

# Initialize the dataset and dataloader
traindataset = MyCustomFSDD(data_path = data_path, train = True)
trainloader = DataLoader(traindataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 0)

testdataset = MyCustomFSDD(data_path = data_path, train=False)
testloader = DataLoader(testdataset, batch_size = 1, shuffle = True, pin_memory = True, num_workers = 0)

# Initialize the NN model
net = models.SpectrogramLearner(fbins = fbins, output_dim = 10, hidden_dims = (L1, L2, L3), batch_size = batch_size)
net = net.to(device)

# Optimizer and Loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = learning_rate)
net = net.float()
# Training
if __name__ == "__main__":
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

        
            for i in range(spec.size()[2]):
                h1 = h1.float()
                h2 = h2.float()
                c1 = c1.float()
                c2 = c2.float()
                state = [h1, c1 ,h2, c2]
                y_pred, last_state = net(spec[:, :, i].float(), state)

                """
                Following gives the error of
                UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() 
                or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
            
                ****Did what it recommends but program gave errors.****
            
                """
                h1 = torch.tensor(last_state[0]) 
                h2 = torch.tensor(last_state[2])
                c1 = torch.tensor(last_state[1])
                c2 = torch.tensor(last_state[3])


                optimizer.zero_grad()
                loss = criterion(y_pred, label)
            
                loss.backward()
                optimizer.step()
            
                print("TRAIN","Epoch:",epoch+1, "Data-Num:",dataiter, "Time-Segment:", i, "Loss:",loss.item(), " label: ", label.tolist())
        
        if run == False:
            print("SIGINT asserted exiting training loop")
            break
    print("Saving trained model...")
    torch.save(net.state_dict(), "./" + netname)
