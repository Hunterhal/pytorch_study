""" 
Fast prototyping for NN architecture benchmark. Modify the class NN,
optimizer and loss function (if needed) and script will calculate 
training time and accuracy. Then it will write all the data to an Excel file.
TODO :
    1. Append results to Google Sheets instead of local Excel file. (or check
internet connectivity, if it fails, write to local Excel file.)
    2. Graphical results using matplotlib


WARNING: Program requires "openpyxl" package,
in Anaconda type "conda install -c anaconda openpyxl" to install it

Author: Mehmet Fatih Gülakar - 19/10/2019

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
import time
import platform
import openpyxl
from datetime import date

# Change this if needed.
batchSize = 4

# Load the dataset, change if you want
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=0)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=0)
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

epochs = 1

# Change the NN architecture for benchmarking.
class Net(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

# Determine whether GPU or CPU is used, then obtain its name to write it on Excel file.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
devicename = torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()

net = Net()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
start_time = time.time()

# Train
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i%2000 == 1999:
            print('[%d %5d] loss: %.3f'%( epoch+1, i+1, running_loss/2000))
            running_loss = 0.0

elapsed_time = (time.time() - start_time)/60
correct = 0
total = 0

# Test
with torch.no_grad():
    for data in testloader:
        test_inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(test_inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()

accuracy = float(100*correct/total)
architecture_description = "Referring to a text file here might be helpful"
simulation_is_made_by = "Who made the simulation? (e.g. John Smith)"
LossFunction = criterion.__class__.__name__
optimizerName = optimizer.__class__.__name__
dataset = trainset.__class__.__name__
time_added = date.today().strftime("%d/%m/%Y")

toWrite = [architecture_description, LossFunction, optimizerName, dataset, batchSize,
            devicename, accuracy, elapsed_time, time_added, simulation_is_made_by]

# Try to load Benchmark.xlsx. If it fails, create a new one, write column names, and append the simulation results.
try:
    workbook = openpyxl.load_workbook("Benchmark.xlsx")
except:
    workbook = openpyxl.Workbook()
    worksheet = workbook.worksheets[0]
    worksheet.append(["Description","Loss Function", "Optimizer", "Dataset", "Batch Size", "Device Name", 
                        "Accuracy(%)", "Training Time(min)", "Simulation Date", "Simulation is done by"])
# Select first worksheet
worksheet = workbook.worksheets[0]
worksheet.append(toWrite)
workbook.save("Benchmark.xlsx")



