""" 
Fast prototyping for NN architecture benchmark. Modify the class NN,
optimizer and loss function (if needed) and script will calculate 
training time and accuracy. Then it will write all the data to an Excel file.
TODO :
    1. Graphical results using matplotlib

WARNING: Program requires "openpyxl", "gspread" and "oauth2client" packages,
in Anaconda type "conda install -c anaconda openpyxl gspread oauth2client"
to install them

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
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Change this if needed.
batchSize = 4

# Load the dataset, change if you want
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=0)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=0)
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# Change if needed.
epochs = 10

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
print("-->Training begins")
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
training_is_made_by = "Who made the training? (e.g. John Smith)"
LossFunction = criterion.__class__.__name__
optimizerName = optimizer.__class__.__name__
dataset = trainset.__class__.__name__
time_added = date.today().strftime("%d/%m/%Y")

toWrite = [architecture_description, LossFunction, optimizerName, dataset, batchSize, epochs,
            devicename, accuracy, elapsed_time, time_added, training_is_made_by]

# Try to connect Google sheets. If connected, append row data to sheet.
try:
    scope = ["https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
    client = gspread.authorize(creds)
    worksheet = client.open("Benchmark").sheet1
    worksheet.append_row(toWrite)
    print("Results are imported to Google sheets successfully.")

# If cannot connect due to Internet or problem at Google.
except:
    # Try to open file
    try:
        workbook = openpyxl.load_workbook("Benchmark.xlsx")
    # If "Benchmark.xlsx" does not exist, create a new one.
    except:
        workbook = openpyxl.Workbook()
        worksheet = workbook.worksheets[0]
        worksheet.append(["Description","Loss Function", "Optimizer", "Dataset", "Batch Size", "Epoch", "Device Name", 
                        "Accuracy(%)", "Training duration(min)", "Training Date", "Training is done by"])
    # Select first worksheet and append the data, then save the file.
    worksheet = workbook.worksheets[0]
    worksheet.append(toWrite)
    workbook.save("Benchmark.xlsx")
    print("Results are imported to local Excel file successfully.")


