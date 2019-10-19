"""
 Test 4 different neural network configuration to see the effect of
 activation function and network size. MNIST dataset is used. 
 Author: Mehmet Fatih Gülakar - 19/10/2019
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from abc import abstractmethod
import time

# Load the MNIST dataset.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

epochs = 20

# Abstraction to avoid writing forward() for each class. Only __init__() changes
class Net_3Layer(nn.Module):
    @abstractmethod
    def __init__(self):
        super().__init__()
        pass
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

class Net_2Layer(nn.Module):
    @abstractmethod
    def __init__(self):
        super().__init__()
        pass
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class Net1(Net_3Layer):
    # 3 Convolution and FC layer with sigmoid as activation function.
    def __init__(self):
        super(Net_3Layer, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(20, 40, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(3*3*40, 400)
        self.fc2 = nn.Linear(400, 100)
        self.fc3 = nn.Linear(100, 10)

class Net2(Net_3Layer):
    # 3 Convolution and FC layer with ReLU as activation function.
    def __init__(self):
        super(Net_3Layer, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(20, 40, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)            
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(3*3*40, 400)
        self.fc2 = nn.Linear(400, 100)
        self.fc3 = nn.Linear(100, 10)

class Net3(Net_2Layer):
    # 2 Convolution and FC layer with sigmoid as activation function.
    def __init__(self):
        super(Net_2Layer, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2)            
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2)            
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7*7*20, 100)
        self.fc2 = nn.Linear(100, 10)

class Net4(Net_2Layer):
    # 2 Convolution and FC layer with ReLU as activation function.
    def __init__(self):
        super(Net_2Layer, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)            
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)            
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7*7*20, 100)
        self.fc2 = nn.Linear(100, 10)


netlist = [Net1(), Net2(), Net3(), Net4()] # List of class objects
timelist = []                              # List of elapsed times
accuracy_list = []                         # List of accuracies
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on GPU" if torch.cuda.is_available() else "Running on CPU")
                       

for netNum in range(4):
    print("------------Network #{}---------------".format(netNum+1))
    net = netlist[netNum]
    net.to(device)
    loss_list = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Start to measure time
    start_time = time.time()

    # Training
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
                loss_list.append(running_loss/2000)
                running_loss = 0.0

    elapsed_time = time.time() - start_time
    timelist.append(elapsed_time/60) # in minutes
    correct = 0
    total = 0

    # Testing
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _,predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy_list.append(float(100*correct/total))
    plt.plot(loss_list, '.')

# Printing accuracies and elapsed times of each neural network
print("Accuracy training duration after training for {} epochs".format(epochs))
print("Net1   Net2   Net3   Net4")
print(*accuracy_list, sep="   ")
print(*timelist, sep="   ")

# Plot training loss during training, then save and show the plot.
plt.legend(["3LayerSigmoid({:.2f}min, {}%)".format(timelist[0], accuracy_list[0]),
            "3LayerReLU({:.2f}min, {}%)".format(timelist[1], accuracy_list[1]),
            "2LayerSigmoid({:.2f}min, {}%)".format(timelist[2], accuracy_list[2]),
            "2LayerReLU({:.2f}min, {}%)".format(timelist[3], accuracy_list[3])],loc='best')
plt.xlabel("Training duration")
plt.ylabel("Training loss")
plt.title("Device: {}".format(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))
plt.savefig("nn-performance.png")
plt.show()