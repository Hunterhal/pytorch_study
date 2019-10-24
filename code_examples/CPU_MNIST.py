#-------------------------------------------------------------------#
# CODE SUMMARY:                                                     #
# 4 Neural Networks are created to show the effects of network size #
# and activation functions. MNIST dataset is used. Accuracy graph   #
# is saved as a png file. Also; accuracies and time takens for all  #
# NNs are saved to an excel file.                                   #
# WARNING!:                                                         #
# "XlsxWriter" package is required! Use either this command         #
# "pip install xlsxwriter" or "conda install -c anaconda xlsxwriter"#
# to install it.                                                    #
#                                                                   #
# Author:     Mehmet KAPSON     20.10.2019                          #
#-------------------------------------------------------------------#
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from pylab import figure
import numpy as np
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer
import xlsxwriter

batch_size = 5 # Change if you want
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./dataMNIST', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.MNIST(root='./dataMNIST', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
# All parameters can change to intended values
class seq_net(nn.Module):
    def __init__(self, act_func):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 20, kernel_size = 4, padding = 1, stride = 1), act_func, nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels = 20, out_channels = 40, kernel_size = 4, padding = 1, stride = 1), act_func,nn.MaxPool2d(2))
        self.fc1 = nn.Linear(in_features = 40*6*6, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 10)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class seq_net_2(nn.Module):
    def __init__(self, act_func):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 10, kernel_size = 4, padding = 2, stride = 1), act_func, nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels = 10, out_channels = 20, kernel_size = 4, padding = 1, stride = 1), act_func,nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels = 20, out_channels = 40, kernel_size = 4, padding = 1, stride = 1), act_func,nn.MaxPool2d(2))
        self.fc1 = nn.Linear(in_features = 40*2*2, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 60)
        self.fc3 = nn.Linear(in_features = 60, out_features = 10)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# act_func can be changed to intended activation function
# NN that using ReLU
net_ReLU = seq_net(act_func = nn.ReLU())  
# Same NN but now using Sigmoid
net_Sigmoid = seq_net(act_func = nn.Sigmoid())
# Different NN with extra 2 layers (1 conv + 1 fc) using ReLU
net_ReLU_2 = seq_net_2(act_func = nn.ReLU())
# Different NN with extra 2 layers (1 conv + 1 fc) but now using Sigmoid
net_Sigmoid_2 = seq_net_2(act_func = nn.Sigmoid())
# List of NNs created
networks = [net_ReLU, net_Sigmoid, net_ReLU_2, net_Sigmoid_2]
length = len(networks)

criterion = nn.CrossEntropyLoss()

accuracy_list = [] 
time_taken_list = []

epochs = 20 # Change if you want
# Training
for net in range(length):
 start = timer() # To see the training time for each NN
 optimizer = optim.SGD(networks[net].parameters(), lr=0.001, momentum=0.9)
 print('##### NET', net+1, '#####')
 for epoch in range(epochs):  # loop over the dataset multiple times
    print('Epoch number:', epoch + 1)
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = networks[net](inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
 end = timer()
 print('Finished training for NET', net+1)
 time_taken = format((end - start)/60, '.3f')
 time_taken_list.append(time_taken) 
 print('Time taken for NET', net+1, ':', time_taken, 'minutes')

 correct = 0
 total = 0
 with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = networks[net](images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

 print('Accuracy of NET', net+1, 'on the 10.000 test images: %d %%' % (100 * correct / total))
 accuracy_list.append(100 * correct / total)

# Save trained model on a file so one can load and use it
torch.save(networks, 'CPU_MNIST_4Networks.pt')

figure(1)
pos = np.arange(length)
  
# Labels for bars 
tick_label = ['NET 1 \n ReLU with 2 conv, 2 fc \n Mins: {0} \n Accuracy: {1}'.format(time_taken_list[0],accuracy_list[0]), 
              'NET 2 \n Sigmoid with same 2 conv, 2 fc \n Mins: {0} \n Accuracy: {1}'.format(time_taken_list[1],accuracy_list[1]), 
              'NET 3 \n ReLU with 3 conv, 3 fc \n Mins: {0} \n Accuracy: {1}'.format(time_taken_list[2],accuracy_list[2]), 
              'NET 4 \n Sigmoid with same 3 conv, 3 fc \n Mins: {0} \n Accuracy: {1}'.format(time_taken_list[3],accuracy_list[3])]
# Plot total accuracies as bar charts
plt.bar(pos, accuracy_list, tick_label = tick_label, 
        width = 0.20, color = ['red', 'green', 'blue', 'yellow']) 

# Naming the y-axis 
plt.ylabel('Accuracy (%)', fontsize = 16) 
# Plot title 
plt.title('Accuracy Comparison (CPU)', fontsize = 18)

plt.figure(1).set_size_inches(25,15)
# Function to save the plot 
plt.savefig('CPU_MNIST_GRAPH.png', bbox_inches='tight')
# Function to show the plot 
plt.show() 

# Write results to an excel file
workbook = xlsxwriter.Workbook('Performance_results.xlsx')
worksheet = workbook.add_worksheet('Results Sheet') 
worksheet.write(0, 0, '[Networks, Results]')
worksheet.write(0, 1, 'Accuracy (%)')
worksheet.write(0, 2, 'Time Taken (mins)') 
row = 0
for net in range(length):
 row+=1
 worksheet.write(row, 0, 'NET ' + str(net+1))
# Write the data to a sequence of cells.
worksheet.write_column(1, 1, accuracy_list)
worksheet.write_column(1, 2, time_taken_list)
workbook.close()