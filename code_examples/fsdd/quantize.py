import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Insert here the neural network class you want here.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 5, padding = 0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 7, 5, padding = 0)
        self.fc1 = nn.Linear(175, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.float())))
        x = self.pool(F.relu(self.conv2(x.float())))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 

# Initialize the network and load the pretained model to it.
model = Net()
model.load_state_dict(torch.load("net_conv.pth"))

# Extract weights and biases, then write them to .h file.
for name, param in model.named_parameters():
    print('name: ', name, "param.shape", param.shape)
    min_wt = param.data.min()
    max_wt = param.data.max()
    int_bits = int(np.ceil(np.log2(max(abs(min_wt), abs(max_wt)))))
    frac_bits = 7-int_bits
    quant = np.round(param.data*(2**frac_bits))

    var_name = name.replace('.', '_')
    with open("weigths_conv.h", 'a') as f:
        f.write("#define " + var_name.upper() + ' {')
        if(len(quant.shape) > 2): # Conv layer weight
            transposed_wts = np.transpose(quant, (3,0,1,2))
        else: # Fully Connected layer weights or biases of any layer
            transposed_wts = np.transpose(quant)
        print(type(transposed_wts))
        transposed_wts.numpy().tofile(f, sep=', ', format="%d")
        f.write('}\n')
    print("==========================")

    


    


    




