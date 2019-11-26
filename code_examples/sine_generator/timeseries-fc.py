import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models

# Switch to CUDA if it is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define batch size, and number of epochs.
NUM_EPOCH = 5000
BATCH_SIZE = 5

# First generate the time sequence (x), then the function y=f(x) (sin(x) here)
start = 0                                                                           # Starting value
end = 2*np.pi                                                                       # Ending value
fsampling = 200                                                                     # Sampling frequency
x = torch.linspace(start, end, fsampling, device=device)                            # Timestep tensor
x = x.view(BATCH_SIZE,-1)                                                           # Reshape the timestep
y = torch.sin(x)                                                                    # Noiseless sine wave

y_noisy = y + 0.05*torch.randn(x.size(), device=device)                             # Noisy sine wave

# Import the 3 layer fully connected models from models.py file
D_in = int(fsampling/BATCH_SIZE)                                                     # Input size
D_out = D_in                                                                         # Output size
H = (20, 30)                                                                         # Tuple of sizes of hidden layers
net = models.FCNet(D_in=D_in, H=H, D_out=D_out).to(device)
print(net)

# Optimizer and loss functions
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
criterion = nn.MSELoss()

# Training phase
for epoch in range(NUM_EPOCH):

    optimizer.zero_grad()
    prediction = net(x)
    loss = criterion(prediction, y_noisy)
    print("Epoch: {}, Loss:{:5f}".format(epoch+1, loss))
    loss.backward()
    optimizer.step()
    
print("Training finished, testing starts")

# Training, finished, testing sine wave with/without noise starts
y_noisy = y + 0.05*torch.randn(x.size(), device=device)         # Uncomment if you want neural network does not trained for specific noise values (If trained with y_noisy)
with torch.no_grad():
    y_test = net(x)
    loss_noisy = criterion(y_test, y_noisy)
    loss_noiseless = criterion(y_test, y)
    
    print("Loss w/ noise:{:2f}, Loss w/out noise:{:2f}".format(loss_noisy, loss_noiseless))




