import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models

# Switch to CUDA if it is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define batch size, and number of epochs and sequence length
NUM_EPOCH = 5000
BATCH_SIZE = 10
SEQ_LENGTH = 25                                                     # value BATCH_SIZE * SEQ_LENGTH should be a factor of fsampling

# First generate the time sequence (x), then the function y=f(x) (sin(x) here)
start = 0
end = 2*np.pi
fsampling = 250                                                     # Sampling frequency
x = torch.linspace(start, end, fsampling, device=device)            # Timestep tensor
x = x.view(SEQ_LENGTH,BATCH_SIZE,-1)                                # Reshape the timestep to be suitable for LSTM network
y = torch.sin(x)                                                    # Noiseless sinusodial signal.

# Generate also the noisy sine wave
y_noisy = y + 0.05*torch.randn(x.size(), device=device)             # Noisy sinusodial signal

# Import lstm from models.py file
size_in = int(fsampling/(BATCH_SIZE*SEQ_LENGTH))  
size_out = size_in
size_hidden = 10
num_layers = 5
net = models.LSTM(size_in=size_in, size_hidden=size_hidden, num_layers=num_layers,  size_out=size_out, size_batch=BATCH_SIZE).to(device)
print(net)

# Optimizer and loss functions
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2) # 1e-2 works best, but loss value oscillates
criterion = nn.MSELoss()

# States
h_state = torch.zeros(num_layers, BATCH_SIZE, size_hidden).to(device)
c_state = torch.zeros(num_layers, BATCH_SIZE, size_hidden).to(device)

# Training phase
for epoch in range(NUM_EPOCH):

    prediction, (h_state, c_state) = net(x, (h_state, c_state))
    h_state.detach_()
    c_state.detach_()
    loss = criterion(prediction, y_noisy)
    print("Epoch: {}, Loss:{:5f}".format(epoch+1, loss))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training finished, testing starts")

# Training, finished, testing sine wave with/without noise starts
# Before the test, h_state and c_state should be resetted as zero tensors. (Comment if you want)
h_state = torch.zeros(num_layers, BATCH_SIZE, size_hidden).to(device)
c_state = torch.zeros(num_layers, BATCH_SIZE, size_hidden).to(device)
y_noisy = y + 0.05*torch.randn(x.size(), device=device)         # Uncomment if you want neural network does not trained for specific noise values (If trained with y_noisy)
with torch.no_grad():
    y_test, _ = net(x, (h_state, c_state))
    loss_noisy = criterion(y_test, y_noisy)
    loss_noiseless = criterion(y_test, y)
    
    print("Loss w/ noise:{:2f}, Loss w/out noise:{:2f}".format(loss_noisy, loss_noiseless))




