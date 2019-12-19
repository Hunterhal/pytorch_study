import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import models

#Signal variables
sampling_freq = 200
signal_duration = 2
total_samples = sampling_freq * signal_duration
#Learning variables
device = torch.device('cpu')
max_epoch = 1000
test_number = 10
learning_rate = 1e-3
batch_size = 16
seq_length = 5

# LSTM parameters 
num_layers = 3
size_hidden = 15
size_in = 1
size_out = size_in

x = torch.linspace(0, signal_duration * 2 * np.pi, total_samples).to(device)
y_gt = torch.sin(x)

model = models.LSTM(size_in=size_in, num_layers=num_layers, size_hidden=size_hidden, size_out=size_out, size_batch=batch_size)
print(model)

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), learning_rate)

# Initialize hidden and cell states
h_state = torch.zeros(num_layers, batch_size, size_hidden).to(device)
c_state = torch.zeros(num_layers, batch_size, size_hidden).to(device)

for epoch in range(max_epoch):
    #Train Loop
    for epoch_index in range(total_samples):

        input_batch = torch.zeros(seq_length, batch_size, 1).to(device)                                 # Tensor for storing random samples of x
        ground_truth_batch = torch.zeros(seq_length, batch_size, 1).to(device)                          # Tensor for storing the samples of y_gt, which has the same indexes with x.

        for batch_index in range(batch_size):
           rand_index = random.randint(0, total_samples-1-seq_length)                                   # Generate a random index through x.
           input_batch[:, batch_index, 0] = x[rand_index:rand_index+seq_length]                         # Copy a portion of it that has size of seq_length, to the input_batch.
           ground_truth_batch[:, batch_index, 0] = y_gt[rand_index : rand_index+seq_length]             # Do the same thing for y.

        out, (h_state, c_state)  = model(input_batch, (h_state, c_state))
        h_state.detach_()
        c_state.detach_()

        loss = loss_function(ground_truth_batch, out)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        print("TRAIN - Epoch: ", str(epoch), " Iteration: ", str(epoch_index), " Loss: ", str(loss.data))

    # Test loop

    if epoch % test_number == 0:

        # Reset the hidden and cell states first.
        h_state = torch.zeros(num_layers, batch_size, size_hidden).to(device)
        c_state = torch.zeros(num_layers, batch_size, size_hidden).to(device)

        out_buffer = torch.zeros(total_samples)

        # In the loop a number of sections of x/y are tested. It is total of (seq_length * batch_size)
        for iteration_index in range(int(total_samples/(seq_length*batch_size))):

            # Take the portion of x (which depends on iteration_index), reshape it so that it can be fed to the LSTM network.
            out, (h_state, c_state) = model(x[iteration_index*seq_length*batch_size : (iteration_index+1)*seq_length*batch_size].view(seq_length, batch_size, -1), (h_state, c_state))

            # Shape of output is [seq_length, batch_size, input_size], we need to use it for testing. So reshape it to (seq_length*batch_size*input_size, 1) tensor
            # Then reduce it to the (seq_length*batch_size*input_size) tensor by [:, 0]
            out = out.view(-1, 1)[:, 0]

            out_buffer[iteration_index*seq_length*batch_size : (iteration_index+1)*seq_length*batch_size] = out                     # Same thing done on tensor x above. 
            loss = loss_function(y_gt[iteration_index*seq_length*batch_size : (iteration_index+1)*seq_length*batch_size], out)      # Calculate loss between the section of y_gt and output of LSTM.

            print("TEST - Epoch: ", str(epoch), " Iteration: ", str(iteration_index), " Loss: ", str(loss.data))

        plt.plot(x.view(total_samples).cpu().detach(), y_gt.view(total_samples).cpu().detach())
        plt.plot(x.view(total_samples).cpu().detach(), out_buffer.cpu().detach())
        plt.show()




