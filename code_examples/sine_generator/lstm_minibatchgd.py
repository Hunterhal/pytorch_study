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
test_number = 25
learning_rate = 1e-3
batch_size = 16
seq_length = 5

num_layers = 5
size_hidden = 10
size_in = 1
size_out = size_in


x = torch.linspace(0, signal_duration * 2 * np.pi, total_samples).to(device)
x = x.view(seq_length, batch_size, -1)
y_gt = torch.sin(x)

print("xsize:", x.size(), "ysize:", y_gt.size())

#plt.plot(x.detach(), y_gt.detach())
#plt.show()

model = models.LSTM(size_in=size_in, num_layers=num_layers, size_hidden=size_hidden, size_out=size_out, size_batch=batch_size)
print(model)

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), learning_rate)

h_state = torch.zeros(num_layers, batch_size, size_hidden).to(device)
c_state = torch.zeros(num_layers, batch_size, size_hidden).to(device)

for epoch in range(max_epoch):
    #Train Loop
    for epoch_index in range(total_samples):

        input_batch = torch.zeros(seq_length, batch_size, 1).to(device)
        ground_truth_batch = torch.zeros(seq_length, batch_size, 1).to(device)

        for batch_index in range(batch_size):
            rand_index_batch = random.randint(0, batch_size- 1)   # generate random index for batch
            rand_index_in = random.randint(0, int(total_samples/(batch_size*seq_length)-1)) # generate random index for input_size

            input_batch[:, batch_index, 0] = x[:, rand_index_batch, rand_index_in]
            ground_truth_batch[:, batch_index, 0] = y_gt[:, rand_index_batch, rand_index_in]

        out, (h_state, c_state)  = model(input_batch, (h_state, c_state))
        h_state.detach_()
        c_state.detach_()

        loss = loss_function(ground_truth_batch, out)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        print("TRAIN - Epoch: ", str(epoch), " Iteration: ", str(epoch_index), " Loss: ", str(loss.data))

    #Test loop
    # if epoch % test_number == 0:
    #     out_buffer = torch.zeros(1, total_samples)
    #     for iteration_index in range(total_samples):
    #         out, (h_state, c_state)= model(x[iteration_index].unsqueeze(0), (h_state, c_state))

    #         out_buffer[0, iteration_index] = out

    #         loss = loss_function(y_gt[iteration_index], out)

    #         print("TEST - Epoch: ", str(epoch), " Iteration: ", str(iteration_index), " Loss: ", str(loss.data))

    #     plt.plot(x.cpu().detach(), y_gt.cpu().detach())
    #     plt.plot(x.cpu().detach(), out_buffer[0, :].cpu().detach())
    #     plt.show()




