import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import random


# signal variables
sampling_freq = 100
signal_duration = 2
total_samples = sampling_freq * signal_duration

# learning variables
max_epoch = 50 + 1
test_number = 5
learning_rate = 1e-3
batch_size = 64

x = torch.linspace(0, signal_duration * 2 * np.pi, total_samples)
y_gt = torch.sin(x)

print(x.size(), y_gt.size())

#plt.plot(x.detach(), y_gt.detach())
#plt.show()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # use gpu if it is available
                                                                         # GPU implementation is used to speed up the process
                                                                         # it takes more time than FCN on CPU
# GRU nn model
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):  # needed parameters to initialize
        super(GRU, self).__init__()
        self.hidden_size = hidden_size  # to use hidden_size as class member
        self.gru = nn.GRU(input_size, hidden_size, num_layers)  # GRU layers
        self.fc = nn.Linear(hidden_size, output_size)  # a linear layer
    
    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)  # use hidden state and input to get output and new hidden state
        output = self.fc(output)  # pass output to linear layer
        return output, hidden

# define some parameters -- hidden_size and num_layers can be changed
input_size = 1
hidden_size = 10
output_size = 1
num_layers = 3

model = GRU(input_size = input_size, hidden_size = hidden_size, output_size = output_size, num_layers = num_layers).to(device)  # send GRU model to GPU
print(model)

h0 = torch.zeros(num_layers, input_size, hidden_size)  # create first hidden state tensor with zeros

# send datas to GPU on next 4 lines
h0 = h0.to(device)
x = x.to(device)
y_gt = y_gt.to(device)


loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), learning_rate)

for epoch in range(max_epoch):

    # train loop
    for epoch_index in range(total_samples):

        input_batch = torch.zeros(batch_size, 1, 1).to(device)
        ground_truth_batch = torch.zeros(batch_size, 1, 1).to(device)
        h0_hidden_batch = torch.zeros(num_layers, 1, hidden_size).to(device)  # create first hidden state tensor with zeros
        
        for batch_index in range(batch_size):
            rand_index = random.randint(0, total_samples - 1)
            input_batch[batch_index, 0, 0] = x[rand_index]
            #h0_hidden_batch[:, batch_index, :] = h0[:, rand_index, :]
            ground_truth_batch[batch_index, 0, 0] = y_gt[rand_index]

        out, hn = model(input_batch, h0_hidden_batch)
        out = out.to(device)
        hn = hn.to(device)

        loss = loss_function(ground_truth_batch, out)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        print("Epoch: ", str(epoch), " Iteration: ", str(epoch_index), " Loss: ", str(loss.data))

    """
    # test loop
    if epoch % test_number == 0:
        out_buffer = torch.zeros(1, 1, total_samples)
        for iteration_index in range(total_samples):
            #x[iteration_index] = torch.unsqueeze(x[iteration_index], dim = 0)  
            #x[iteration_index] = torch.unsqueeze(x[iteration_index], dim = 0) # to add 1 more dimension, so new x data has 3 dimension, which is needed as input of GRU model
            out, hn = model(x[iteration_index].unsqueeze(0), h0[num_layers, iteration_index, hidden_size])
            out = out.to(device)
            hn = hn.to(device)

            out_buffer[0, 0, iteration_index] = out

            loss = loss_function(y_gt[iteration_index], out)

            optimizer.zero_grad()

            loss.backward()

            if iteration_index < 172:
                optimizer.step()

        
            print("Epoch: ", str(epoch), " Iteration: ", str(iteration_index), " Loss: ", str(loss.data))
  
        plt.plot(x.detach(), y_gt.detach())
        plt.plot(x.detach(), out_buffer[0, :].detach())
        plt.show()
    """
