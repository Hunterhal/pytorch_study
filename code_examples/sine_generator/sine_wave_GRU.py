import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random


# signal variables
sampling_freq = 100                
signal_duration = 2
total_samples = sampling_freq*signal_duration

# learning variables
max_epoch = 150 + 1
test_number = 25
learning_rate = 1e-3
batch_size = 16
seq_length = 5

# GRU parameters 
num_layers = 3
hidden_size = 15
input_size = 1
size_out = input_size

x=torch.linspace(0, signal_duration*2*np.pi, total_samples)  # dividing signal_duration * 2 * np.pi interval into (total_samples) pieces with same length, x data
y_gt = torch.sin(x)  # create sine wave for x points, y data

print(x.size(),y_gt.size())  # print sizes of x and y_gt tensors

#plt.plot(x.detach(), y_gt.detach())
#plt.show()

# GRU model
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):  # needed parameters to initialize
        super(GRU, self).__init__()
        self.hidden_size = hidden_size  # to use hidden_size as class member
        self.gru = nn.GRU(input_size, hidden_size, num_layers)  # GRU layers
        self.fc = nn.Linear(hidden_size, output_size)  # a linear layer
    
    def forward(self, x, h):
        out, h = self.gru(x, h)  # use hidden state and input to get output and new hidden state
        out = self.fc(out)  # pass output to linear layer
        return out, h
       
model = GRU(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, output_size = size_out)
print(model)  # print information about GRU (our model)

loss_function = nn.MSELoss()  # loss function for regression mean squared loss
optimizer = optim.Adam(model.parameters(), learning_rate)  # Adam optimizer

# initialize hidden state
h_state = torch.zeros(num_layers, batch_size, hidden_size)

for epoch in range(max_epoch):

    # train loop
    for epoch_index in range(total_samples):            

        input_batch =  torch.zeros(seq_length, batch_size, 1)  # create tensor of zeros with batch_size as its size to use as input
        ground_truth_batch = torch.zeros(seq_length, batch_size, 1)  # create tensor of zeros with batch_size as its size to use as ground truth

        for batch_index in range(batch_size):  # loop for getting random pieces 
            rand_index = random.randint(0, total_samples - 1 - seq_length)   # generate random int between 0 and total_samples-1
            input_batch[:, batch_index, 0] = x[rand_index : rand_index + seq_length]  # copy a portion of it that has size of seq_length to the input_batch
                                                                                      # use random sort order to avoid optimizer be stuck on local minimum
            ground_truth_batch[:, batch_index, 0] = y_gt[rand_index : rand_index+seq_length]  # assign the same random element of y_gt to ground truth tensor's element 

        out, h_state = model(input_batch, h_state)  # prediction and new h_state of model for input_batch and h_state                  
        h_state.detach_()
        loss = loss_function(ground_truth_batch, out)  # ground_truth_batch = target (actual value), out = prediction of our model
                                                       # loss function compares model's prediction and target
        optimizer.zero_grad()  # zero the parameter gradients to prevent them adding up 
        loss.backward()  # compute gradients, backpropagation
        optimizer.step()  # apply gradients to update weights
        # print epoch, iteration and its loss
        print("Epoch: ", str(epoch), " Iteration: ", str(epoch_index), " Loss: ", str(loss.data))
       
    # in the training we took pieces ,but in the test we want to test all xs and ys 
    h_state = torch.zeros(num_layers, batch_size, hidden_size)  # initialize fresh hidden state to use in test loop
    # test loop
    if epoch % test_number == 0:  # put every 'test_number' of epochs to test loop
        out_buffer = torch.zeros(total_samples)  # create tensor of zeros with total_samples as its size to use as out_buffer 
        for iteration_index in range(int(total_samples / (seq_length * batch_size))):
            
            out, h_state = model(x[iteration_index * seq_length * batch_size : (iteration_index + 1) * seq_length * batch_size].view(seq_length, batch_size, -1), h_state)
            out = out.view(-1, 1)[:, 0]

            out_buffer[iteration_index * seq_length * batch_size : (iteration_index + 1) * seq_length * batch_size] = out                
            loss = loss_function(y_gt[iteration_index * seq_length * batch_size : (iteration_index + 1) * seq_length * batch_size], out)

            # print epoch, iteration and its loss
            print("Epoch: ", str(epoch), " Iteration: ", str(iteration_index), " Loss: ", str(loss.data))
        # plot graph
        plt.plot(x.view(total_samples).cpu().detach(), y_gt.view(total_samples).detach())
        plt.plot(x.view(total_samples).cpu().detach(), out_buffer.detach())
        plt.show()
