import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from timeit import default_timer as timer
import time


# signal variables
sampling_freq = 100
signal_duration = 2
total_samples = sampling_freq * signal_duration

# learning variables
max_epoch = 150 + 1
test_number = 25
learning_rate = 1e-3
batch_size = 16

x = torch.linspace(0, signal_duration * 2 * np.pi, total_samples)  # dividing signal_duration * 2 * np.pi interval into (total_samples) pieces with same length, x data
y_gt = torch.sin(x)  # create sine wave for x points, y data

print(x.size(), y_gt.size())  # print sizes of x and y_gt tensors

#plt.plot(x.detach(), y_gt.detach())
#plt.show()

# model FCN with 3 layers created 10 neurons used, parameters can be changed to intended values
class FCN_3L(nn.Module):
    def __init__(self):
        super(FCN_3L, self).__init__()
        self.fc1 = nn.Linear(in_features = 1, out_features = 10)
        self.fc2 = nn.Linear(in_features = 10, out_features = 10)
        self.fc3 = nn.Linear(in_features = 10, out_features = 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # use activation function to add non-linearity
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = FCN_3L()  # fully connected neural network
print(model)  # print information about FCN_3L (our model)

loss_function = nn.MSELoss()  # loss function for regression mean squared loss
optimizer = optim.Adam(model.parameters(), learning_rate)  # Adam optimizer
start = timer()  # start timer

for epoch in range(max_epoch):

    # train loop
    for epoch_index in range(total_samples):

        input_batch = torch.zeros(batch_size, 1)  # create tensor of zeros with batch_size as its size to use as input
        ground_truth_batch = torch.zeros(batch_size, 1)  # create tensor of zeros with batch_size as its size to use as ground truth

        for batch_index in range(batch_size):  # loop for getting random pieces 
            rand_index = random.randint(0, total_samples - 1)  # generate random int between 0 and total_samples-1
            input_batch[batch_index, 0] = x[rand_index]  # assign random element of x to input_batch tensor's element
                                                         # use random sort order to avoid optimizer be stuck on local minimum
            ground_truth_batch[batch_index, 0] = y_gt[rand_index]  # assign the same random element of y_gt to ground truth tensor's element

        out = model(input_batch)  # prediction of model for input_batch

        loss = loss_function(ground_truth_batch, out)  # ground_truth_batch = target (actual value), out = prediction of our model
                                                       # loss function compares model's prediction and target
        optimizer.zero_grad()  # zero the parameter gradients

        loss.backward()  # compute gradients, backpropagation

        optimizer.step()  # apply gradients to update weights
        # print epoch, iteration and its loss
        print("Epoch: ", str(epoch), " Iteration: ", str(epoch_index), " Loss: ", str(loss.data))

    # test loop
    if epoch % test_number == 0:  # put every 'test_number' of epochs to test loop
        out_buffer = torch.zeros(1, total_samples)  # create tensor of zeros with total_samples as its size to use as out_buffer 
        for iteration_index in range(total_samples):
            out = model(x[iteration_index].unsqueeze(0))  # a prediction of model for an element of x data at a time, unsqueeze it to pass data to model
        
            out_buffer[0, iteration_index] = out  # assign prediction of model to out_buffer tensor's element

            loss = loss_function(y_gt[iteration_index], out)  # y_gt[iteration_index] = target (actual value), out = prediction of our model

            optimizer.zero_grad()  # zero the parameter gradients

            loss.backward()  # compute gradients, backpropagation

            if iteration_index < 172:  # while this condition is true, use optimizer.step(), after this condition do not train again just test
                optimizer.step()  # apply gradients

            # print epoch, iteration and its loss
            print("Epoch: ", str(epoch), " Iteration: ", str(iteration_index), " Loss: ", str(loss.data))
        # plot graph
        plt.plot(x.detach(), y_gt.detach())
        plt.plot(x.detach(), out_buffer[0, :].detach())
        plt.show()
        end = timer()  # end timer
        elapsed_time = format((end - start)/60, '.3f')  # calculate elapsed time
        print('Elapsed time: ', elapsed_time, ' mins')
        time.sleep(2)
