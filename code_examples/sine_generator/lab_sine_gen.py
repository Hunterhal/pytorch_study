import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

#Signal variables
sampling_freq = 200
signal_duration = 2
total_samples = sampling_freq * signal_duration
#Learning variables
max_epoch = 1000
test_number = 25
learning_rate = 1e-3
batch_size = 32

x = torch.linspace(0, signal_duration * 2 * np.pi, total_samples)
y_gt = torch.sin(x)

#print(x.size(), y_gt.size())
#plt.plot(x.detach(), y_gt.detach())
#plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(1, 100)
        self.lin2 = nn.Linear(100, 25)
        self.lin3 = nn.Linear(25, 1)

    def forward(self, x):
        x = F.relu( self.lin1(x) )
        x = F.relu( self.lin2(x) )
        x = self.lin3(x)

        return x

model = Net()
print(model)

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), learning_rate)

for epoch in range(max_epoch):
    #Train Loop
    for epoch_index in range(total_samples):

        input_batch = torch.zeros(batch_size, 1)
        ground_truth_batch = torch.zeros(batch_size, 1)

        for batch_index in range(batch_size):
            rand_index = random.randint(0, total_samples - 1)
            input_batch[batch_index, 0] = x[rand_index] 
            ground_truth_batch[batch_index, 0] = y_gt[rand_index] 

        out = model(input_batch)

        loss = loss_function(ground_truth_batch, out)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        print("Epoch: ", str(epoch), " Iteration: ", str(epoch_index), " Loss: ", str(loss.data))

    #Test loop
    if epoch % test_number == 0:
        out_buffer = torch.zeros(1, total_samples)
        for iteration_index in range(total_samples):
            out = model(x[iteration_index].unsqueeze(0))

            out_buffer[0, iteration_index] = out

            loss = loss_function(y_gt[iteration_index], out)

            print("Epoch: ", str(epoch), " Iteration: ", str(iteration_index), " Loss: ", str(loss.data))

        plt.plot(x.detach(), y_gt.detach())
        plt.plot(x.detach(), out_buffer[0, :].detach())
        plt.show()




