import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
#signal variables
sampling_freq=200                
signal_duration=2
total_samples=sampling_freq*signal_duration
#learning variables
max_epoch=50
test_number=5
learning_rate=1e-3
batch_size=16
seq_length=5
# RNN parameters 
num_layers = 3
hidden_size = 15
input_size = 1
size_out = input_size
x=torch.linspace(0, signal_duration*2*np.pi, total_samples)
y_gt=torch.sin(x)                         #ground truth y
#print(x.size(),y_gt.size())
#plt.plot(x.detach(), y_gt.detach())
#plt.show()


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, size_out, batch_size):
        super(RNN, self).__init__()
        self.rnn=nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.lin1=nn.Linear(hidden_size, size_out)
        

    def forward(self, x, h):
        out, h=self.rnn(x,h)
        out=self.lin1(out)
        return out,h
       
model= RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, size_out=size_out, batch_size=batch_size)
print(model)

loss_function=nn.MSELoss()
optimizer = optim.Adam(model.parameters(), learning_rate)
# Initialize hidden state
h_state = torch.zeros(num_layers, batch_size, hidden_size)

for epoch in range(max_epoch):                          #how many times we pass the data

    #train loop
    for epoch_index in range(total_samples):            

        input_batch =  torch.zeros(seq_length, batch_size, 1)       #to keep pieces we took from xs and y_gts in one place,
        ground_truth_batch = torch.zeros(seq_length, batch_size, 1)  #we use torch.zeros

        for batch_index in range(batch_size):                   # loop for getting random pieces 
            rand_index = random.randint(0, total_samples - 1-seq_length)   #generates random numbers in the range of 0 and total samples
            input_batch[:, batch_index, 0] = x[rand_index:rand_index+seq_length]                         # Copy a portion of it that has size of seq_length, to the input_batch.
            ground_truth_batch[:, batch_index, 0] = y_gt[rand_index : rand_index+seq_length]  

        out, h_state= model(input_batch, h_state)                   
        h_state.detach_()
        loss=loss_function(ground_truth_batch, out)  #calculate the loss between ground truth y and trained y
        optimizer.zero_grad()                        #zero the gradients to prevent them adding up 
        loss.backward()                              #compute gradients
        optimizer.step()                             #update weights
        print("Epoch: ", str(epoch), " Iteration: ", str(epoch_index), " Loss: ", str(loss.data))
       
       #in the training we took pieces ,but in the test we want to test all xs and ys 
    h_state = torch.zeros(num_layers, batch_size, hidden_size)   
    #test loop
    if epoch % test_number == 0:
        out_buffer=torch.zeros(total_samples)              #hold all the samples
        for iteration_index in range(int(total_samples/(seq_length*batch_size))):

            out, h_state = model(x[iteration_index*seq_length*batch_size : (iteration_index+1)*seq_length*batch_size].view(seq_length, batch_size, -1), h_state)
            out = out.view(-1, 1)[:, 0]

            out_buffer[iteration_index*seq_length*batch_size : (iteration_index+1)*seq_length*batch_size] = out                    
            loss = loss_function(y_gt[iteration_index*seq_length*batch_size : (iteration_index+1)*seq_length*batch_size], out)     

            print("Epoch:", str(epoch), "Iteration:", str(iteration_index), "loss:", str(loss.data))

        plt.plot(x.view(total_samples).cpu().detach(), y_gt.view(total_samples).detach())
        plt.plot(x.view(total_samples).cpu().detach(), out_buffer.detach())
        plt.show()

 
 