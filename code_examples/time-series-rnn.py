# Generate sine wave, train it with a Recurrent Neural Network
# author: Mehmet Fatih Gülakar 20/11/2019

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import imageio

num_epoch = 5000
seq_length = 250
fsampling = 250

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

x = torch.unsqueeze(torch.linspace(0, 2*np.pi, fsampling), dim=1)   # shape = (fsampling, 1)
y = torch.sin(x)                                                    # same shape as x

# Add dummy dimensions to be utilizable by RNN
x = torch.unsqueeze(x, dim=0).to(device)
y = torch.unsqueeze(y, dim=0).to(device)
# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)

# view data
plt.figure(figsize=(10,4))
plt.scatter(x.cpu().data.numpy(), y.cpu().data.numpy(), color = "blue")
plt.title('Regression Analysis')
plt.xlabel('Independent varible')
plt.ylabel('Dependent varible')
plt.grid()
plt.show()

# Define the class RNN
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
        self.outlayer = nn.Linear(32, 1)

    def forward(self, x, h_state):
        y_out, h_state = self.rnn(x, h_state)
        y_out = y_out.view(-1, 32)
        out = self.outlayer(y_out)
        out = out.view(-1, seq_length, 1)
        return out, h_state

# Net object initialization and printing, loss and optimizers
net = RNN().to(device)
print(net)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)


h_state = None  # init state
gif_list = []   # List to store the gifs generated images of selected epochs during the training
fig, ax = plt.subplots(figsize=(12,7))

for epoch in range(num_epoch):
    
    prediction, h_state = net(x, h_state)
    h_state = h_state.data
    
    loss = criterion(prediction, y)
    optimizer.zero_grad()
    print("Epoch:%d, loss=%.4f"%(epoch, loss))

    loss.backward()
    optimizer.step()

    # Append every 40th epoch to the gif_list
    if epoch % 40 == 39:
        plt.cla()
        ax.set_title('Regression Analysis')
        ax.set_xlabel('Independent variable')
        ax.set_ylabel('Dependent variable')
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(-1.05, 1.05)
        ax.scatter(np.squeeze(x.cpu().data.numpy()), np.squeeze(y.cpu().data.numpy()), color="orange")
        ax.plot(np.squeeze(x.cpu().data.numpy()), np.squeeze(prediction.cpu().data.numpy()), 'g-', lw=3)
        ax.text(1.0, 0.1, "Step = %d"%(epoch+1), fontdict={'size':24, 'color': 'red'})
        ax.text(1.0, 0, "Loss = %.4f"%loss.cpu().data.numpy(), fontdict={'size':24, 'color': 'red'})

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3, ))

        gif_list.append(frame)


imageio.mimsave('./rnn.gif', gif_list, fps=25)