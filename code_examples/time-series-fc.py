# Generate sine wave, train it with a fully connected network (2 hidden layer)
# author: Mehmet Fatih Gülakar 14/11/2019

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable

import imageio

fsampling = 250
x = torch.unsqueeze(torch.linspace(0, 2*np.pi, fsampling), dim=1)  # x data (tensor), shape=(fsampling, 1)
y = torch.sin(x)                                                   # y data (tensor), shape=(fsampling, 1)

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)

# view data
plt.figure(figsize=(10,4))
plt.scatter(x.data.numpy(), y.data.numpy(), color = "green")
plt.title('Regression Analysis')
plt.xlabel('Independent var.')
plt.ylabel('Dependent var.')
plt.grid()
plt.show()

num_epoch = 12000

class FCNet(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(n_feature, n_hidden[0]), nn.ReLU(), nn.Linear(n_hidden[0], n_hidden[1]))
        self.outlayer = nn.Linear(n_hidden[1], n_output)

    def forward(self, x):
        x = nn.functional.relu(self.hidden(x))
        x = self.outlayer(x)
        return x

net  = FCNet(n_feature=1, n_hidden=(15,25), n_output=1)
print(net)
opitimizer = torch.optim.SGD(net.parameters(), lr = 0.1) # Tried both Adam and SGD, Adam trains faster but loss starts to oscillate, while SGD does not change much.
criterion = nn.MSELoss()


gif_list = []   # For storing the graphs as training continues.
fig, ax = plt.subplots(figsize=(12,7))

# Training
for epoch in range(num_epoch):
    
    prediction = net(x)
    loss = criterion(prediction, y)
    print("Epoch:%d, loss=%.4f"%(epoch, loss))
    opitimizer.zero_grad()
    loss.backward()
    opitimizer.step()

    # Append every 40th epoch
    if epoch % 40 == 39:
        plt.cla()
        ax.set_title('Regression Analysis')
        ax.set_xlabel('Independent variable')
        ax.set_ylabel('Dependent variable')
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(-1.05, 1.05)
        ax.scatter(x.data.numpy(), y.data.numpy(), color="orange")
        ax.plot(x.data.numpy(), prediction.data.numpy(), 'g-', lw=3)
        ax.text(1.0, 0.1, "Step = %d"%(epoch+1), fontdict={'size':24, 'color': 'red'})
        ax.text(1.0, 0, "Loss = %.4f"%loss.data.numpy(), fontdict={'size':24, 'color': 'red'})

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3, ))

        gif_list.append(frame)


imageio.mimsave('./fullyconnected.gif', gif_list, fps=25)
