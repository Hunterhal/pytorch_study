import torch
import torch.nn as nn
import torch.nn.functional as F


# Model used in learn-spec.py
class SpectrogramLearner(nn.Module):
    def __init__(self, fbins, output_dim, hidden_dims):
        super().__init__()
        self.L1, self.L2, self.L3 = hidden_dims
        self.lstm1  = nn.LSTMCell(fbins, self.L1)
        self.lstm2  = nn.LSTMCell(self.L1, self.L2)
        self.fc1    = nn.Linear(self.L2, self.L3)
        self.fc2    = nn.Linear(self.L3, output_dim)
    
    def forward(self, x, state):
        state[0], state[1] = self.lstm1(x, (state[0], state[1]))
        state[2], state[3]= self.lstm2(state[0], (state[2], state[3]))
        output = F.relu(self.fc1(state[2]))
        output = self.fc2(output)
        retState = [state[0], state[1], state[2], state[3]]
        return output, retState

# Model used in learn_spec_img.py (ADJUST IT)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 5, padding = 0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 13, 5, padding = 0)
        self.fc1 = nn.Linear(4901, 1200)
        self.fc2 = nn.Linear(1200, 200)
        self.fc3 = nn.Linear(200, 50)
        self.fc4 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.float())))
        x = self.pool(F.relu(self.conv2(x.float())))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x    
