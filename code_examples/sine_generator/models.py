import torch
import torch.nn as nn
import torch.nn.functional as F

# Fully Connected Network, 3 layer
class FCNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.layer1 = nn.Linear(D_in, H[0])
        self.layer2 = nn.Linear(H[0], H[1])
        self.outlayer = nn.Linear(H[1], D_out)
    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        return self.outlayer(out)

# LSTM Network
class LSTM(nn.Module):
    def __init__(self, size_in, size_hidden, num_layers, size_out, size_batch):
        super().__init__()

        self.lstm = nn.LSTM(input_size=size_in, hidden_size=size_hidden, num_layers=num_layers)
        self.outlayer = nn.Linear(size_hidden, size_out)

    def forward(self, x, h_and_c):
        out, h_and_c = self.lstm(x, h_and_c)
        out = self.outlayer(out)
        return out, h_and_c
        
# Model used in learn-spec.py
class SpectrogramLearner(nn.Module):
    def __init__(self, fbins, output_dim, hidden_dims, batch_size):
        super().__init__()
        self.batch_size = batch_size
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

       

        