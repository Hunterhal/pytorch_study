import torch
import torch.nn as nn
import torch.nn.functional as F

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

       

        
