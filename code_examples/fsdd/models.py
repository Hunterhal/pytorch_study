import torch
import torch.nn as nn
import torch.nn.functional as F


# Model used in learn-spec.py
class SpectrogramLearner(nn.Module):
    def __init__(self, fbins, output_dim, hidden_dims, device):
        super().__init__()
        self.device = device
        self.L1, self.L2, self.L3, self.L4 = hidden_dims
        self.lstm1  = nn.LSTMCell(fbins, self.L1)
        self.lstm2  = nn.LSTMCell(self.L1, self.L2)
        self.lstm3  = nn.LSTMCell(self.L2, self.L3)
        self.fc1    = nn.Linear(self.L3, self.L4)
        self.fc2    = nn.Linear(self.L4, output_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        tlen = x.size(2)
       
        h1 = torch.rand(batch_size, self.L1, device=self.device)
        h2 = torch.rand(batch_size, self.L2, device=self.device)
        h3 = torch.rand(batch_size, self.L3, device=self.device)
        c1 = torch.rand(batch_size, self.L1, device=self.device)
        c2 = torch.rand(batch_size, self.L2, device=self.device)
        c3 = torch.rand(batch_size, self.L3, device=self.device)
        

        for t in range(tlen):
            x_ = x[:, :, t]
            h1, c1 = self.lstm1(x_, (h1, c1))
            h2, c2 = self.lstm2(h1,(h2, c2))
            h3, c3 = self.lstm3(h2,(h3, c3))
            output = F.relu(self.fc1(h3))
            output = self.fc2(output)
        
        return output

       

        