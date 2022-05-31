import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_dim, num_layers = 1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, input_dim, num_layers, batch_first=True)

    def forward(self, x):
        out, _ = self.lstm(x)  #out : (batch_size, sequence_len, hidden_size)
        
        return out
