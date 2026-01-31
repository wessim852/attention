import torch
import torch.nn as nn
import math
max_len = 500
d_model = 64
class positional_encoding(nn.Module):
    def __init__(self, d_model = 64 , max_len = 500):
        super().__init__()
        pe = torch.zeros(max_len , d_model)
        pos = torch.arange(0 , max_len).unsqueeze(1) # making pos a column vector 
        freq_term = torch.exp(torch.arange(0 , d_model , 2)*-(math.log(10000.0)/d_model))
        
        pe[: , 0::2] = torch.sin(pos * freq_term)
        pe[: , 1::2] = torch.cos(pos * freq_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
    def forward(self, x):
        # x shape = batch_size , sequence , d_model
        x = x + self.pe[:,:x.size(1),:]
        return x
