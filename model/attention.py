import torch
import torch.nn.functional as F
import math 
import torch.nn as nn


# Attention ( Q K V)
def scaled_dot_product(Q , K , V , mask=None):
    """calculate the dot product """
    dk = Q.size(-1)
    q = (Q @ K.transpose(-2,-1)) / math.sqrt(dk)
    if mask is not None:
        q = q.masked_fill(mask == 0, -1e9)
    return F.softmax(q , dim=-1) @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dk = self.d_model // h
        self.Wi_Q = nn.Linear(d_model , d_model) # output shape 
        self.Wi_K = nn.Linear(d_model , d_model)
        self.Wi_V = nn.Linear(d_model , d_model)
        self.W_o = nn.Linear(d_model , d_model)
    
    def forward(self, X, mask=None):
        # shape of X is (batch, sequence ,d_model)
        batch , sequence , d_model = X.size()
        query = self.Wi_Q(X) # Output shape = (batch , sequence , d_model)
        query = query.view(batch , sequence , self.h, self.dk)
        query = query.transpose(1 , 2) # (batch , head, sequence , dk)
        key = self.Wi_K(X) # Output shape = (batch , sequence , d_model)
        key = key.view(batch , sequence , self.h, self.dk)
        key = key.transpose(1 , 2) # (batch , head, sequence , dk)
        value = self.Wi_V(X) # Output shape = (batch , sequence , d_model)
        value = value.view(batch , sequence , self.h, self.dk)
        value = value.transpose(1 , 2) # (batch , head, sequence , dk)
        attention_out = scaled_dot_product(query , key, value)
        attention_out = attention_out.transpose(1, 2).contiguous()
        attention_out = attention_out.view(batch , sequence , self.d_model)
        output = self.W_o(attention_out)
        return output

#------------------------PPE-------------------
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

#-------------------------FFN----------------
class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        
        self.flin = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.slin = nn.Linear(d_ff, d_model)
    def forward(self, X):
        X = self.flin(X)
        X = F.relu(X)
        X = self.dropout(X) # we add dropout to 
        X = self.slin(X)
        return X

class encoder_layer(nn.Module):
    def __init__(self, d_model, d_ff, h,dropout=0.1):
        super().__init__()
        
        self.mha = MultiHeadAttention(h , d_model)
        self.ffn = FFN(d_model , d_ff, dropout)
        self.flayer_norm = nn.LayerNorm(d_model)
        self.slayer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        res = x
        x = self.mha(x, mask=mask)
        
        x = self.flayer_norm(res + self.dropout(x))
        res = x
        x = self.ffn(x)
        
        x = self.slayer_norm(res + self.dropout(x))
        return x

    
if __name__ == '__main__':
    mha = MultiHeadAttention(h=8 , d_model=512)
    X = torch.randn(1, 10, 512)
    out = mha(X)
    print(f"Output shape : {out.shape}")











