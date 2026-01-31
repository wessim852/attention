import torch
import torch.nn.functional as F
import math 
import torch.nn as nn

N = 5
dk = 64
d_model = 512
heads = 8
batch_size = 1
# Attention ( Q K V)
def scaled_dot_product(Q , K , V , mask=None):
    """calculate the dot product """
    dk = Q.size(-1)
    q = (Q @ K.transpose(-2,-1)) / math.sqrt(dk)
    if mask is not None:
        q = q.masked_fill(mask == 0, -1e9)
    return F.softmax(q , dim=-1) @ V


Wi_Q = nn.Linear(d_model , d_model) # output shape 
Wi_K = nn.Linear(d_model , d_model)
Wi_V = nn.Linear(d_model , d_model)
W_o = nn.Linear(d_model , d_model)
def multi_head_attention(X):
    """Linear Projection
    Input (batch , N , d_model)
    Output shape (batch , sequence , d_model)
    Output (1,5,512)"""
    batch , sequence , d_model= X.size()

    """Reshape to (batch,sequence,heads,dk)"""
    query = Wi_Q(X) # X (batch , N , d_model) => batch , sequence , d_model
    
    query = query.view(batch , sequence , 8 ,64) # heads = 8 , dk = 64
    query = query.transpose(1,2)
    key = Wi_K(X)
    key = key.view(batch , sequence , 8 ,64)
    key = key.transpose(1,2)
    value = Wi_V(X)
    value = value.view(batch, sequence , 8 ,64)
    value = value.transpose(1,2)
    attention_out = scaled_dot_product(query , key , value) # (batch , 8 , 5 , 64) head = 8 seq =5
    attention_out = attention_out.transpose(1,2).contiguous()
    attention_out = attention_out.view(batch , sequence , d_model)
    output = W_o(attention_out)
    return output




X = torch.randn(batch_size , N , d_model)
output = multi_head_attention(X)
print(f'Input shape :{X.shape}')
print(f'Output shape :{output.shape}')
