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
    
    def forward(self, q, k, v, mask=None):
        batch = q.size(0)
        query = self.Wi_Q(q).view(batch, -1, self.h, self.dk).transpose(1,2)
        key = self.Wi_K(k).view(batch, -1, self.h, self.dk).transpose(1,2)
        value = self.Wi_V(v).view(batch, -1, self.h, self.dk).transpose(1,2)
        attention_out = scaled_dot_product(query, key, value, mask)
        attention_out = attention_out.transpose(1,2).contiguous()
        output = self.W_o(attention_out.view(batch, -1, self.d_model))

        
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
        x = self.mha(x, x, x,mask=mask)
        
        x = self.flayer_norm(res + self.dropout(x))
        res = x
        x = self.ffn(x)
        
        x = self.slayer_norm(res + self.dropout(x))
        return x

class decoder_layer(nn.Module):
    def __init__(self, d_model, d_ff, h, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(h, d_model)
        self.cross_attn = MultiHeadAttention(h, d_model)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.flayer_norm = nn.LayerNorm(d_model)
        self.slayer_norm = nn.LayerNorm(d_model)
        self.tlayer_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
    def forward(self, x , enc_out,src_mask, target_mask):
        # x = decoder sequence
        # first sublayer
        res = x 
        x = self.self_attn(x,x,x,mask=target_mask)
        x = self.flayer_norm(res + self.dropout(x))
        # second sublayer query from decoder which is x , key value from encoder
        query = x 
        out = self.cross_attn(query,enc_out,enc_out,mask=src_mask)
        out = self.slayer_norm(query + self.dropout(out))
        temp = out
        out = self.ffn(out)
        out = self.tlayer_norm(temp + self.dropout(out))

        return out 
        
class Decoder(nn.Module):
    def __init__(self,vocab_size, N_layers, d_model, max_len, d_ff, h):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = positional_encoding(d_model, max_len)
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.block = nn.ModuleList([decoder_layer(d_model, d_ff, h, dropout=0.1) for _ in range(N_layers)])   
    
    def forward(self, x, enc_out, src_mask, target_mask):
        x = self.word_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for layer in self.block:
            x = layer(x, enc_out, src_mask, target_mask)
        return x




class Encoder(nn.Module):
    def __init__(self, vocab_size, N_layers, d_model, max_len , d_ff, h):
        super().__init__()
        self.d_model = d_model
        
        
        self.pos_encoding = positional_encoding(d_model, max_len)
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        
        self.block = nn.ModuleList([encoder_layer(d_model, d_ff, h, dropout=0.1) for _ in range(N_layers)])

    def forward(self , x, mask=None):
        
        x = self.word_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.block:
            x = layer(x, mask)
        return x

class Transformer(nn.Module):
    def __init__(self, encoder, decoder ,src_vocab_size, trg_vocab_size,N_layers, d_model, max_len , d_ff, h):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, N_layers, d_model, max_len, d_ff, h)
        self.decoder = Decoder(trg_vocab_size, N_layers, d_model, max_len, d_ff, h)
        self.linear = nn.Linear(d_model, trg_vocab_size)
    
    def forward(self, src , trg , src_mask , trg_mask):
        out = self.encoder(src, mask=src_mask)
        decoder_out = self.decoder(trg, out, src_mask, trg_mask)
        decoder_out = self.linear(decoder_out)
        return decoder_out


    
if __name__ == '__main__':
    mha = MultiHeadAttention(h=8 , d_model=512)
    X = torch.randn(1, 10, 512)
    out = mha(X)
    print(f"Output shape : {out.shape}")











