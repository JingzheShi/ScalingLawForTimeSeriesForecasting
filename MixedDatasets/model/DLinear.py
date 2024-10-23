import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, MLP
import numpy as np

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.linear = nn.Linear(configs.seq_len, configs.pred_len)
        # nn.Sequential(
        #     MLP(configs.seq_len, configs.d_model, 2*configs.d_model, configs.activation),
        #     # torch.nn.Identity(),
        #     torch.nn.LayerNorm(configs.d_model),
        #     torch.nn.GELU(),
        #     MLP(configs.d_model, configs.d_model, 2*configs.d_model, configs.activation),
        #     # torch.nn.Identity(),
        #     torch.nn.LayerNorm(configs.d_model),
        #     torch.nn.GELU(),
        #     MLP(configs.d_model, configs.pred_len, 2*configs.d_model, configs.activation),
        # )
        # MLP(configs.seq_len, configs.pred_len, configs.d_model, configs.activation)
        self.use_norm = configs.use_norm
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # print(x_enc.shape)
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        # x_enc: B L N. transpose to B N L.
        x_enc = x_enc.transpose(-1, -2)
        y_enc = self.linear(x_enc)
        y_enc = y_enc.transpose(-1, -2)
        if self.use_norm:
            y_enc = y_enc * stdev[:,0,:].unsqueeze(1).repeat(1, self.pred_len, 1)
            y_enc = y_enc + means[:,0,:].unsqueeze(1).repeat(1, self.pred_len, 1)
        return y_enc
    def forward(self,x_enc,x_mark_enc,x_dec, x_mark_dec,mask=None):
        dec_out = self.forecast(x_enc,x_mark_enc,x_dec,x_mark_dec)
        return dec_out[:,-self.pred_len:,:]