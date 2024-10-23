import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, MLP, DataEmbedding_inverted_small
import numpy as np
class inInterpolate(nn.Module):
    def __init__(self,seq_len,interpolate_len):
        super().__init__()
        self.seq_len = seq_len
        self.interpolate_len = interpolate_len
    def forward(self,x):
        x = x.transpose(1,2)
        x = F.interpolate(x, size=self.interpolate_len, mode='linear', align_corners=True)
        # interpolate in the last dimension.
        # x = x.transpose(1,2)
        return x
    
class Backbone(nn.Module):
    def __init__(self,seq_len,pred_len,configs):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_layer = configs.e_layers
        self.d_model = configs.d_model
        in_dim = self.seq_len
        current_dim = self.d_model
        self.linear_layers = nn.ModuleList()
        for i in range(self.num_layer):
            self.linear_layers.append(nn.Linear(in_dim,current_dim))
            in_dim = current_dim
        self.normactlayers = nn.ModuleList()
        for i in range(self.num_layer):
            self.normactlayers.append(nn.Sequential(nn.GELU(),nn.LayerNorm(current_dim)))
        self.final_linear = nn.Linear(current_dim,self.pred_len)
    
    def forward(self,x):
        
        for idx, (linear, normact) in enumerate(zip(self.linear_layers,self.normactlayers)):
            y = linear(x)
            y = normact(y)
            if idx > 0:
                x = x + y
            else:
                x = y
        
        x = self.final_linear(x)
        
        return x
            
    

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.in_patch = inInterpolate(configs.seq_len,configs.interpolate_len)
        print("Only in patch stride and in patch size works. out patch stride and out patch size are not working here.")
        self.seq_len = configs.interpolate_len
        self.label_len = configs.interpolate_len
        self.pred_len = configs.pred_len
        self.backbone = Backbone(self.seq_len, self.pred_len, configs)
        
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # x_enc = self.revin(x_enc, 'norm')
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-1)
            x_enc /= stdev
        x_enc = self.in_patch(x_enc)
        # print("!!!,x_enc.shape",x_enc.shape)
            

        y = self.backbone(x_enc)
        
        dec_out = y.transpose(1,2)
        
        if self.use_norm:
            # dec_out = self.revin(dec_out, 'denorm')
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out
        
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out = self.encoder(enc_out)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # dec_out = self.revin(dec_out, 'denorm')
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forecast_and_retrieve(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_enc_mask):
        assert False
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        mask_enc_out = self.mask_encoder(x_enc_mask.float().permute(0,2,1))
        mask_enc_out = mask_enc_out - (1 - 1e-6) * mask_enc_out.mean(dim=-1, keepdim=True)
        
        if x_mark_enc is not None:
            enc_out = torch.cat([enc_out[:,:N,:]+mask_enc_out,enc_out[:,N:,:]],dim=-2) # only add mask embedding to the non-covariate tokens.
        else:
            enc_out = enc_out + mask_enc_out
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out = self.encoder(enc_out)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates
        
        # B N E -> B N L -> B L N
        past_retrieved = self.past_projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates
        
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            
            past_retrieved = past_retrieved * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
            past_retrieved = past_retrieved + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
            
        return dec_out, past_retrieved
        



    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]