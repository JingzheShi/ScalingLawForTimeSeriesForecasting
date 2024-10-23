import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, MLP
import numpy as np


        

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = 192
        self.label_len = configs.label_len
        self.pred_len = 144
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        if not configs.linear_embedding:
            self.enc_embedding = DataEmbedding_inverted(self.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_inverted_small(self.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        
        encoder_list = []
        for _ in range(configs.e_layers):
            encoder_list.append(MLP(configs.d_model, configs.d_model, 2*configs.d_model, configs.activation))
            encoder_list.append(torch.nn.LayerNorm(configs.d_model)) if _ != configs.e_layers-1 else None
            encoder_list.append(torch.nn.GELU()) if _ != configs.e_layers-1 else None
        self.encoder = nn.Sequential(*encoder_list)
        # self.encoder = Encoder(
        #     [
        #         EncoderLayer(
        #             AttentionLayer(
        #                 FullAttention(False, configs.factor, attention_dropout=configs.dropout,
        #                               output_attention=configs.output_attention), configs.d_model, configs.n_heads),
        #             configs.d_model,
        #             configs.d_ff,
        #             dropout=configs.dropout,
        #             activation=configs.activation
        #         ) for l in range(configs.e_layers)
        #     ],
        #     norm_layer=torch.nn.LayerNorm(configs.d_model)
        # )
        
        # self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        # projector: feature -> pred 
        # assert configs.seq_len == configs.pred_len
        self.projector = MLP(configs.d_model, self.pred_len, activation = configs.activation)
        # self.past_projector = self.projector
        
        # mask encoder: 1 is masked, 0 is unmasked
        self.mask_encoder = torch.nn.Linear(self.seq_len, configs.d_model, bias=False)
        self.past_projector = MLP(configs.d_model, self.seq_len, activation = configs.activation)
        
        
        
    # forecast without further training
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape # B L N
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
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forecast_and_retrieve(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_enc_mask):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

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