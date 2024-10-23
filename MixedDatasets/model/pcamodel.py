import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, MLP
import numpy as np
# code from https://github.com/ts-kim/RevIN, with minor modifications

import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(1,1,self.num_features).float())
        self.affine_bias = nn.Parameter(torch.zeros(1,1,self.num_features).float())

    def _get_statistics(self, x):
        # x: B,N,C
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,:,-1].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class ResMLP(nn.Module):
    def __init__(self, in_and_out_dim, activation, layer_num, hidden_dim=None):
        super(ResMLP, self).__init__()
        self.mlp_layers = nn.ModuleList()
        WIDTH=4
        print("Current width:", WIDTH)
        hidden_dim = WIDTH*in_and_out_dim if hidden_dim is None else hidden_dim
        for _ in range(layer_num):
            self.mlp_layers.append(MLP(in_and_out_dim, in_and_out_dim, hidden_dim, activation))
        self.norm_layers = nn.ModuleList()
        for _ in range(layer_num - 1):
            self.norm_layers.append(torch.nn.LayerNorm(in_and_out_dim))
    def forward(self,x):
        for idx, mlp in enumerate(self.mlp_layers):
            y = mlp(x)
            x = x + y
            if idx < len(self.mlp_layers) - 1:
                x = self.norm_layers[idx](x)
                # x = F.gelu(x)
        return x
        

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        print('self.enc_embedding.num param:', sum(p.numel() for p in self.enc_embedding.parameters()))
        print("configs.d_model:", configs.d_model)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        
        num_layers = configs.e_layers
        encoder_list = []
        encoder_list.append(ResMLP(configs.d_model, configs.activation, num_layers))
        # for _ in range(num_layers):
        #     encoder_list.append(MLP(configs.d_model, configs.d_model, 2*configs.d_model, configs.activation))
        #     encoder_list.append(torch.nn.LayerNorm(configs.d_model)) if _ != num_layers-1 else None
        #     encoder_list.append(torch.nn.GELU()) if _ != num_layers-1 else None
        self.encoder = nn.Sequential(*encoder_list)
        print('self.encoder.num param:', sum(p.numel() for p in self.encoder.parameters()))
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
        
        projector_list = []
        # projector_list.append(ResMLP(configs.d_model, configs.activation, num_layers-1))
        projector_list.append(MLP(configs.d_model, configs.pred_len, 2*configs.d_model, configs.activation))
        # for _ in range(num_layers):
        #     projector_list.append(MLP(configs.d_model, configs.d_model if _ != num_layers-1 else configs.pred_len, 2*configs.d_model, configs.activation))
        #     projector_list.append(torch.nn.LayerNorm(configs.d_model)) if _ != num_layers-1 else None
        #     projector_list.append(torch.nn.GELU()) if _ != num_layers-1 else None
        # self.projector = torch.nn.Linear(configs.d_model, configs.pred_len)
        self.projector = nn.Sequential(*projector_list)
        print('self.projector.num param:', sum(p.numel() for p in self.projector.parameters()))
        
        # past_projector_list = []
        # past_projector_list.append(ResMLP(configs.d_model, configs.activation, num_layers-1))
        # past_projector_list.append(MLP(configs.d_model, configs.seq_len, 2*configs.d_model, configs.activation))
        # # for _ in range(num_layers):
        # #     past_projector_list.append(MLP(configs.d_model, configs.d_model if _ != num_layers-1 else configs.seq_len, 2*configs.d_model, configs.activation))
        # #     past_projector_list.append(torch.nn.LayerNorm(configs.d_model)) if _ != num_layers-1 else None
        # #     past_projector_list.append(torch.nn.GELU()) if _ != num_layers-1 else None
        # self.past_projector = nn.Sequential(*past_projector_list)
        
        # self.past_projector = self.projector
        
        # mask encoder: 1 is masked, 0 is unmasked
        # self.mask_encoder = torch.nn.Linear(configs.seq_len, configs.d_model, bias=False)
        # self.past_projector = MLP(configs.d_model, configs.seq_len, activation = configs.activation)
        # self.revin = RevIN(862)
        
        
    # forecast without further training
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # x_enc = self.revin(x_enc, 'norm')
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