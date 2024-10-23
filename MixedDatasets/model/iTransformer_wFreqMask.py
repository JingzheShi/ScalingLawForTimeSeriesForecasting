import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, MLP
import numpy as np

class Emb(nn.Module):
    def __init__(self, configs, indicating_dict):
        super().__init__()
        self.idx_indicating_dict = dict()
        idx = 0
        self.embed_list = []
        for label in indicating_dict:
            self.idx_indicating_dict[label] = idx
            idx += 1
            self.embed_list.append(torch.nn.Parameter(torch.zeros([1, indicating_dict[label], configs.d_model]).float(), requires_grad = True))
        self.embed_list = torch.nn.ParameterList(self.embed_list)
        weights = []
        for _ in range(128): # only 64 dims are used for the embedding
            weights.append(1.0)
        for _ in range(512-128):
            weights.append(0.0)
        self.weights = torch.nn.Parameter(torch.tensor(weights).unsqueeze(0).unsqueeze(0).float(), requires_grad=False)
    def forward(self, x, x_label):
        assert x_label in self.idx_indicating_dict.keys()
        param = self.embed_list[self.idx_indicating_dict[x_label]]
        if len(x.shape) == 3:
            return x + 0.2*param*self.weights
        elif len(x.shape) == 2:
            return x + 0.2*param[0]*self.weights[0]
        else:
            raise ValueError("The input shape is not supported.")
        
        

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        
        # self.var_embed_inpt = Emb(configs, {"25":25,"866":866})
        # self.var_embed_otpt = Emb(configs, {"25":25,"866":866})
        
        
        
        # Embedding
        
        
        self.cut_freq = configs.cut_freq if configs.cut_freq is not None else configs.seq_len // 2
        
        self.enc_embedding = DataEmbedding_inverted(2*self.cut_freq, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        # projector: feature -> pred 
        
        # self.projector = torch.nn.Sequential(
        #         Encoder(
        #         [
        #             EncoderLayer(
        #                 AttentionLayer(
        #                     FullAttention(False, configs.factor, attention_dropout=configs.dropout,
        #                                 output_attention=configs.output_attention), configs.d_model, configs.n_heads),
        #                 configs.d_model,
        #                 configs.d_ff,
        #                 dropout=configs.dropout,
        #                 activation=configs.activation
        #             ) for l in range(configs.e_layers//2)
        #         ],
        #         norm_layer=torch.nn.LayerNorm(configs.d_model),
        #         return_attns=False,
        #     ),
        #     MLP(configs.d_model, 2*self.cut_freq, activation = configs.activation),
        # )
        
        # self.past_projector = torch.nn.Sequential(
        #         Encoder(
        #         [
        #             EncoderLayer(
        #                 AttentionLayer(
        #                     FullAttention(False, configs.factor, attention_dropout=configs.dropout,
        #                                 output_attention=configs.output_attention), configs.d_model, configs.n_heads),
        #                 configs.d_model,
        #                 configs.d_ff,
        #                 dropout=configs.dropout,
        #                 activation=configs.activation
        #             ) for l in range(configs.e_layers//2)
        #         ],
        #         norm_layer=torch.nn.LayerNorm(configs.d_model),
        #         return_attns=False,
        #     ),
        #     MLP(configs.d_model, 2*self.cut_freq, activation = configs.activation),
        # )
        
        
        # self.projector = torch.nn.Sequential(
        #     MLP(configs.d_model, configs.d_model, activation = configs.activation),
        #     MLP(configs.d_model, 2*self.cut_freq, activation = configs.activation),
        # )
        self.projector = MLP(configs.d_model, 2*self.cut_freq, activation = configs.activation)
        
        # self.past_projector = torch.nn.Sequential(
        #     MLP(configs.d_model, configs.d_model, activation = configs.activation),
        #     MLP(configs.d_model, 2*self.cut_freq, activation = configs.activation),
        # )
        
        self.past_projector = MLP(configs.d_model, 2*self.cut_freq, activation = configs.activation)
        
        
        # mask encoder: 1 is masked, 0 is unmasked
        self.mask_encoder = torch.nn.Linear(2*self.cut_freq, configs.d_model, bias=False)
        
        
        
        
    # forecast without further training
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        assert self.use_norm
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        _, _, N = x_enc.shape
        
        if x_mark_enc is not None:
            x_enc = torch.cat([x_enc, x_mark_enc], dim = -1)
            
        low_specx = torch.fft.rfft(x_enc, dim = 1, norm = 'ortho')
        orig_len = low_specx.shape[1]
        low_specx = torch.view_as_real(low_specx[:,0:self.cut_freq,:])
        low_specx_real = low_specx[:,:,:,0] # B cut_freq N
        low_specx_imag = low_specx[:,:,:,1] # B cut_freq N
        
        
        x_enc = torch.cat([low_specx_real, low_specx_imag], dim = -2)
        
        
        enc_out = self.enc_embedding(x_enc, None) 
        # enc_out = self.var_embed_inpt(enc_out, str(enc_out.shape[-2]))
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # enc_out = self.var_embed_otpt(enc_out, str(enc_out.shape[-2]))

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        dec_out_freq_real = dec_out[:,:dec_out.shape[1]//2,:]
        dec_out_freq_imag = dec_out[:,dec_out.shape[1]//2:,:]
        
        low_specxy_R = torch.zeros([dec_out.shape[0], self.pred_len//2, dec_out.shape[2]], device = dec_out.device).float()
        low_specxy_I = torch.zeros([dec_out.shape[0], self.pred_len//2, dec_out.shape[2]], device = dec_out.device).float()
        
        
        low_specxy_R[:,:dec_out.shape[1]//2,:] = dec_out_freq_real
        low_specxy_I[:,:dec_out.shape[1]//2,:] = dec_out_freq_imag
        
        low_specxy = torch.complex(low_specxy_R, low_specxy_I)
        dec_out = torch.fft.irfft(low_specxy,dim = 1, n = self.pred_len, norm = 'ortho')
        
        
        
        assert self.use_norm
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def retrieve(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_enc_mask):
        
        # x_enc_mask  B 2*cut_freq N, 1 is unmasked, 0 is masked
        
        
        assert self.use_norm
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev # B L N
        _, _, N = x_enc.shape # B L N
        
        if x_mark_enc is not None:
            x_enc = torch.cat([x_enc, x_mark_enc], dim = -1)
        
        # seq space to freq space
        
        low_specx = torch.fft.rfft(x_enc, dim = 1, norm = 'ortho')
        low_specx = torch.view_as_real(low_specx[:,0:self.cut_freq,:])
        low_specx_real = low_specx[:,:,:,0] # B cut_freq N
        low_specx_imag = low_specx[:,:,:,1] # B cut_freq N
        low_specx = torch.cat([low_specx_real,low_specx_imag],dim=1)
        
        
        x_enc = torch.cat([low_specx_real, low_specx_imag], dim = -2)
        
        
        # print(x_enc.shape)
        # print(low_specx.shape)
        # print(x_enc_mask.shape)
        x_enc[:,:,:N] = x_enc[:,:,:N] * x_enc_mask

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, None) # covariates (e.g timestamp) can be also embedded as tokens
        mask_enc_out = self.mask_encoder(x_enc_mask.float().permute(0,2,1))
        mask_enc_out = mask_enc_out - (1 - 1e-6) * mask_enc_out.mean(dim=-1, keepdim=True)
        
        if x_mark_enc is not None:
            enc_out = torch.cat([enc_out[:,:N,:]+mask_enc_out,enc_out[:,N:,:]],dim=-2) # only add mask embedding to the non-covariate tokens.
        else:
            enc_out = enc_out + mask_enc_out
            
        # enc_out = self.var_embed_inpt(enc_out, str(enc_out.shape[-2]))
        
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # enc_out = self.var_embed_otpt(enc_out, str(enc_out.shape[-2]))

        # B N E -> B N S -> B S N 
        # dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates
        # we do not need it here since we are doing retrieval.
        
        
        
        # B N E -> B N L -> B L N
        past_retrieved = self.past_projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates
        
        return low_specx[:, :, :N], past_retrieved
            # return target_in_freq_space, retrieved_in_freq_space
        
        
        
        
        
        
        
        
        
        
        
        
        assert self.use_norm
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        past_retrieved = past_retrieved * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        past_retrieved = past_retrieved + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
            
        return dec_out, past_retrieved
        
    def forecast_when_training(self, x_enc, x_mark_enc, x_dec, x_mark_dec, y_gt):
        assert self.use_norm
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        _, _, N = x_enc.shape
        
        if x_mark_enc is not None:
            x_enc = torch.cat([x_enc, x_mark_enc], dim = -1)
            
        low_specx = torch.fft.rfft(x_enc, dim = 1, norm = 'ortho')
        orig_len = low_specx.shape[1]
        low_specx = torch.view_as_real(low_specx[:,0:self.cut_freq,:])
        low_specx_real = low_specx[:,:,:,0] # B cut_freq N
        low_specx_imag = low_specx[:,:,:,1] # B cut_freq N
        
        
        x_enc = torch.cat([low_specx_real, low_specx_imag], dim = -2)
        
        
        enc_out = self.enc_embedding(x_enc, None) 
        # enc_out = self.var_embed_inpt(enc_out, str(enc_out.shape[-2]))
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # enc_out = self.var_embed_otpt(enc_out, str(enc_out.shape[-2]))

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates
        
        y_gt = y_gt - (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        y_gt = y_gt / (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        y_gt_freq = torch.fft.rfft(y_gt, dim = 1, norm = 'ortho')
        y_gt_freq = torch.view_as_real(y_gt_freq[:,0:self.cut_freq,:])
        y_gt_freq_real = y_gt_freq[:,:dec_out.shape[1]//2,:,0] # B cut_freq N
        y_gt_freq_imag = y_gt_freq[:,:dec_out.shape[1]//2,:,1] # B cut_freq N
        y_gt_freq = torch.cat([y_gt_freq_real, y_gt_freq_imag], dim = -2)
        return dec_out, y_gt_freq
        
        
        
        
        
        
        

        dec_out_freq_real = dec_out[:,:dec_out.shape[1]//2,:]
        dec_out_freq_imag = dec_out[:,dec_out.shape[1]//2:,:]
        
        low_specxy_R = torch.zeros([dec_out.shape[0], self.pred_len//2, dec_out.shape[2]], device = dec_out.device).float()
        low_specxy_I = torch.zeros([dec_out.shape[0], self.pred_len//2, dec_out.shape[2]], device = dec_out.device).float()
        
        
        low_specxy_R[:,:dec_out.shape[1]//2,:] = dec_out_freq_real
        low_specxy_I[:,:dec_out.shape[1]//2,:] = dec_out_freq_imag
        
        low_specxy = torch.complex(low_specxy_R, low_specxy_I)
        dec_out = torch.fft.irfft(low_specxy,dim = 1, n = self.pred_len, norm = 'ortho')
        
        
        
        assert self.use_norm
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]