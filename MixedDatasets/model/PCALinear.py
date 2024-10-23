import math
import torch
import torch.nn as nn
class Gated_MLP(nn.Module):
    def __init__(self,in_dim,out_dim,hid_dim=None,activation='gelu'):
        super().__init__()
        self.linear_input = nn.Linear(in_dim,hid_dim)
        self.linear_gate = nn.Linear(in_dim,hid_dim)
        self.linear_out = nn.Linear(hid_dim,out_dim)
        self.gated_activation = nn.Sigmoid()
        self.drop_out = nn.Dropout(0.1)
    def forward(self,x):
        input = self.linear_input(x)
        gate = self.gated_activation(self.linear_gate(x))
        x = input * gate
        x = self.drop_out(x)
        x = self.linear_out(x)
        return x
class Model(torch.nn.Module):
    def __init__(self,args):
        super(Model, self).__init__()
        self.args = args
        N1,N2 = 30,30
        self.N1 = N1
        self.N2 = N2
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.linear = Gated_MLP(N1, N2,2*N2)
        mean_and_pca_input_dict = torch.load('/root/autodl-tmp/pcas_for_weather_336+0.pt')
        mean_and_pca_ioput_dict = torch.load('/root/autodl-tmp/pcas_for_weather_336+192.pt')
        self.input_mean = mean_and_pca_input_dict['mean'].unsqueeze(0) # 1, seq_len
        print("self.input_mean**2.mean==",torch.mean(self.input_mean**2))
        self.input_pca = mean_and_pca_input_dict['pcas'][:N1] # N1, seq_len
        self.ioput_mean = mean_and_pca_ioput_dict['mean'].unsqueeze(0) # 1, seq_len + pred_len
        print("self.ioput_mean**2.mean==",torch.mean(self.ioput_mean**2))
        # assert False
        self.ioput_pca = mean_and_pca_ioput_dict['pcas'][:N2] # N2, seq_len + pred_len
    def to_devices(self,tensor):
        if self.input_mean.device != tensor.device:
            self.input_mean = self.input_mean.to(tensor.device)
            self.input_pca = self.input_pca.to(tensor.device)
            self.ioput_mean = self.ioput_mean.to(tensor.device)
            self.ioput_pca = self.ioput_pca.to(tensor.device)
        
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        self.to_devices(x_enc)
        x_enc = x_enc[...,0] # B, T
        
        assert self.args.use_norm
        means = x_enc.mean(1,keepdim=True).detach()
        x_enc = x_enc - means
        stdenv = x_enc.std(1,keepdim=True,unbiased=False) + 1e-1
        x_enc = x_enc / stdenv - self.input_mean
        B, T = x_enc.shape
        zeros = torch.zeros([B,self.pred_len],dtype=x_enc.dtype,device=x_enc.device)
        padded_x_enc = torch.cat([x_enc,zeros],dim=1)
        projected_x_enc = torch.einsum('bt,it->bi',x_enc,self.input_pca) # t=seq_len,i=N1
        
        predicted_ioput = self.linear(projected_x_enc)
        
        projected_ioput = torch.einsum('bj,jT->bT',predicted_ioput,self.ioput_pca) # T=seq_len + pred_len, j=N2
        
        predicted_ioput = projected_ioput + self.ioput_mean
        
        predicted_ioput = predicted_ioput * stdenv[...,0:1] + means[...,0:1]
        
        return predicted_ioput[...,-self.pred_len:].unsqueeze(-1)
    
    def predict_and_retrieve(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        self.to_devices(x_enc)
        # print("x_enc.shape==",x_enc.shape)
        x_enc = x_enc[...,0] # B, T
        
        assert self.args.use_norm
        means = x_enc.mean(1,keepdim=True).detach()
        x_enc = x_enc - means
        stdenv = x_enc.std(1,keepdim=True,unbiased=False) + 1e-1
        x_enc = x_enc / stdenv
        # print("Current x_enc**2.mean==",torch.mean(x_enc**2))
        x_enc = x_enc  - self.ioput_mean[:,:self.seq_len]
        # print("After subtraction,  x_enc**2.mean==",torch.mean(x_enc**2))
        B, T = x_enc.shape
        zeros = torch.zeros([B,self.pred_len],dtype=x_enc.dtype,device=x_enc.device)
        # print("x_enc.shape==",x_enc.shape)
        # print("zeros.shape==",zeros.shape)
        padded_x_enc = torch.cat([x_enc,zeros],dim=1)
        
        
        # print("torch.sum(self.ioput_pca**2,dim=1).mean()==",torch.sum(self.ioput_pca**2,dim=1).mean())
        projected_x_enc = torch.einsum('bt,it->bi',x_enc,self.input_pca)
        # projected_x_enc = torch.einsum('bt,it->bi',padded_x_enc,self.ioput_pca) # t=seq_len,i=N1
        # projected_x_enc = projected_x_enc / math.sqrt(self.N1)
        # print("projected_x_enc**2.mean==",torch.mean(projected_x_enc**2))
        predicted_ioput = self.linear(projected_x_enc)
        
        projected_ioput = torch.einsum('bj,jT->bT',predicted_ioput,self.ioput_pca) # T=seq_len + pred_len, j=N2
        
        # print("First projected_ioput**2.mean==",torch.mean(projected_ioput**2))
        predicted_ioput = projected_ioput + self.ioput_mean
        # print("predicted_ioput**2.mean==",torch.mean(predicted_ioput**2))
        
        predicted_ioput = predicted_ioput * stdenv[...,0:1] + means[...,0:1]
        
        return predicted_ioput.unsqueeze(-1)

    def forward(self,*args,**kwargs):
        return self.forecast(*args,**kwargs)

    
        
        
        