import torch
import numpy as np
def myFFT(y,beta,S,x_start_devided_by_delta_T):
    '''
        y: sequence. shape: ...*P
        x_start_devided_by_delta_T: start time of the sequence devided by delta_T. shape: ...
        beta: \delta T/delta t. shape: ... 
            \delta T be the time interval of y and 1/S\deltat be the base frequency of the desired fft.
        S: number of frequency bins. 
        
        returns:
            A, B, C that minimizes the following loss:
            \sum_{j=1}^{P} (\sum_{i=1}^{S} y_j - A_i \cos(\alpha_j*i) - B_i \sin(\alpha_j*i) - C_i)^2
            where \alpha_j = 2\pi/S/delta t * x_start + 2\pi\delt T / \delta t / S * j
    '''
    P = y.shape[-1]
    x_start_devided_by_delta_T = x_start_devided_by_delta_T.unsqueeze(-1)
    # x_start_devided_by_delta_T: ... * 1
    beta = beta.unsqueeze(-1)
    alphas = torch.arange(0,P,device=y.device).double()
    for _ in range(len(y.shape)-1):
        alphas = alphas.unsqueeze(0)
    alphas = (x_start_devided_by_delta_T + alphas) * beta * 2 * np.pi / S
    # alphas: ... * P
    alphas = alphas.unsqueeze(-1) # ... * P * 1
    i_tensor = torch.arange(1,S+1,device=y.device).double()
    for _ in range(len(y.shape)):
        i_tensor = i_tensor.unsqueeze(0)
    # i_tensor: ... * 1 * S
    # print(alphas.shape)
    # print(i_tensor.shape)
    alpha_j_i = alphas * i_tensor.repeat(*[1]*(len(y.shape)-1),P,1)
    
    cosalpha_j_i = torch.cos(alpha_j_i)
    sinalpha_j_i = torch.sin(alpha_j_i)
    
    cosalpha_jk_cosalpha_ji = torch.einsum('...jk,...ji->...ki',cosalpha_j_i,cosalpha_j_i)
    cosalpha_jk_sinalpha_ji = torch.einsum('...jk,...ji->...ki',cosalpha_j_i,sinalpha_j_i)
    sumj_cosalpha_jk = cosalpha_j_i.sum(-2) # ... * S
    sumj_yj_cosalpha_jk = torch.einsum('...j,...jk->...k',y,cosalpha_j_i) # ... * S
    
    
    sinalpha_jk_cosalpha_ji = torch.einsum('...jk,...ji->...ki',sinalpha_j_i,cosalpha_j_i)
    sinalpha_jk_sinalpha_ji = torch.einsum('...jk,...ji->...ki',sinalpha_j_i,sinalpha_j_i)
    sumj_sinalpha_jk = sinalpha_j_i.sum(-2) # ... * S
    sumj_yj_sinalpha_jk = torch.einsum('...j,...jk->...k',y,sinalpha_j_i) # ... * S
    
    sumj_cos = cosalpha_j_i.sum(-2) # ... * S
    sumj_sin = sinalpha_j_i.sum(-2) # ... * S
    sumj = y.shape[-1]
    sumj_yj = y.sum(-1) # ...
    
    
    
    E = torch.zeros(*y.shape[:-1],2*S+1,2*S+1,device=y.device,dtype=y.dtype)
    
    E[..., 0:S,0:S] = cosalpha_jk_cosalpha_ji
    E[..., 0:S,S:2*S] = cosalpha_jk_sinalpha_ji
    E[..., S:2*S,0:S] = sinalpha_jk_cosalpha_ji
    E[..., S:2*S,S:2*S] = sinalpha_jk_sinalpha_ji
    E[..., 0:S,2*S] = sumj_cosalpha_jk
    E[..., S:2*S,2*S] = sumj_sinalpha_jk
    
    E[..., 2*S,0:S] = sumj_cos
    E[..., 2*S,S:2*S] = sumj_sin
    E[..., 2*S,2*S] = sumj
    
    
    target = torch.zeros(*y.shape[:-1],2*S+1,device=y.device,dtype=y.dtype)
    target[..., 0:S] = sumj_yj_cosalpha_jk
    target[..., S:2*S] = sumj_yj_sinalpha_jk
    target[..., 2*S] = sumj_yj
    
    
    E = E + torch.randn_like(E)*0.0
    
    A = torch.linalg.solve(E,target)
    # print(A.shape)
    return A[...,0:S],A[...,S:2*S],A[...,2*S:2*S+1]
y = torch.arange(-12,12).double()
y = y + 0.0 * torch.randn_like(y)
S = 12
x_start_devided_by_delta_T = torch.tensor([0.0])
beta = torch.tensor([1.02])
for item in myFFT(y,beta,S,x_start_devided_by_delta_T):
    print(item)
def recovered_func(A,B,C,x,beta):
    # print(A.shape)
    # print(B.shape)
    # print(C.shape)
    # print(beta.shape)
    '''
        A: shape: ... * S
        B: shape: ... * S
        C: shape: ... * 1
        x: shape: ...
    '''
    answer = C
    for i in range(1,A.shape[-1]):
        answer += A[i-1]*torch.cos(2*np.pi*beta/S*i*x) + B[i-1]*torch.sin(2*np.pi*beta/S*i*x)
    return answer
print(y)
result_list = []
for i in range(24):
    result_list.append(recovered_func(*(myFFT(y,beta,S,x_start_devided_by_delta_T)),float(i),beta))
result_tensor = torch.cat(result_list,dim=0)
print(result_tensor)