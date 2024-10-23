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
    alphas = torch.arange(0,P,device=y.device).float()
    for _ in range(len(y.shape)-1):
        alphas = alphas.unsqueeze(0)
    alphas = (x_start_devided_by_delta_T + alphas) * beta * 2 * np.pi / S
    # alphas: ... * P
    alphas = alphas.unsqueeze(-1) # ... * P * 1
    i_tensor = torch.arange(1,S+1,device=y.device).float()
    for _ in range(len(y.shape)):
        i_tensor = i_tensor.unsqueeze(0)
    # i_tensor: ... * 1 * S
    # print(alphas.shape)
    # print(i_tensor.shape)
    alpha_j_i = alphas * i_tensor.repeat(*[1]*(len(y.shape)-1),P,1)
    sin_square = torch.sin(alpha_j_i).pow(2)
    cos_square = torch.cos(alpha_j_i).pow(2)
    sincos = torch.sin(alpha_j_i) * torch.cos(alpha_j_i)
    cos = torch.cos(alpha_j_i)
    sin = torch.sin(alpha_j_i)
    sumj_cos_square = cos_square.sum(-2) # ... * S
    sumj_sin_square = sin_square.sum(-2) # ... * S
    sumj_sincos = sincos.sum(-2) # ... * S
    sumj_cos = cos.sum(-2) # ... * S
    sumj_sin = sin.sum(-2) # ... * S
    sumj_cosjiyj = (cos * y.unsqueeze(-1)).sum(-2) # ... * S
    sumj_sinjiyj = (sin * y.unsqueeze(-1)).sum(-2) # ... * S
    E = torch.zeros(*y.shape[:-1],2*S+1,2*S+1,device=y.device,dtype=y.dtype)
    
    
    eye_matrix = torch.eye(S,device=y.device,dtype=y.dtype)
    for _ in range(len(y.shape)-1):
        eye_matrix = eye_matrix.unsqueeze(0)
    sumj_cos_square = torch.einsum('...ij,...j->...ij',eye_matrix,sumj_cos_square)
    sumj_sincos = torch.einsum('...ij,...j->...ij',eye_matrix,sumj_sincos)
    sumj_sin_square = torch.einsum('...ij,...j->...ij',eye_matrix,sumj_sin_square) # ... * S * S
    E[..., 0:S,0:S] = sumj_cos_square
    E[..., 0:S,S:2*S] = sumj_sincos
    E[..., S:2*S,0:S] = sumj_sincos
    E[..., S:2*S,S:2*S] = sumj_sin_square
    E[..., 0:S,2*S] = sumj_cos
    E[..., S:2*S,2*S] = sumj_sin
    E[..., 2*S,0:S] = sumj_cos
    E[..., 2*S,S:2*S] = sumj_sin
    E[..., 2*S,2*S] = P
    # E: ... * 2S+1 * 2S+1
    target = torch.zeros(*y.shape[:-1],2*S+1,device=y.device,dtype=y.dtype)
    target[..., 0:S] = sumj_cosjiyj+1e-7
    target[..., S:2*S] = sumj_sinjiyj+1e-7
    target[..., 2*S] = y.sum(-1)
    E = E +  torch.eye(2*S+1,device=y.device,dtype=y.dtype)
    # print(E.shape)
    # print(target.shape)
    # target: ... * 2S+1
    A = torch.linalg.solve(E,target)
    # print(A.shape)
    return A[...,0:S],A[...,S:2*S],A[...,2*S:2*S+1]
y = torch.sin(2*np.pi/3.0*torch.arange(0,7).float())
S = 3
x_start_devided_by_delta_T = torch.tensor([0.0])
beta = torch.tensor([1.0])
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
for i in range(7):
    print(recovered_func(*(myFFT(y,beta,S,x_start_devided_by_delta_T)),float(i),beta))