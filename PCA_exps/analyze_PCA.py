import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm
from sklearn import decomposition

from argparse import ArgumentParser

from matplotlib import pyplot as plt

def obtain_topk_normalized_vectors_with_eigenvalues(input_traj, K):
    # input_traj: N*D torch.tensor
    # output: components: K*D torch.tensor, eigenvalues: K-element torch.tensor
    
    # Ensure input is a numpy array for PCA compatibility
    input_traj_np = input_traj.numpy()
    
    # Apply PCA without explicitly centering data
    pca = decomposition.PCA(n_components=K)
    pca.fit(input_traj_np)
    
    # Extract the top K components and normalize them
    components = pca.components_  # Shape will be (K, D)
    
    # Eigenvalues (variance explained by each PC) from the PCA
    eigenvalues = pca.explained_variance_  # Shape will be (K,)
    
    # Convert the components and eigenvalues back to a torch tensor
    output_components = torch.tensor(components, dtype=torch.float)
    output_eigenvalues = torch.tensor(eigenvalues, dtype=torch.float)
    
    return output_components, output_eigenvalues
def obtain_topk_normalized_vectors(input_traj, K):
    # input_traj: N*D torch.tensor
    # output: K*D torch.tensor
    
    # Ensure input is a numpy array for PCA compatibility
    input_traj_np = input_traj.numpy()
    
    # Apply PCA without explicitly centering data
    pca = decomposition.PCA(n_components=K)
    pca.fit(input_traj_np)
    
    # Extract the top K components and normalize them
    components = pca.components_  # Shape will be (K, D)
    
    # Convert the components back to a torch tensor
    output = torch.tensor(components, dtype=torch.float)
    
    return output

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input_tensor_dir', type=str, default='./PCA_exps/mid_results/')
    parser.add_argument('--output_position', type=str, default='./PCA_exps/mid_results.pth')
    parser.add_argument('--to_image', type=str, default='./PCA_exps_mid_results.png')
    parser.add_argument('--component_number', type=int, default=500)
    args = parser.parse_args()
    # read all .pth file in input_tensor_dir.
    pth_list = list()
    for root, dirs, files in os.walk(args.input_tensor_dir):
        for file in files:
            if file.endswith('.pth'):
                pth_list.append(torch.load(os.path.join(root, file)))
    pth = torch.cat(pth_list, dim=0)
    # merge the rest dimensions apart from dim 0.
    pth = pth.view(pth.size(0), -1)
    mean = pth.mean(dim=-1, keepdim=True)
    pth = pth - mean
    
    pcas, eig_vals = obtain_topk_normalized_vectors_with_eigenvalues(pth, args.component_number)
    torch.save(dict(
        mean = mean[:,0],
        pcas = pcas,
        eig_vals = eig_vals,
    ), args.output_position)
    
    eig_vals = eig_vals / eig_vals[0]
    # draw a graph, x: idx, y: eig_vals.
    plt.plot(eig_vals)
    # use log scale for x and y axis.
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(args.to_image)
    plt.close()