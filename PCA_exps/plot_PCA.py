import os
import torch
from matplotlib import pyplot as plt
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--result_pth', type=str, default='./PCA_exps/org_tensor.pth')
    parser.add_argument('--to_image', type=str, default='./PCA_exps_org_tensor.png')
    args = parser.parse_args()
    result = torch.load(args.result_pth)
    eig_vals = result['eig_vals']
    eig_vals = eig_vals / eig_vals[0]
    # draw a graph, x: idx, y: eig_vals.
    x = torch.tensor(list(range(eig_vals.shape[0])))+1
    plt.plot(x, eig_vals)
    # use log scale for x and y axis.
    print(eig_vals[:5])
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(args.to_image)
    plt.close()