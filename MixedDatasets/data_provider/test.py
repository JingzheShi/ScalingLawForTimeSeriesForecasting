import torch
import math
from matplotlib import pyplot as plt
def create_square_wave(num, total_length, NUM_LEVELS = 10, T_min = 5, T_max = 500):
    # Initialize the index tensor representing time steps.
    index = torch.arange(0, total_length)
    index = index.unsqueeze(0).unsqueeze(0).repeat(num, NUM_LEVELS, 1)  # (num,NUM_LEVELS, total_length)

    
    
    # Generate random periods for each wave within the specified range.
    T = torch.randint(low=T_min, high=T_max, size=(num, NUM_LEVELS))  # (num, NUM_LEVELS)
    T = T.unsqueeze(-1).repeat(1, 1, total_length)  # (num, NUM_LEVELS, total_length)
    bias = torch.rand(num, NUM_LEVELS).unsqueeze(-1).repeat(1, 1, total_length)  # (num, NUM_LEVELS, total_length)
    bias = (bias * (T.float())).long()  # (num, NUM_LEVELS, total_length)

    # Calculate the duty cycle (proportion of time the signal is high vs low).
    # This example assumes a 50% duty cycle for simplicity.
    duty_cycle = torch.rand(num, NUM_LEVELS)*0.925+0.05  # (num, NUM_LEVELS)
    duty_cycle = duty_cycle.unsqueeze(-1).repeat(1, 1, total_length)  # (num, NUM_LEVELS, total_length)
    # Initialize the square wave tensor.
    square_wave = torch.zeros(num, NUM_LEVELS, total_length)

    # Determine the high and low states of the square wave at each point in time.
    # for i in range(num):
    #     # Calculate the state (high or low) based on the current point in the period.
    #     square_wave[i] = torch.floor(((index[i]+bias[i]) % T[i]) / (T[i] * duty_cycle[i])) * 2 - 1  # Results in values of -1 or 1
    
    square_wave = torch.floor(((index+bias)%T)/(T*duty_cycle))*2-1
    
    weights = torch.rand(num,NUM_LEVELS) + 1e-4
    weights = weights / weights.sum(dim=1, keepdim=True)
    
    square_wave = (square_wave * weights.unsqueeze(2)).sum(dim=1) # (num, total_length)
    
    return square_wave
def create_sinosoid(num, total_length, NUM_FREQ = 3, T_min = 5, T_max = 500):
    index = torch.arange(0, total_length).float()
    index = index.unsqueeze(0).unsqueeze(0).repeat(num, NUM_FREQ, 1) # (num, 1, total_length)
    T = torch.rand(num, NUM_FREQ) * (T_max - T_min) + T_min # (num, NUM_FREQ)
    freq = 2 * math.pi / T # (num, NUM_FREQ)
    bias = torch.rand(num, NUM_FREQ) * 2 * math.pi # (num, NUM_FREQ)
    sinosoid = torch.sin(index * freq.unsqueeze(2) + bias.unsqueeze(2)) # (num, NUM_FREQ, total_length)
    weights = torch.rand(num,NUM_FREQ) + 1e-4
    weights = weights / weights.sum(dim=1, keepdim=True)
    sinosoid = (sinosoid * weights.unsqueeze(2)).sum(dim=1) # (num, total_length)
    return sinosoid
    

def add_noise(tensor, noise_std = 0.02):
    noise = torch.randn_like(tensor) * noise_std
    return tensor + noise


    
FUNC_LIST = [create_sinosoid, create_square_wave]

    
def create_sequence(num, total_length, TYPE=2, noise_std=0.002):
    # returns a tensor of shape (num, total_length)
    weights = torch.rand(num, TYPE) + 1e-4
    weights = torch.nn.functional.softmax(weights*3.5, dim=1)
    seq = torch.zeros(num, TYPE, total_length).float()
    for idx in range(TYPE):
        seq[:,idx] = FUNC_LIST[idx](num, total_length)
    seq = (seq * weights.unsqueeze(2)).sum(dim=1)
    seq = add_noise(seq, noise_std)
    return seq
my_wave = create_sequence(5,336+96)
for N in range(5):
    plt.plot(my_wave[N])
    plt.savefig(f'wave_{N}.png') if N == 4 else 0