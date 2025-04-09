from config import setup_MCconfig
from main_mc import main_mc
import matplotlib.pyplot as plt
import numpy as np

def run_no_launch():
    seeds = [1, 2, 3]
    deorbit_matrix = np.zeros((3, 730))
    for i, seed in enumerate(seeds):
        print(f"\nRunning simulation for seed {seed}")
        cfgMC = setup_MCconfig(seed, ICfile='2020.mat')
        nS, nD, nN, nB, deorbit_list = main_mc(cfgMC, seed)
        deorbit_matrix[i, :] = deorbit_list

    # Plotting
    years = np.linspace(2020, 2030, 730)
    plt.plot(years, deorbit_matrix[0], label='Seed 1')
    plt.plot(years, deorbit_matrix[1], label='Seed 2')
    plt.plot(years, deorbit_matrix[2], label='Seed 3')
    plt.title("Population Evolution")
    plt.xlabel("Time (Year)")
    plt.ylabel("Decayed Objects")
    plt.legend()
    plt.xlim([2020, 2030])
    plt.grid()
    plt.show()

if __name__ == "__main__":
    run_no_launch()
