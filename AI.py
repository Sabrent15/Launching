import numpy as np
import scipy.io
import os
import h5py
import matplotlib.pyplot as plt

# Function to load .mat files (handles both v7.3 and older formats)
def load_mat_file(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Error: '{filename}' not found.")
    
    try:
        # Attempt to load using SciPy (supports v4, v6, v7, v7.1)
        data = scipy.io.loadmat(filename)
        print("File loaded successfully using SciPy!")
        return data
    except NotImplementedError:
        print("SciPy could not read the file. Trying h5py (likely v7.3 format)...")
    
    try:
        # Load v7.3 (HDF5 format) using h5py
        with h5py.File(filename, 'r') as file:
            print("File loaded successfully using h5py!")
            return {key: file[key][()] for key in file.keys()}
    except Exception as e:
        raise ValueError(f"Error loading '{filename}': {e}")

# Initializes Monte Carlo simulation
def initSim(cfgMC, simulation, launch_model, ICfile):
    """
    Configures the initial satellite population and launch models.
    """
    print("Initializing simulation...")
    ic_data = load_mat_file(ICfile)

    # Extract initial conditions
    mat_sats = ic_data.get("mat_sats")
    time0 = ic_data.get("time0_num", None)  # Use numeric timestamp if available

    if mat_sats is None:
        raise ValueError("Missing 'mat_sats' in .mat file.")

    if time0 is None:
        print("Warning: 'time0' is missing or in an unsupported format. Proceeding without it.")

    print(f"Loaded {mat_sats.shape[0]} satellite entries from {ICfile}")

    # Apply altitude filtering
    radiusearthkm = cfgMC["radiusearthkm"]
    a_all = mat_sats[:, 0] * radiusearthkm
    e_all = mat_sats[:, 1]
    ap_all = a_all * (1 - e_all)
    
    # Filter satellites within altitude range
    mask = (ap_all > (cfgMC["altitude_limit_low"] + radiusearthkm)) & \
           (ap_all < (cfgMC["altitude_limit_up"] + radiusearthkm))
    mat_sats = mat_sats[mask]

    # Launch model handling
    if launch_model == "no_launch":
        repeatLaunches = []
    elif launch_model in ["random", "matsat"]:
        repeatLaunches = mat_sats.copy()
    else:
        raise ValueError(f"Unknown launch model: {launch_model}")

    cfgMC["mat_sats"] = mat_sats
    cfgMC["repeatLaunches"] = repeatLaunches
    cfgMC["time0"] = time0
    return cfgMC

# Function to set up the Monte Carlo configuration
def setup_MCconfig(seed, ICfile):
    """
    Sets up the configuration for the Monte Carlo simulation.
    """
    # Example configuration dictionary
    cfgMC = {
        "radiusearthkm": 6371,  # Earth radius in km
        "altitude_limit_low": 300,  # Minimum altitude in km
        "altitude_limit_up": 2000,  # Maximum altitude in km
        "seed": seed,
    }

    # Load initial conditions and update the configuration
    cfgMC = initSim(cfgMC, "random", "matsat", ICfile)
    
    return cfgMC

# Placeholder for main_mc function (replace with actual logic)
def main_mc(cfgMC, seed):
    """
    Simulates the Monte Carlo process for a given configuration and seed.
    """
    # Example: model the decay and launch process
    num_days = 730  # simulate for 2 years (730 days)
    
    # Start with initial population size (from cfgMC)
    initial_population = cfgMC["mat_sats"].shape[0]
    
    # Simulate growth over time (simplified exponential growth model)
    growth_rate = 0.01  # 1% increase in satellites per day
    population = initial_population * (1 + growth_rate) ** np.arange(num_days)
    
    # Apply noise reduction: use a moving average to smooth the data
    deorbitlist_r = np.convolve(population, np.ones(30)/30, mode='valid')  # 30-day moving average
    
    # Ensure the size matches the original array (730 elements)
    padded_deorbitlist_r = np.pad(deorbitlist_r, (0, 730 - deorbitlist_r.size), mode='constant', constant_values=deorbitlist_r[-1])

    return None, None, None, None, padded_deorbitlist_r  # Return padded deorbit list

# Main execution function
def main():
    ICfile = '2020_fixed.mat'  # Ensure this file exists with time0_num instead of time0

    # Array to store deorbit lists for 3 seeds
    deorbit_list_m = np.zeros((3, 730))

    for idx in range(3):
        seed = idx + 1
        cfgMC = setup_MCconfig(seed, ICfile)
        print(f"Seed {seed} - Initial Population: {cfgMC['mat_sats'].shape[0]} sats")
        
        _, _, _, _, deorbitlist_r = main_mc(cfgMC, seed)
        
        # Assign the padded deorbit list
        deorbit_list_m[idx, :] = deorbitlist_r

    # Plot results
    time_axis = np.linspace(2020, 2030, 730)

    plt.figure()
    for i in range(3):
        plt.plot(time_axis, deorbit_list_m[i, :], linewidth=2, label=f"Seed {i+1}")

    plt.legend()
    plt.xlabel("Time (Year)")
    plt.ylabel("Population Size")
    plt.title("Population Growth Over Time")
    plt.xlim([2020, 2030])
    plt.show()

if __name__ == "__main__":
    main()
