import numpy as np
from datetime import datetime
from main_mc import run_batch_mc, plot_batch_results
from utils import getidx

# Get indices for satellite data
idx = getidx()

# Build an example satellite
example_sat = np.zeros((1, 40))
example_sat[0, idx['a']] = 6771.0
example_sat[0, idx['ecco']] = 0.001
example_sat[0, idx['inclo']] = 51.6
example_sat[0, idx['nodeo']] = 0.0
example_sat[0, idx['argpo']] = 0.0
example_sat[0, idx['mo']] = 0.0
example_sat[0, idx['mass']] = 500
example_sat[0, idx['radius']] = 1.0
example_sat[0, idx['objectclass']] = 1  # operational
example_sat[0, idx['controlled']] = 1
example_sat[0, idx['ID']] = 1

# Simulation config
n_steps = 100
dt_days = 73
tsince = np.arange(n_steps) * dt_days * 1440  # minutes

cfgMC = {
    'mat_sats': example_sat,
    'tsince': tsince,
    'n_time': n_steps,
    'radiusearthkm': 6371.0,
    'mu': 398600.4418,
    'time0': datetime(2025, 1, 1),
    'dt_days': dt_days,
    'DAY2MIN': 1440,
    'alph': 1e-6,
    'alph_a': 1e-6,
    'PMD': 0.9,
    'launch_model': 'no_launch',
    'output_file': 'test_no_launch'
}

# Run the simulation
results = run_batch_mc(cfgMC, num_runs=1)
plot_batch_results(results, cfgMC['dt_days'])
