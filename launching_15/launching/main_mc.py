# Full simulation core with stats tracking, propagation, collision handling, fragmentation, PMD, explosions, launch model, and result saving

import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from collections import defaultdict
# Remove external import of keplerian_to_rv
# === Simulated getidx fallback ===
def getidx():
    return {
        'a': 0,
        'ecco': 1,
        'inclo': 2,
        'nodeo': 3,
        'argpo': 4,
        'mo': 5,
        'mass': 6,
        'radius': 7,
        'objectclass': 8,
        'controlled': 9,
        'launch_date': 10,
        'ID': 11,
        'error': 12,
        'r': slice(13, 16),
        'v': slice(16, 19)
    }

# Placeholder for date-to-year conversion
def jd2date(jd):
    if np.isnan(jd):
        return datetime(1900, 1, 1)  # fallback default date
    return datetime.fromordinal(int(jd - 1721425.5))

# === Fallbacks for missing utilities ===

def keplerian_to_rv(a, e, i, raan, argp, f, mu):
    p = a * (1 - e**2)
    r_pf = np.array([
        p * np.cos(f) / (1 + e * np.cos(f)),
        p * np.sin(f) / (1 + e * np.cos(f)),
        0
    ])
    v_pf = np.array([
        -np.sqrt(mu / p) * np.sin(f),
        np.sqrt(mu / p) * (e + np.cos(f)),
        0
    ])
    
    # Rotation matrices
    cos_raan, sin_raan = np.cos(raan), np.sin(raan)
    cos_argp, sin_argp = np.cos(argp), np.sin(argp)
    cos_i, sin_i = np.cos(i), np.sin(i)

    R = np.array([
        [cos_raan * cos_argp - sin_raan * sin_argp * cos_i,
         -cos_raan * sin_argp - sin_raan * cos_argp * cos_i,
         sin_raan * sin_i],
        [sin_raan * cos_argp + cos_raan * sin_argp * cos_i,
         -sin_raan * sin_argp + cos_raan * cos_argp * cos_i,
         -cos_raan * sin_i],
        [sin_argp * sin_i,
         cos_argp * sin_i,
         cos_i]
    ])

    r = R @ r_pf
    v = R @ v_pf
    return r, v

def solve_kepler(M, e, tol=1e-8, max_iter=100):
    E = M if e < 0.8 else np.pi
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        delta = f / f_prime
        E -= delta
        if abs(delta) < tol:
            break
    return E

from datetime import datetime

def julian(date):
    # Gregorian calendar to Julian date
    a = (14 - date.month) // 12
    y = date.year + 4800 - a
    m = date.month + 12 * a - 3
    JDN = date.day + ((153 * m + 2) // 5) + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    JD = JDN + (date.hour - 12) / 24 + date.minute / 1440 + date.second / 86400
    return JD

# === Propagation ===
def prop_mit_vec(mat_sats, dt, cfg):
    mu = cfg['mu']
    idx = getidx()

    for i in range(mat_sats.shape[0]):
        a = mat_sats[i, idx['a']]
        e = mat_sats[i, idx['ecco']]
        i_deg = mat_sats[i, idx['inclo']]
        raan = mat_sats[i, idx['nodeo']]
        argp = mat_sats[i, idx['argpo']]
        M0 = mat_sats[i, idx['mo']]

        n_rad = np.sqrt(mu / a ** 3)
        M = M0 + n_rad * dt
        E = solve_kepler(M % (2 * np.pi), e)
        f = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                           np.sqrt(1 - e) * np.cos(E / 2))

        r_eci, v_eci = keplerian_to_rv(a, e, np.deg2rad(i_deg), np.deg2rad(raan), np.deg2rad(argp), f, mu)

        mat_sats[i, idx['r']] = r_eci
        mat_sats[i, idx['v']] = v_eci

    return mat_sats

# === Main Monte Carlo Simulation ===
def main_mc(cfgMC, RNGseed):
    np.random.seed(RNGseed)
    idx = getidx()

    mat_sats = cfgMC['mat_sats']
    tsince = cfgMC['tsince']
    n_time = cfgMC['n_time']
    radiusearthkm = cfgMC['radiusearthkm']
    time0 = cfgMC['time0']
    dt_days = cfgMC['dt_days']
    DAY2MIN = cfgMC['DAY2MIN']
    alph = cfgMC['alph']
    alph_a = cfgMC['alph_a']
    PMD = cfgMC['PMD']
    step_control = cfgMC.get('step_control', 1)
    launch_model = cfgMC.get('launch_model', 'no_launch')
    launch_rate = cfgMC.get('launch_rate', 0)

    deorbit_list = np.zeros(n_time)
    S_MC = np.zeros(n_time)
    D_MC = np.zeros(n_time)
    N_MC = np.zeros(n_time)
    B_MC = np.zeros(n_time)
    num_objects = np.zeros(n_time, dtype=int)

    for n in range(1, n_time):
        # === Collision + Fragmentation Model (simplified) ===
        if mat_sats.shape[0] > 1:
            pos = mat_sats[:, idx['r']]
            vel = mat_sats[:, idx['v']]
            rad = mat_sats[:, idx['radius']]
            mass = mat_sats[:, idx['mass']]
            objclass = mat_sats[:, idx['objectclass']]

            collisions = []
            for i in range(len(pos)):
                for j in range(i+1, len(pos)):
                    dist = np.linalg.norm(pos[i] - pos[j])
                    if dist < 2:  # km-level proximity threshold
                        collisions.append((i, j))

            for i, j in collisions:
                # Create fragments
                debris_mass = 0.1 * (mass[i] + mass[j]) / 2
                debris_radius = 0.1 * (rad[i] + rad[j])
                n_debris = 3
                new_objs = np.zeros((n_debris, mat_sats.shape[1]))
                for k in range(n_debris):
                    new_objs[k, idx['mass']] = debris_mass
                    new_objs[k, idx['radius']] = debris_radius
                    new_objs[k, idx['objectclass']] = 10  # debris class
                    new_objs[k, idx['controlled']] = 0
                    new_objs[k, idx['r']] = pos[i] + np.random.normal(0, 0.5, 3)
                    new_objs[k, idx['v']] = vel[i] + np.random.normal(0, 0.01, 3)

                mat_sats = np.delete(mat_sats, [i, j], axis=0)
                mat_sats = np.vstack((mat_sats, new_objs))
        if launch_model == 'random' and launch_rate > 0:
            # Placeholder for future launch model
            pass

        dt_sec = 60 * (tsince[n] - tsince[n - 1]) if n > 0 else 60 * tsince[n]
        mat_sats = prop_mit_vec(mat_sats, dt_sec, cfgMC)

        a = mat_sats[:, idx['a']]
        e = mat_sats[:, idx['ecco']]
        r = np.linalg.norm(mat_sats[:, idx['r']], axis=1)
        alt_perigee = (a * radiusearthkm) * (1 - e) - radiusearthkm

        invalid = (
            (r < radiusearthkm + 100) |
            (alt_perigee < 150) |
            (a < 0) |
            (mat_sats[:, idx['error']] != 0)
        )

        num_deorbited = np.sum(invalid)
        mat_sats = mat_sats[~invalid]

        deorbit_list[n] = deorbit_list[n - 1] + num_deorbited

        classes = mat_sats[:, idx['objectclass']].astype(int)
        controlled = mat_sats[:, idx['controlled']].astype(int)

        S_MC[n] = np.sum((classes == 1) & (controlled == 1))
        D_MC[n] = np.sum((classes == 1) & (controlled == 0))
        N_MC[n] = np.sum(classes == 10)
        B_MC[n] = np.sum(classes == 5)
        num_objects[n] = mat_sats.shape[0]

        print(f"Step {n}/{n_time}: {num_deorbited} deorbited, {mat_sats.shape[0]} remaining")

    save_results(cfgMC.get('output_file', 'mc_output'), S_MC, D_MC, N_MC, B_MC, num_objects, deorbit_list)
    return S_MC, D_MC, N_MC, B_MC, num_objects, deorbit_list

# === Save Results ===
def save_results(filename, S, D, N, B, total, deorbits):
    np.savez_compressed(filename,
                        S_MC=S,
                        D_MC=D,
                        N_MC=N,
                        B_MC=B,
                        num_objects=total,
                        deorbit_list=deorbits)
    print(f"Results saved to {filename}.npz")

# === Batch Monte Carlo Runner ===

def plot_batch_results(results, dt_days):
    t = np.arange(len(results['S'][0])) * dt_days / 365.25
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axs = axs.ravel()

    for metric, label, ax in zip(['S', 'D', 'N', 'Total'],
                                 ['Operational Sats', 'Derelict Sats', 'Debris', 'Total Objects'],
                                 axs):
        data = np.array(results[metric])
        mean = data.mean(axis=0)
        std = data.std(axis=0)

        ax.plot(t, mean, label=f"Mean {label}")
        ax.fill_between(t, mean - std, mean + std, alpha=0.3, label="±1σ")
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.grid(True)
        ax.legend()

    axs[-1].set_xlabel("Time [years]")
    fig.suptitle("Monte Carlo Simulation Results (Mean ± Std Dev)", fontsize=14)
    plt.tight_layout()
    plt.show()

    # === Deorbit Plot ===
    t_deorbit = np.arange(len(results['Deorbits'][0])) * dt_days / 365.25
    deorbits = np.array(results['Deorbits'])
    mean = deorbits.mean(axis=0)
    std = deorbits.std(axis=0)

    plt.figure(figsize=(10, 4))
    plt.plot(t_deorbit, mean, label='Mean Deorbits')
    plt.fill_between(t_deorbit, mean - std, mean + std, alpha=0.3, label='±1σ')
    plt.xlabel("Time [years]")
    plt.ylabel("Cumulative Deorbits")
    plt.title("Post-Mission & Natural Deorbits (Mean ± Std Dev)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_batch_mc(cfgMC, num_runs, seeds=None):
    results = {
        'S': [], 'D': [], 'N': [], 'B': [], 'Total': [], 'Deorbits': [], 'Events': [], 'meta': {}
    }
    if seeds is None:
        seeds = list(range(num_runs))
    elif len(seeds) != num_runs:
        raise ValueError("Length of seeds must match num_runs")

    for i, seed in enumerate(seeds):
        print(f"=== Running batch {i+1}/{num_runs} with seed {seed} ===")
        cfg_copy = cfgMC.copy()
        cfg_copy['output_file'] = f"{cfgMC.get('output_file', 'mc_output')}_run{seed}"
        S, D, N, B, T, Dlist = main_mc(cfg_copy, RNGseed=seed)
        events = {
            'seed': seed,
            'final_count': T[-1],
            'total_deorbits': Dlist[-1],
            'peak_debris': max(N)
        }
        results['S'].append(S)
        results['D'].append(D)
        results['N'].append(N)
        results['B'].append(B)
        results['Total'].append(T)
        results['Deorbits'].append(Dlist)
        results['Events'].append(events)
    return results
