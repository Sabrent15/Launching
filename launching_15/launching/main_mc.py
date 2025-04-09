import numpy as np
import os
import psutil

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
    save_interval = cfgMC.get('save_interval', 10)
    output_file = cfgMC.get('output_file', 'mc_output')

    deorbit_list = np.zeros(n_time)
    S_MC = np.zeros(n_time)
    D_MC = np.zeros(n_time)
    N_MC = np.zeros(n_time)
    B_MC = np.zeros(n_time)
    num_objects = np.zeros(n_time, dtype=int)

    for n in range(1, n_time):
        # === Launch Model ===
        if launch_model == 'random' and launch_rate > 0:
            new_launches = generate_random_launches(launch_rate, mat_sats.shape[1], idx)
            mat_sats = np.vstack((mat_sats, new_launches))
        elif launch_model == 'data':
            new_launches = get_data_launches(n, cfgMC)
            mat_sats = np.vstack((mat_sats, new_launches))

        # === Collision + Fragmentation Model ===
        if mat_sats.shape[0] > 1:
            pos = mat_sats[:, idx['r']]
            vel = mat_sats[:, idx['v']]
            rad = mat_sats[:, idx['radius']]
            mass = mat_sats[:, idx['mass']]
            objclass = mat_sats[:, idx['objectclass']]

            collisions = []
            for i in range(len(pos)):
                for j in range(i + 1, len(pos)):
                    dist = np.linalg.norm(pos[i] - pos[j])
                    rel_vel = np.linalg.norm(vel[i] - vel[j])
                    prob_collision = compute_collision_probability(dist, rad[i], rad[j], rel_vel)
                    if np.random.rand() < prob_collision:
                        collisions.append((i, j))

            for i, j in collisions:
                debris = generate_collision_debris(mass[i], mass[j], rad[i], rad[j], pos[i], vel[i], idx)
                mat_sats = np.delete(mat_sats, [i, j], axis=0)
                mat_sats = np.vstack((mat_sats, debris))

        # === Explosion Model ===
        rocket_bodies = np.where(mat_sats[:, idx['objectclass']] == 5)[0]
        for rb in rocket_bodies:
            if np.random.rand() < cfgMC.get('explosion_prob', 0.01):
                debris = generate_explosion_debris(mat_sats[rb], idx)
                mat_sats = np.delete(mat_sats, rb, axis=0)
                mat_sats = np.vstack((mat_sats, debris))

        # === Propagation ===
        dt_sec = 60 * (tsince[n] - tsince[n - 1]) if n > 0 else 60 * tsince[n]
        mat_sats = prop_mit_vec(mat_sats, dt_sec, cfgMC)

        # === Deorbiting ===
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

        # === Categorization ===
        classes = mat_sats[:, idx['objectclass']].astype(int)
        controlled = mat_sats[:, idx['controlled']].astype(int)

        S_MC[n] = np.sum((classes == 1) & (controlled == 1))
        D_MC[n] = np.sum((classes == 1) & (controlled == 0))
        N_MC[n] = np.sum(classes == 10)
        B_MC[n] = np.sum(classes == 5)
        num_objects[n] = mat_sats.shape[0]

        # === Save Intermediate Results ===
        if n % save_interval == 0 or n == n_time - 1:
            save_results(f"{output_file}_part_{n // save_interval}", S_MC, D_MC, N_MC, B_MC, num_objects, deorbit_list)

        # === Memory Management ===
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 ** 3)  # in GB
        if memory_usage > 2.0:  # Example threshold
            print(f"Warning: High memory usage detected ({memory_usage:.2f} GB). Consider optimizing data storage.")

        print(f"Step {n}/{n_time}: {num_deorbited} deorbited, {mat_sats.shape[0]} remaining")

    save_results(output_file, S_MC, D_MC, N_MC, B_MC, num_objects, deorbit_list)
    return S_MC, D_MC, N_MC, B_MC, num_objects, deorbit_list


# === Helper Functions ===
def generate_random_launches(rate, num_cols, idx):
    n_launches = np.random.poisson(rate)
    new_objs = np.zeros((n_launches, num_cols))
    for i in range(n_launches):
        new_objs[i, idx['mass']] = np.random.uniform(100, 1000)  # Example mass range
        new_objs[i, idx['radius']] = np.random.uniform(0.5, 2)  # Example radius range
        new_objs[i, idx['objectclass']] = 1  # Satellite class
        new_objs[i, idx['controlled']] = 1
        new_objs[i, idx['r']] = np.random.uniform(-1e4, 1e4, 3)  # Random position
        new_objs[i, idx['v']] = np.random.uniform(-1, 1, 3)  # Random velocity
    return new_objs


def compute_collision_probability(dist, rad1, rad2, rel_vel):
    combined_radius = rad1 + rad2
    if dist > combined_radius:
        return 0
    return np.exp(-dist / combined_radius) * (rel_vel / 10)


def generate_collision_debris(mass1, mass2, rad1, rad2, pos, vel, idx):
    n_debris = 3
    debris = np.zeros((n_debris, len(idx)))
    debris_mass = 0.1 * (mass1 + mass2) / 2
    debris_radius = 0.1 * (rad1 + rad2)
    for k in range(n_debris):
        debris[k, idx['mass']] = debris_mass
        debris[k, idx['radius']] = debris_radius
        debris[k, idx['objectclass']] = 10  # debris class
        debris[k, idx['controlled']] = 0
        debris[k, idx['r']] = pos + np.random.normal(0, 0.5, 3)
        debris[k, idx['v']] = vel + np.random.normal(0, 0.01, 3)
    return debris


def generate_explosion_debris(rocket_body, idx):
    n_debris = 5
    debris = np.zeros((n_debris, len(idx)))
    for k in range(n_debris):
        debris[k, idx['mass']] = rocket_body[idx['mass']] * 0.2
        debris[k, idx['radius']] = rocket_body[idx['radius']] * 0.5
        debris[k, idx['objectclass']] = 10  # debris class
        debris[k, idx['controlled']] = 0
        debris[k, idx['r']] = rocket_body[idx['r']] + np.random.normal(0, 0.5, 3)
        debris[k, idx['v']] = rocket_body[idx['v']] + np.random.normal(0, 0.01, 3)
    return debris