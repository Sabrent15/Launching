import scipy.io
import numpy as np
from datetime import datetime, timedelta
from utils import getidx, jd2date

def setup_MCconfig(seed, ICfile='2020.mat'):
    # Constants
    radiusearthkm = 6371.0
    mu = 398600.4418  # km^3/s^2
    j2 = 1.08262668e-3
    DAY2MIN = 1440
    YEAR2MIN = 525600

    np.random.seed(seed)
    
    # Create time vector
    dt_days = 5
    nyears = 10
    tf_prop = YEAR2MIN * nyears
    DeltaT = dt_days * DAY2MIN
    tsince = np.arange(0, tf_prop + DeltaT, DeltaT)
    n_time = len(tsince)

    # Load ICfile
    mat = scipy.io.loadmat(f'data/{ICfile}')
    mat_sats = mat['mat_sats']
    time0 = mat['time0'][0] if 'time0' in mat else datetime(2020, 1, 1)

    idx = getidx()
    a_all = mat_sats[:, idx['a']] * radiusearthkm
    e_all = mat_sats[:, idx['ecco']]
    ap_all = a_all * (1 - e_all)
    aa_all = a_all * (1 + e_all)

    # Remove sats outside altitude bounds
    altitude_limit_low = 200
    altitude_limit_up = 2000
    keep = (ap_all >= altitude_limit_low + radiusearthkm) & (ap_all <= altitude_limit_up + radiusearthkm)
    mat_sats = mat_sats[keep]

    # Derelict tagging
    mission_lifetime = 8
    launch_dates = mat_sats[:, idx['launch_date']]
    derelict_cutoff_year = jd2date(launch_dates).year < (time0.year - mission_lifetime)
    mat_sats[derelict_cutoff_year, idx['controlled']] = 0

    # Init config dict
    cfgMC = {
        'seed': seed,
        'radiusearthkm': radiusearthkm,
        'mu': mu,
        'j2': j2,
        'DAY2MIN': DAY2MIN,
        'YEAR2MIN': YEAR2MIN,
        'tsince': tsince,
        'n_time': n_time,
        'dt_days': dt_days,
        'altitude_limit_low': altitude_limit_low,
        'altitude_limit_up': altitude_limit_up,
        'missionlifetime': mission_lifetime,
        'launch_model': 'no_launch',
        'mat_sats': mat_sats,
        'repeatLaunches': np.empty((0, 23)),  # empty for no launch
        'time0': time0,
        'skipCollisions': 0,
        'use_sgp4': False,
        'save_output_file': 0,
        'animation': 'no',
        'CUBE_RES': 50,
        'collision_alt_limit': 45000,
        'PMD': 0.95,
        'alph': 0.01,
        'alph_a': 0,
        'step_control': 2,
        'orbtol': 5,
        'P_frag': 0,
        'P_frag_cutoff': 18,
        'max_frag': float('inf'),
        'filename_save': f"TLEIC_year{time0.year}_rand{seed}.mat",
        'maxID': int(np.max(mat_sats[:, idx['ID']])) if mat_sats.shape[0] > 0 else 0,
        'a_all': a_all,
        'ap_all': ap_all,
        'aa_all': aa_all,
    }

    return cfgMC
