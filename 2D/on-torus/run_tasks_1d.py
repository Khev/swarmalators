""" Next: increase sampling near mixed region. Zoom in on it.

    Vary:
        - dt
        - T
        - N 
        - N_trials

 Do we see covergence with this?
"""

import os
import logging
import time
from utils import header_print

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

dist_types = ['uniform_deterministic', 'gaussian_deterministic', 'cauchy_deterministic',
              'uniform_random', 'gaussian_random', 'cauchy_random']

# Args
num_workers = 9
n = 10000
kmin = -3.0
kmax = 6.0
numk = 101
n_trial = 1
T = 200

type = 'cauchy'
dist_types = [f'{type}_deterministic', f'{type}_random']
for n_trial in [1,5,10,50]:
    for dist_type in dist_types:
        header = f'Starting simulation with N = {n} and distribution type = {dist_type}'
        header_print(header) 
        start_time = time.time()  # Get the current time in seconds since the Epoch
        cmd = (
                f'python compute_S_versus_K.py '
                f'--n={n} '
                f'--T={T} '
                f'--kmin={kmin} '
                f'--kmax={kmax} '
                f'--numk={numk} '
                f'--omega_dist={dist_type} '
                f'--n_trials={n_trial} '
                f'--num_workers={num_workers} '
                f'--parallel'
            )
        os.system(cmd)
        end_time = time.time()  # Get the current time in seconds since the Epoch
        elapsed_time = end_time - start_time
        logger.info(f"Finished simulation for N = {n}, Distribution = {dist_type}")
        logger.info(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
