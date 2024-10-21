import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from multiprocessing import Pool
import os

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Find Wp and Wm
def find_ws(z):
    Wp = np.mean(np.exp(1j * (z[0] + z[1])))
    Wm = np.mean(np.exp(1j * (z[0] - z[1])))
    return Wp, Wm

# Define rhs equivalent function in Python
def rhs(z, n, J, k):
    x, theta = z
    xi = x + theta
    eta = x - theta
    Wp = np.mean(np.exp(1j * xi))
    Wm = np.mean(np.exp(1j * eta))
    dotx = (J / 2.0) * (np.imag(Wp * np.exp(-1j * xi)) + np.imag(Wm * np.exp(-1j * eta)))
    dottheta = (k / 2.0) * (np.imag(Wp * np.exp(-1j * xi)) - np.imag(Wm * np.exp(-1j * eta)))
    return np.array([dotx, dottheta])

# Implement Runge-Kutta 4 method
def rk4(z, F, dt, J, k):
    n = z.shape[1]
    k1 = F(z, n, J, k)
    k2 = F(z + dt/2 * k1, n, J, k)
    k3 = F(z + dt/2 * k2, n, J, k)
    k4 = F(z + dt * k3, n, J, k)
    return z + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# Main simulation with logging
def run(K, n1, dt, T, eps=1e-10, seed=0):
    J = 1
    np.random.seed(seed)
    z0 = np.array([np.random.uniform(0, 2*np.pi, n1), np.random.uniform(0, 2*np.pi, n1)])
    NT = int(T / dt)
    Wps, Wms = [], []

    # Log the start of the simulation
    logging.info(f"Starting simulation with K={K}, n1={n1}, dt={dt}, T={T}, seed={seed}")

    for t in range(NT):
        z0 = rk4(z0, rhs, dt, J, K)
        Wp, Wm = find_ws(z0)
        Wps.append(Wp)
        Wms.append(Wm)
        
        # Log progress every 10% completion
        if (t+1) % (NT // 10) == 0:
            completion = ((t+1) / NT) * 100
            logging.info(f"N={n1}, {completion:.1f}% complete")

    r, s = np.abs(Wps), np.abs(Wms)
    return r, s, dt


# Helper function for multiprocessing
def run_simulation(args):
    dt, seed = args
    K = -2.0
    T = 10**8
    n = 4
    r, s, _ = run(K, n, dt, T, seed=seed)
    t = [dt*i for i in range(len(r))]

    # Create directory if it doesn't exist
    save_dir = 'data'
    os.makedirs(save_dir, exist_ok=True)

    # Save the data to a file with an appropriate filename
    fname = f'{save_dir}/r_s_data_K{K}_n{n}_dt{dt}_T{T}_seed{seed}.npz'
    np.savez(fname, r=r, s=s, t=t)
    logging.info(f"Data saved to {fname}")
    return fname

# List of dt values and seeds to iterate over
num_workers = 8
dt_list = [0.01, 0.05, 0.1, 0.2, 0.5]
dt_list = [0.01]
seed_list = [0]  # Example seeds to iterate over

# Create a list of argument tuples (dt, seed) to pass to the multiprocessing pool
args_list = [(dt, seed) for dt in dt_list for seed in seed_list]

# Use multiprocessing to run simulations in parallel
if __name__ == '__main__':
    with Pool(processes=num_workers) as pool:
        pool.map(run_simulation, args_list)
