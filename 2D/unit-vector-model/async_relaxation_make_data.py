import numpy as np
import swarmalators_funcs as f
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time
import os
from multiprocessing import Pool

# ----------------------- Helper Functions ----------------------- #

def save_velocity(v, filename):
    """Helper function to save velocity data to file."""
    save_dir = os.path.join('data', f'async_relaxation_{filename}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the velocity data as a .npy file
    filepath = os.path.join(save_dir, f'velocity_{filename}.npy')
    np.save(filepath, v)

def plot_and_save_figure(v, dt, T, n, filename):
    """Helper function to plot results and save figures."""
    ts = np.linspace(0, T, int(T/dt)-1)
    vmean = [np.mean(v[t, :]) for t in range(v.shape[0])]

    # Create a 2x1 subplot layout
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))

    # Log plot (semilogarithmic on y-axis)
    ax[0].semilogy(ts, vmean)
    ax[0].set_title(f'(N, dt, T) = ({n}, {dt}, {T})')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Mean Velocity (log scale)')

    # Log-log plot
    ax[1].loglog(ts, vmean)
    ax[1].set_xlabel('Time (log scale)')
    ax[1].set_ylabel('Mean Velocity (log scale)')

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the figure in the 'figures/' directory
    figure_dir = 'figures'
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    figure_path = os.path.join(figure_dir, f'plot_{filename}.png')
    fig.savefig(figure_path)
    print(f"Figure saved as {figure_path}")

    # Optionally, you can show the plot
    # plt.show()
    plt.close(fig)  # Close the figure to free up memory

def run_simulation(pars, dt, T, n, L, seed):
    """Runs the simulation and saves only the `v` results to a file."""
    # Unpack parameters
    J, K, omega = pars
    t = [dt * i for i in range(int(T/dt))]
    
    # Set the random seed for reproducibility
    np.random.seed(seed)
    x0 = np.random.uniform(-L, L, n)
    y0 = np.random.uniform(-L, L, n)
    theta0 = np.random.uniform(-np.pi, np.pi, n)

    # Initial conditions
    z0 = np.array([x0, y0, theta0]).flatten()

    # Solve the ODE
    tic = time.time()
    sols = odeint(f.rhs_unit_vector, z0, t, args=(J, K, n, omega))
    x, y, theta = f.unpack(sols, n)
    
    # Extract the velocity results
    v = f.find_vel_t(x, y, theta, dt)
    
    # Filename to save data (includes parameters and seed)
    filename = f'K_{K}_J_{J}_N_{n}_T_{T}_dt_{dt}_seed_{seed}'
    
    # Save only the velocity data
    save_velocity(v, filename)
    
    toc = time.time()
    print(f'(J, K) = ({J}, {K}) with seed={seed} took {(toc - tic)/3600.0:.2f} hours')

    # Plot the results and save the figure
    plot_and_save_figure(v, dt, T, n, filename)

    return v

# ----------------------- Multiprocessing Wrapper ----------------------- #

def run_simulation_with_seed(seed, pars, dt, T, n, L):
    return run_simulation(pars, dt, T, n, L, seed)

def load_velocity_for_seed(filename):
    """Helper function to load velocity data for a given seed."""
    filepath = os.path.join('data', f'async_relaxation_{filename}', f'velocity_{filename}.npy')
    if os.path.exists(filepath):
        return np.load(filepath)
    else:
        raise FileNotFoundError(f"Velocity data for {filename} not found.")

def average_over_seeds(seeds, dt, T, n, J, K):
    """Loads data for all seeds, averages them, and plots the average."""
    velocities = []
    
    for seed in seeds:
        filename = f'K_{K}_J_{J}_N_{n}_T_{T}_dt_{dt}_seed_{seed}'
        v = load_velocity_for_seed(filename)
        velocities.append(v)

    # Stack the velocities and average across the seed dimension
    velocities = np.array(velocities)
    v_avg = np.mean(velocities, axis=0)  # Averaging over the seeds

    # Plot the averaged velocity
    plot_and_save_figure(v_avg, dt, T, n, f'average_K_{K}_J_{J}_N_{n}_T_{T}_dt_{dt}')

# ----------------------- Main Script ----------------------- #

if __name__ == "__main__":
    
    # Simulation Parameters
    dt, T, n, L = 0.1, 10**3, 10**2, 1
    J, K = 1.0, -2.0
    
    # Set up the system parameters
    omega = np.zeros(n)
    pars = [J, K, omega]
    
    # Define the list of seeds for parallelization
    seeds = [0, 1, 2, 3, 4]  # Example seeds, can be extended
    
    # Use multiprocessing to run simulations in parallel
    num_workers = len(seeds)  # Set number of workers to number of seeds
    pool = Pool(processes=num_workers)
    
    # Use starmap to run simulations in parallel, passing both seed and parameters
    pool.starmap(run_simulation_with_seed, [(seed, pars, dt, T, n, L) for seed in seeds])
    
    # Close the pool and wait for the jobs to finish
    pool.close()
    pool.join()

    # After all simulations are done, average over all seeds and plot the result
    average_over_seeds(seeds, dt, T, n, J, K)
    
    print("All simulations and averaging completed.")
