import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from multiprocessing import Pool
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# Find relaxation time
def find_trelax(Sp, Sm, eps=1e-10):
    for i in range(len(Sp)):
        if np.all(np.abs(Sp[i:]) < eps) and np.all(np.abs(Sm[i:]) < eps):
            return i
    return 0

# Find Wp and Wm
def find_ws(z):
    Wp = np.mean(np.exp(1j * (z[0] + z[1])))
    Wm = np.mean(np.exp(1j * (z[0] - z[1])))
    
    return Wp, Wm

# Format time
def format_time(total_seconds):
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"Took {hours}h {minutes}m {seconds}s" if hours > 0 else f"Took {minutes}m {seconds}s"

# Main simulation
def run(k, n1, dt, T, eps=1e-10):
    J = 1
    z0 = np.array([np.random.uniform(0, 2*np.pi, n1), np.random.uniform(0, 2*np.pi, n1)])
    NT = int(T / dt)
    Wps, Wms = [], []
    
    for t in range(NT):
        z0 = rk4(z0, rhs, dt, J, k)
        Wp, Wm = find_ws(z0)
        Wps.append(Wp)
        Wms.append(Wm)
        
    Sp = np.abs(Wps)
    Sm = np.abs(Wms)

    Trelax = find_trelax(Sp, Sm) 

    # Return final relaxation time
    return Trelax

# Multiprocessing wrapper
def parallel_run(args):
    return run(*args)

# Main script with multiprocessing and file existence check
def main(n_values=[5, 6, 7, 8, 9, 10], dt=0.5, T=10, ntrial=10, k=-1.5, num_workers=9):
    results = {}
    avg_times = []
    logging.info('Simulation started')

    # Create pool with specified number of workers
    with Pool(processes=num_workers) as pool:
        for n in n_values:
            filename = f"data/data_Trelax_n{n}_dt{dt}_T{T}_k{k}_ntrial{ntrial}.npy"

            # Check if the data for this n value already exists
            if os.path.exists(filename):
                logging.info(f"N={n}: Data already exists. Loading from file.")
                Ts = np.load(filename)
            else:
                start_time = time.time()

                # Prepare arguments for parallel runs
                args = [(k, n, dt, T) for _ in range(ntrial)]
                
                # Track progress over trials
                Ts = []
                for i, result in enumerate(pool.imap(parallel_run, args), 1):
                    Ts.append(result)
                    Ts = [t for t in Ts if t != 0]  # Filter out zero results
                    percent_complete = (i / ntrial) * 100
                    logging.info(f'N={n}, {percent_complete:.1f}% complete')

                time_taken = time.time() - start_time
                logging.info(format_time(time_taken))

                # Save data
                np.save(filename, Ts)

            # Print missed trials
            num_fail = ntrial - len(Ts)
            logging.info(f"N={n}: Missed {num_fail}")

            # Save average relaxation time for plotting later
            avg_time = np.mean(Ts) if len(Ts) > 0 else 0
            avg_times.append(avg_time)
            results[n] = avg_time

    # Plot <T>(n)
    plt.figure(figsize=(8, 18))

    # Regular plot (linear scale)
    plt.subplot(3, 1, 1)
    plt.plot(n_values, avg_times, 'o-', linewidth=2, markersize=8, color='b', label=r'$\langle T_{relax} \rangle$')
    plt.xlabel(r'$n$', fontsize=14)
    plt.ylabel(r'$\langle T_{relax} \rangle$', fontsize=14)
    plt.title('Regular Plot', fontsize=14)
    plt.grid(True)
    plt.legend()

    # Semi-log plot (logarithmic scale on y-axis only)
    plt.subplot(3, 1, 2)
    plt.semilogy(n_values, avg_times, 'o-', linewidth=2, markersize=8, color='g', label=r'$\log \langle T_{relax} \rangle$')
    plt.xlabel(r'$n$', fontsize=14)
    plt.ylabel(r'$\log(\langle T_{relax} \rangle)$', fontsize=14)
    plt.title('Logarithmic Plot (Y-axis)', fontsize=14)
    plt.grid(True, which="both")
    plt.legend()

    # Log-log plot (logarithmic scale on both axes)
    plt.subplot(3, 1, 3)
    plt.loglog(n_values, avg_times, 'o-', linewidth=2, markersize=8, color='r', label=r'$\log \langle T_{relax} \rangle$')
    plt.xlabel(r'$\log(n)$', fontsize=14)
    plt.ylabel(r'$\log(\langle T_{relax} \rangle)$', fontsize=14)
    plt.title('Log-Log Plot', fontsize=14)
    plt.grid(True, which="both")
    plt.legend()

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the final plot
    plt.savefig(f'figures/avg_Trelax_vs_n_combined_k_{k}_ntrial_{ntrial}.png')
    plt.show()
    
    logging.info('Simulation completed')

if __name__ == "__main__":

    # Parameters
    n_values = [5,6,7,8,9,10, 50, 10**2, 500, 10**3, 5000, 10**4, 10**5]  # Varying n
    n_values = [5,6,7,8,9,10, 32, 10**2, 316, 10**3, 3162, 10**4, 31622, 10**5]  # Varying n
    n_values = [316222, 10**6]  # Varying n
    dt = 0.1
    T = 10**4
    ntrials = 10**2
    k = -5.0  # Hold k constant

    main(n_values=n_values, dt=dt, T=T, ntrial=ntrials, k=k)
