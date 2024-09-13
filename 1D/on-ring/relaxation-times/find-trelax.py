import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from multiprocessing import Pool

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

    Trelax = find_trelax(Sp,Sm) 

    #plt.plot(Sp);plt.plot(Sm)
    #plt.show()
    #breakpoint()
    
    # Return final relaxation time (just returning T for this version)
    return Trelax

# Multiprocessing wrapper
def parallel_run(args):
    return run(*args)

# Main script with multiprocessing
def main(n=5, dt=0.5, T=10, ntrial=10, ks=[-4, -3, -2], num_workers=9):
    results = {}
    avg_times = []
    logging.info('Simulation started')

    # Create pool with specified number of workers
    with Pool(processes=num_workers) as pool:
        for k in ks:
            start_time = time.time()

            # Prepare arguments for parallel runs
            args = [(k, n, dt, T) for _ in range(ntrial)]
            
            # Track progress over trials
            Ts = []
            for i, result in enumerate(pool.imap(parallel_run, args), 1):
                Ts.append(result)
                Ts = [t for t in Ts if t != 0]  # Filter out zero results
                percent_complete = (i / ntrial) * 100
                logging.info(f'K={k}, {percent_complete:.1f}% complete')

            time_taken = time.time() - start_time
            logging.info(format_time(time_taken))

            # Save data
            filename = f"data/data_Trelax_n{n}_dt{dt}_T{T}_k{k}_ntrial{ntrial}.npy"
            np.save(filename, Ts)

            # Print missed trials
            num_fail = ntrial - len(Ts)
            logging.info(f"K={k}: Missed {num_fail}")

            # Save average relaxation time for plotting later
            avg_time = np.mean(Ts) if len(Ts) > 0 else 0
            avg_times.append(avg_time)
            results[k] = avg_time

    # Plot <T>(k)
    plt.figure(figsize=(8, 6))
    plt.plot(ks, avg_times, 'o-', linewidth=2, markersize=8, color='b', label=r'$\langle T_{relax} \rangle$')
    plt.xlabel(r'$k$', fontsize=14)
    plt.ylabel(r'$\langle T_{relax} \rangle$', fontsize=14)
    plt.title(f'N, dt, T, ntrial = {n}, {dt}, {T}, {ntrial}', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the final plot
    plt.savefig('figures/avg_Trelax_vs_k.png')
    plt.show()
    
    logging.info('Simulation completed')

if __name__ == "__main__":

    # Parameters
    n = 5
    dt = 0.1
    T = 10**4
    ntrials = 100
    ks = [-4, -3.5, -3, -2.5, -2, -1.5, -1.1]

    main(n=n, dt=dt, T=T, ntrial=ntrials, ks=ks)
