"""
DESCRIPTION:
This script performs a series of simulations to explore the dynamics of swarmalators,
which are swarming oscillators with coupled phase and spatial dynamics. The simulations
vary the strength of the phase coupling parameter K within a range and either processes
the simulation results in parallel (if the parallel flag is set) or sequentially.

The swarmalator model includes a coupling constant J, a mean frequency mu, and a scale
parameter gamma that determines the nature of the frequency distribution (omega distribution).
The script computes order parameters that describe the level of synchronization within the
swarmalator system. The resulting order parameters for each K are then plotted and saved
for further analysis.

Usage:
Run the script with optional arguments to define simulation parameters such as time step,
total time, number of swarmalators, and others. If the parallel flag is set, it uses
multiprocessing to run simulations in parallel across different values of K.

Example:
`python compute_S_versus_K.py --dt 0.25 --T 50 --n 100 --J 9 --gamma 1 --kmin -4 --kmax 9 --numk 131 --parallel`

Output:
The script outputs a JSON file containing the parameters and results of the simulations,
as well as a PNG image of the plot showing the order parameters across the range of K values.
"""


import os
import argparse
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from utils import make_omega_nu, find_rainbow_order_parameters, rk4, rhs_swarmlator
from multiprocessing import Pool

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def simulate_for_K(K, z0, J, mu, gamma, n, omega_dist, omega, nu, dt, NT, cutoff_percen, n_trials):
    # Initialize arrays to store trial results
    Sp_trials = np.zeros(n_trials)
    Sm_trials = np.zeros(n_trials)
    
    for trial in range(n_trials):
        # Copy the initial state to avoid modifying the original between trials
        z = np.copy(z0)
        Ws = []
        for t in range(NT):
            Wp, Wm = find_rainbow_order_parameters(z)
            z = rk4(dt, z, rhs_swarmlator, (J, K, nu, omega))
            Ws.append((Wp, Wm))
        Sp = [np.abs(wp) for wp, _ in Ws]
        Sm = [np.abs(wm) for _, wm in Ws]
        cutoff = int(cutoff_percen * len(Sp))
        Sp_trials[trial] = np.mean(Sp[cutoff:])
        Sm_trials[trial] = np.mean(Sm[cutoff:])
    
    # Average over all trials
    Sp_final = np.mean(Sp_trials)
    Sm_final = np.mean(Sm_trials)
    
    if Sp_final < Sm_final:
        Sp_final, Sm_final = Sm_final, Sp_final
    
    return Sp_final, Sm_final

def plot_results(K_range, Sp_values, Sm_values, filename,args):
    plt.figure(figsize=(10, 5))
    plt.plot(K_range, Sp_values, label="Sp", color="blue")
    plt.plot(K_range, Sm_values, label="Sm", color="red")
    plt.xlabel('K', fontsize=20)
    plt.ylim([0,1.05])
    plt.title(f'(J, gamma, N) = ({args.J}, {args.gamma}, {args.n})', fontsize=17)
    plt.legend(fontsize=16, frameon=False)
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    directory = os.path.dirname(filename)
    os.makedirs(directory, exist_ok=True)
    plt.savefig(filename)

def main(args):
    # Set up initial parameters
    n = args.n
    nu, omega = make_omega_nu(n, args.mu, args.mu, args.gamma, args.gamma, dist_type=args.omega_dist)
 
    (x0, theta0) = np.random.uniform(0, 2*np.pi, n), np.random.uniform(0, 2*np.pi, n)
    z0 = np.concatenate([x0, theta0])
    NT = int(args.T / args.dt)

    # Set up multiprocessing if the parallel flag is set
    K_range = np.linspace(args.kmin, args.kmax, int(args.numk))
    if args.parallel:
        logging.info("Starting parallel processing...")
        pool = Pool(args.num_workers)
        results = pool.starmap(simulate_for_K, [(K, z0, args.J, args.mu, args.gamma, n, args.omega_dist, omega, nu,
                                             args.dt, NT, args.cutoff_percen, args.n_trials) for K in K_range])
        Sp_values = [result[0] for result in results]
        Sm_values = [result[1] for result in results]
        logging.info("Parallel processing complete.")
    else:
        # Sequential processing
        logging.info("Starting sequential processing...")
        Sp_values = []
        Sm_values = []
        for K in K_range:
            Sp_final, Sm_final = simulate_for_K(K, z0, args.J, args.mu, args.gamma, n, args.omega_dist, omega, nu, \
                                                 args.dt, NT, args.cutoff_percen, args.n_trials)
            Sp_values.append(Sp_final)
            Sm_values.append(Sm_final)
            logging.info(f"Simulation for K={K:.2f} complete.")
        logging.info("Sequential processing complete.")

    # Save the data
    current_date = datetime.now().strftime('%Y-%m-%d')
    data_directory = f'data/{current_date}/'
    os.makedirs(data_directory, exist_ok=True)
    base_filename = f"1D_swarmalator_J={args.J}_n={args.n}_T_{args.T}_ntrials_{args.n_trials}_{args.omega_dist}"
    data_dict = vars(args).copy()  # Make a copy of the args dictionary
    data_dict.update({
        'Ks': K_range.tolist(),  # Add K values
        'Sps': Sp_values,        # Add S+ values
        'Sms': Sm_values         # Add S- values
    })
    json_fname = os.path.join(data_directory, f"{base_filename}.json")
    with open(json_fname, 'w') as f:
        json.dump(data_dict, f, indent=4)

    # Plot the results
    figures_directory = f'figures/{current_date}/'
    fig_fname = os.path.join(figures_directory, f"{base_filename}.png")
    plot_results(K_range, Sp_values, Sm_values, fig_fname, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Swarmalator Dynamics Simulation")
    parser.add_argument('--dt', type=float, default=0.1, help='Time step')
    parser.add_argument('--T', type=float, default=50, help='Total time')
    parser.add_argument('--cutoff_percen', type=float, default=0.9, help='Cutoff percen')
    parser.add_argument('--n', type=int, default=100, help='Number of swarmalators')
    parser.add_argument('--J', type=float, default=9, help='Coupling constant J')
    parser.add_argument('--mu', type=float, default=0, help='Mean of the omega distribution')
    parser.add_argument('--gamma', type=float, default=1, help='Scale parameter gamma for the \
                         omega distribution')
    parser.add_argument('--omega_dist', type=str, default='cauchy_random', \
                         help='Distribution of omega')
    parser.add_argument('--kmin', type=float, default=-4, help='Minimum K value')
    parser.add_argument('--kmax', type=float, default=9, help='Maximum K value')
    parser.add_argument('--numk', type=float, default=131, help='Number of K values')
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials for each K')
    parser.add_argument('--parallel', action='store_true', help='Run simulations in parallel')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of K values')
    args = parser.parse_args()
    main(args)
