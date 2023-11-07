import os
import argparse
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from multiprocessing import Pool
from utils import make_omega, rhs_2Dswarmlator_torus, rk4, find_2D_rainbow_order_parameters


def simulate_for_K(K, z0_init, J, mu, gamma, N, omega_dist, dt, T, cutoff_percen, n_trials):
    # Initialize arrays to store trial results for the averages
    Spx_trials = np.zeros(n_trials)
    Spy_trials = np.zeros(n_trials)
    Smx_trials = np.zeros(n_trials)
    Smy_trials = np.zeros(n_trials)
    
    # Convert the total simulation time and cutoff percentage to number of steps
    NT = int(T / dt)
    cutoff = int(cutoff_percen * NT)
    
    for trial in range(n_trials):
        # Generate new omega values for each trial
        u = make_omega(mu, gamma, N, omega_dist=omega_dist)
        v = make_omega(mu, gamma, N, omega_dist=omega_dist)
        w = make_omega(mu, gamma, N, omega_dist=omega_dist)
        # Copy the initial state to avoid modifying the original between trials
        z0 = z0_init.copy()
        
        # Initialize lists to store instantaneous order parameters
        Spx_list, Spy_list, Smx_list, Smy_list = [], [], [], []
        
        for t in range(NT):
            z = rk4(dt, z0, rhs_2Dswarmlator_torus, (J, K, u, v, w))
            x, y, theta = z[:N], z[N:2*N], z[2*N:]
            Wp_x, Wm_x, Wp_y, Wm_y = find_2D_rainbow_order_parameters(x, y, theta)
            Spx, Spy, Smx, Smy = np.abs(Wp_x), np.abs(Wp_y), np.abs(Wm_x), np.abs(Wm_y)
            Spx_list.append(Spx)
            Spy_list.append(Spy)
            Smx_list.append(Smx)
            Smy_list.append(Smy)
            z0 = z

        # Store the average of the order parameters after the cutoff
        Spx_trials[trial] = np.mean(Spx_list[cutoff:])
        Spy_trials[trial] = np.mean(Spy_list[cutoff:])
        Smx_trials[trial] = np.mean(Smx_list[cutoff:])
        Smy_trials[trial] = np.mean(Smy_list[cutoff:])
    
    # Average over all trials
    Spx_avg = np.mean(Spx_trials)
    Spy_avg = np.mean(Spy_trials)
    Smx_avg = np.mean(Smx_trials)
    Smy_avg = np.mean(Smy_trials)
    
    # Make sure Spx and Smx, Spy and Smy are correctly ordered
    if Spx_avg < Smx_avg:
        Spx_avg, Smx_avg = Smx_avg, Spx_avg
    if Spy_avg < Smy_avg:
        Spy_avg, Smy_avg = Smy_avg, Spy_avg
    
    return Spx_avg, Spy_avg, Smx_avg, Smy_avg


def simulate_for_fixed_K_old(K, dt, T, N, J, mu, gamma, z0_init, omega_dist):
    u = make_omega(mu, gamma, N, omega_dist = omega_dist) 
    v = make_omega(mu, gamma, N, omega_dist = omega_dist) 
    w = make_omega(mu, gamma, N, omega_dist = omega_dist)
    z0 = z0_init.copy() 
    NT = int(T / dt)
    cutoff = int(0.9 * NT)
    Spx_list, Spy_list, Smx_list, Smy_list = [], [], [], []
    for t in range(NT):
        z = rk4(dt, z0, rhs_2Dswarmlator_torus, (J, K, u, v, w))
        x, y, theta = z[:N], z[N:2*N], z[2*N:]
        Wp_x, Wm_x, Wp_y, Wm_y = find_2D_rainbow_order_parameters(x, y, theta)
        Spx, Spy, Smx, Smy = np.abs(Wp_x), np.abs(Wp_y), np.abs(Wm_x), np.abs(Wm_y)
        Spx_list.append(Spx)
        Spy_list.append(Spy)
        Smx_list.append(Smx)
        Smy_list.append(Smy)
        z0 = z
    Spx_avg = np.mean(Spx_list[cutoff:])
    Spy_avg = np.mean(Spy_list[cutoff:])
    Smx_avg = np.mean(Smx_list[cutoff:])
    Smy_avg = np.mean(Smy_list[cutoff:])

    if Spx_avg < Smx_avg:
        Spx_avg, Smx_avg = Smx_avg, Spx_avg
    if Spy_avg < Smy_avg:
        Spy_avg, Smy_avg = Smy_avg, Spy_avg

    return Spx_avg, Spy_avg, Smx_avg, Smy_avg


def plot_results(K_values, Spx_values, Smx_values, Spy_values, Smy_values, filename, args):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14))

    ax1.plot(K_values, Spx_values, label='Spx', color="blue")
    ax1.plot(K_values, Smx_values, label='Smx', color="red")
    ax1.set_title(f'Simulation Results for x-axis')
    ax1.set_title(f'$dt={args.dt}$, $J={args.J}$, $\\gamma={args.gamma}$, $N={args.n}$', fontsize=18)
    ax1.legend(fontsize=16, frameon=False)
    ax1.grid(True)

    ax2.plot(K_values, Spy_values, label='Spy', color="blue")
    ax2.plot(K_values, Smy_values, label='Smy', color="red")
    ax2.set_xlabel('K', fontsize=18)
    ax2.legend(fontsize=16, frameon=False)
    ax2.grid(True)

    # New third plot for max(Spx, Smx)
    S_max_values = np.maximum(Spx_values, Spy_values)
    ax3.plot(K_values, S_max_values, label='S := max(Spx,Smx)', color="green")
    ax3.set_xlabel('K', fontsize=18)
    ax3.legend(fontsize=16, frameon=False)
    ax3.grid(True)

    plt.tight_layout()
    directory = os.path.dirname(filename)
    os.makedirs(directory, exist_ok=True)
    plt.savefig(filename)
    plt.close(fig)


def main(args):
    z0_init = np.random.uniform(-np.pi, np.pi, 3*args.n)
    K_range = np.linspace(args.kmin, args.kmax, int(args.numk))

    # Check if the parallel flag is set
    if args.parallel:
        logging.info("Starting parallel processing...")
        with Pool(args.num_workers) as pool:
            params = [(K, z0_init, args.J, args.mu, args.gamma, args.n, args.omega_dist,
                       args.dt, args.T, args.cutoff_percen, args.n_trials) for K in K_range]
            results = pool.starmap(simulate_for_K, params)

        Spx_values, Spy_values, Smx_values, Smy_values = zip(*results)
        logging.info("Parallel processing complete.")
    else:
        logging.info("Starting sequential processing...")
        Spx_values, Spy_values, Smx_values, Smy_values = [], [], [], []
        for K in K_range:
            Spx_avg, Spy_avg, Smx_avg, Smy_avg = simulate_for_K(K, z0_init, args.J, args.mu, args.gamma, args.n,
                                                                args.omega_dist, args.dt, args.T, args.cutoff_percen,
                                                                args.n_trials)
            Spx_values.append(Spx_avg)
            Smx_values.append(Smx_avg)
            Spy_values.append(Spy_avg)
            Smy_values.append(Smy_avg)
        logging.info("Sequential processing complete.")

    # Save the data
    current_date = datetime.now().strftime('%Y-%m-%d')
    data_directory = f'data/{current_date}/'
    os.makedirs(data_directory, exist_ok=True)
    base_filename = f"2D_swarmalator_torus_J={args.J}_n={args.n}_T_{args.T}_ntrials_{args.n_trials}_{args.omega_dist}"
    data_dict = vars(args).copy()  # Make a copy of the args dictionary
    data_dict.update({
        'Ks': K_range.tolist(),  # Add K values
        'Spx': Spx_values,        # Add S+ values
        'Spy': Spy_values,        # Add S+ values
        'Smx': Smx_values,         # Add S- values
        'Smy': Smy_values         # Add S- values
    })
    json_fname = os.path.join(data_directory, f"{base_filename}.json")
    with open(json_fname, 'w') as f:
        json.dump(data_dict, f, indent=4)

    # Plot the results
    figures_directory = f'figures/{current_date}/'
    fig_fname = os.path.join(figures_directory, f"{base_filename}.png")
    plot_results(K_range, Spx_values, Smx_values, Spy_values, Smy_values, fig_fname, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Swarmalator Dynamics Simulation")
    parser.add_argument('--dt', type=float, default=0.1, help='Time step')
    parser.add_argument('--T', type=float, default=200, help='Total time')
    parser.add_argument('--cutoff_percen', type=float, default=0.9, help='Cutoff percen')
    parser.add_argument('--n', type=int, default=10000, help='Number of swarmalators')
    parser.add_argument('--J', type=float, default=9, help='Coupling constant J')
    parser.add_argument('--mu', type=float, default=0, help='Mean of the omega distribution')
    parser.add_argument('--gamma', type=float, default=1, help='Scale parameter gamma for the \
                         omega distribution')
    parser.add_argument('--omega_dist', type=str, default='cauchy_random', \
                         help='Distribution of omega')
    parser.add_argument('--kmin', type=float, default=-4, help='Minimum K value')
    parser.add_argument('--kmax', type=float, default=6, help='Maximum K value')
    parser.add_argument('--numk', type=float, default=131, help='Number of K values')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials for each K')
    parser.add_argument('--parallel', action='store_true', help='Run simulations in parallel')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of K values')
    args = parser.parse_args()

    main(args)