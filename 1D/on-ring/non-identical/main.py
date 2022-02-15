import time

import funcs as f
import numpy as np

from scipy.stats import cauchy
from scipy.integrate import odeint


if __name__ == '__main__':

    # Numerical parameters
    dt, T, n  = 0.1, 50, 5 
    np.random.seed(0)
    x0 = np.random.uniform(-np.pi,np.pi,n);
    theta0 = np.random.uniform(-np.pi,np.pi,n)
    t = [dt*i for i in range(int(T/dt))]    
    z0 = np.array([x0,theta0])
    z0 = z0.flatten()

    # Physical parameters
    J = 9.0
    cutoff = int(0.8*T)
    #nu, omega = np.zeros(n), np.zeros(n)
    nu, omega = cauchy.rvs(size=n), cauchy.rvs(size=n)

    Wp, Wm = [], []
    Sp, Sm = [], []
    Ks = np.linspace(-2,8,4)
    for K in Ks:
        tic = time.clock()
        sols = odeint(f.rhs, z0, t, args=(J,K,n,nu,omega))
        x, theta = f.unpack(sols,n)
        W_plus, W_minus = f.find_Ws(x,theta)
        S_plus, S_minus = np.abs(W_plus), np.abs(W_minus)
        S_plus, S_minus = np.mean(S_plus[cutoff:-1]), np.mean(S_minus[cutoff:-1])
        if S_plus < S_minus:
            S_plus, S_minus = S_minus, S_plus
        Sp.append(S_plus); Sm.append(S_minus);
        Wp.append(W_plus); Wm.append(W_minus)
        toc = time.clock()
        tim = (toc-tic) / 60.0
        print(f'K = {K:.2f} took {tim:.2f} mins')

    # Plot data
    plt.figure(figsize=(15,5))
    plt.plot(Ks,Sp,'ro');plt.plot(Ks,Sm,'bo')

    # Plot theoretical curves
    Ks1 = np.linspace(-2,8,1000)
    J, delta_v, delta_omega = 9, 1, 1
    Ss = [f.S_sync(J,K,delta_v,delta_omega) for K in Ks1]
    Sph = [f.S_phase_wave(J,K,delta_v,delta_omega) for K in Ks1]
    plt.plot(Ks1,Ss);plt.plot(Ks1,Sph)
    plt.xlabel('K', fontsize=16);
    plt.legend(["S+","S-",'S sync', 'S phase wave'], fontsize=14, frameon=False)
    plt.ylim([0,1])
    plt.title(f'(J, n, dt, T) = ({J}, {n}, {dt}, {T})', fontsize=14);

    # Save data
    fname = f'data/Wplus_versus_K_J_{J:.2f}_n_{n}.npy'
    np.save(fname,Wp)

    fname = f'data/Wminus_versus_K_J_{J:.2f}_n_{n}.npy'
    np.save(fname,Wm)

    fname = f'data/Ks_J_{J:.2f}_n_{n}.npy'
    np.save(fname,Ks)

    # Save figure
    figname = f'figures/S_versus_K_J_{J:.2f}_n_{n}.png'
    plt.savefig(figname)
