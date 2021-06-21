""" Make data for S_{+-}(K) graphs """

import time

import numpy as np
import swarmalators_funcs as f

from multiprocessing import Pool
from scipy.integrate import odeint


# Function that will be passed to the multi-processor
def func(pars):

    #Define parameters
    a, dt, T, n, L  = 1, 0.5, 500, 100, 1 
    t = [dt*i for i in range(int(T/dt))]
    np.random.seed(0)
    x0 = np.random.uniform(-L,L,n);y0=np.random.uniform(-L,L,n);theta0 = np.random.uniform(-np.pi,np.pi,n)
    J, K, omega = pars[0], pars[1], np.zeros(n);

    
    #Do simulation
    tic = time.clock()
    z0 = np.array([x0,y0,theta0])
    z0 = z0.flatten()
    sols = odeint(f.rhs_unit_vector, z0, t, args=(J,K,n,omega))
    x, y, theta = f.unpack(sols,n)
    r, phi = f.cart_to_polar(x,y)
    gamma = f.find_gamma(phi)
    v = f.find_vel_t(x,y,theta,dt)
    transient_index = int(0.9*x.shape[0])
    v_mean = np.mean([np.mean(v[t,:]) for t in range(transient_index,v.shape[0])])
    S_plus = np.mean(np.abs(f.find_W_plus(x,y,theta))[transient_index:-1])
    S_minus = np.mean(np.abs(f.find_W_minus(x,y,theta))[transient_index:-1])
    R_max = np.mean([np.max( np.sqrt(x[t,:]**2 + y[t,:]**2) ) for t in range(x.shape[0])][transient_index:-1])
    R_min = np.mean([np.min( np.sqrt(x[t,:]**2 + y[t,:]**2) ) for t in range(x.shape[0])][transient_index:-1])

    
    #Save to file
    string =  'K_' + str(K) + '_J_' + str(J) + '_N_' + str(n) + '_T_' + str(T) + '_dt_' + str(dt) +  '.csv'
    with open('data/S_plus_' + string, 'wb') as f1:
        np.savetxt(f1, [S_plus])
    with open('data/S_minus_' + string, 'wb') as f2:
        np.savetxt(f2, [S_minus])
    with open('data/v_mean_' + string, 'wb') as f6:
        np.savetxt(f6, [v_mean])
    with open('data/gamma_' + string, 'wb') as f65:
        np.savetxt(f65, np.array([gamma]))
    with open('data/R_max_' + string, 'wb') as f7:
        np.savetxt(f7, [R_max])
    with open('data/R_min_' + string, 'wb') as f8:
        np.savetxt(f8, [R_min])
    toc = time.clock()
    print(f'(J,K) = ({J},{K}) took {(toc-tic)/60.0**2:.2f} hours')
    return v_mean


if __name__ == '__main__':
    num_of_processes = 6
    Ks = np.linspace(-2,0,11)
    Js = np.array([1])
    
    print('Starting calculations for ')

    for J in Js:
        #Set up parameters in right foramt
        temp = np.full(len(Ks),J)
        pars = zip(temp,Ks)

        #Do sims
        workers = Pool(num_of_processes)  #create the workers
        workers.map(func, pars)           #set them working!
        workers.close()                   #close them once they're done
        workers.join()