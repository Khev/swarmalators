import time

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import cauchy
from scipy.integrate import odeint

def rhs(z,t,J,K,n,nu,omega):
    
    x = z[:n]
    theta = z[n:]
    
    xd = x[:, np.newaxis] - x
    theta_d = theta[:, np.newaxis] - theta
    
    x_rhs = -J*np.nan_to_num(np.sin(xd)*np.cos(theta_d)) 
    theta_rhs = -K*np.nan_to_num(np.sin(theta_d)*np.cos(xd)) 
    
    x_next = np.nan_to_num( nu + (1/float(n))*np.sum((1-np.eye(xd.shape[0]))*x_rhs, axis=1))    
    theta_next = np.nan_to_num(omega +  (1/float(n))*np.sum((1-np.eye(xd.shape[0]))*theta_rhs, axis=1))
    
    return np.concatenate((x_next, theta_next))
    
    
def unpack(sols,n):
    """ I store the positions and phases of each swarmalator as a 3d vector z.
        Then the state vector for the systems is z[t][n]. This functions
        "unpacks" z[t][n] into x[t][n], y[t][n], theta[t][n]
    """

    T = len(sols)     #num timesteps
    x = np.array(np.zeros((T,n)))
    theta = np.array(np.zeros((T,n)))
    
    for t in range(T):
        x[t] = sols[t, 0:n]
        theta[t] = sols[t, n:2*n]
    
    return [x,theta]


def find_Ws(x,theta):
    """ Finds time series of order parameter W^+  = < e^{i*( phi_j + theta_j)} >_j -- see paper """
    numT, num_osc = x.shape
    W_plus = 1j*np.zeros(numT)
    W_minus = 1j*np.zeros(numT)
    for t in range(numT):
        W_plus[t] = np.sum(np.exp(1j*(x[t,:] + theta[t,:]))) / float(num_osc)
        W_minus[t] = np.sum(np.exp(1j*(x[t,:] - theta[t,:]))) / float(num_osc)
    return W_plus, W_minus


def S_sync(J,K,delta_v,delta_omega):
    delta_v, delta_omega = 1.0, 1.0
    delta_hat = (delta_v/J + delta_omega/K)
    return np.sqrt(1 - 2*delta_hat)

def S_phase_wave(J,K,delta_v,delta_omega):
    Jp = (J+K)/2.
    return np.sqrt(1 - 2*( delta_v + delta_omega) / Jp)


def scatter_t(x,theta,t):
    fig = plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.scatter(np.mod(x[t,:],2*np.pi)-np.pi, np.mod(theta[t,:],2*np.pi)-np.pi)
    plt.xlim([-np.pi,np.pi])
    plt.ylim([-np.pi,np.pi])
    plt.xlabel('x', fontsize=14);
    plt.ylabel('theta', fontsize=14)
