import numpy as np
from scipy.integrate import odeint
from scipy import stats
from scipy.stats import uniform
#import matplotlib.pyplot as plt
#from matplotlib  import cm
#import matplotlib as mpl



def rhs_unit_vector(z,t,J,K,n,omega):
    """
    z0 = [x0, y0, theta0], where x0[i], y0[i], theta0[i] gives the x[t=0], y[y=0], theta[t=0] of the i-th swarmalator
    t = time parameter (need this here to pass to python's odeint function)
    J = float
    K = float
    n = number of swarmalators
    omega = [omega1, omega2, ..., ] natural frequencies of swarmaltors
    """
    
    #Instantiate -- set up 
    x = z[0:n]
    y = z[n:2*n]
    theta = z[2*n:3*n]
    
    # Set up as a numpy.array to make the computation faster
    xd = x[:, np.newaxis] - x
    yd = y[:, np.newaxis] - y
    theta_d = theta[:, np.newaxis] - theta
    inverse_dist_sq = np.nan_to_num(1.0/((xd)**2 + (yd)**2))
    np.fill_diagonal(inverse_dist_sq, 0.0)  ## correct 1 / d_ii = 1 / o error
    inverse_dist = np.sqrt(inverse_dist_sq)
    
    x_rhs = -1*np.nan_to_num( xd*( (1+J*np.cos(theta_d))*inverse_dist - inverse_dist_sq)) 
    y_rhs = -1*np.nan_to_num( yd*( (1+J*np.cos(theta_d))*inverse_dist - inverse_dist_sq)) 
    theta_rhs = -1*K*np.sin(theta_d)*inverse_dist
    
    
    #The actual R.H.S.
    x_next = np.nan_to_num((1/float(n))*np.sum((1-np.eye(xd.shape[0]))*x_rhs, axis=1))
    y_next = np.nan_to_num((1/float(n))*np.sum((1-np.eye(xd.shape[0]))*y_rhs, axis=1))
    theta_next = np.nan_to_num(omega +  (1/float(n))*np.sum((1-np.eye(xd.shape[0]))*theta_rhs, axis=1))
                                              
    return np.concatenate((x_next, y_next, theta_next))



def rhs_linear(z,t,J,K,n,omega):
    
    #Instantiate
    x = z[0:n]
    y = z[n:2*n]
    theta = z[2*n:3*n]
    xd = x[:, np.newaxis] - x
    yd = y[:, np.newaxis] - y
    theta_d = theta[:, np.newaxis] - theta
    inverse_dist_sq = np.nan_to_num(1.0/((xd)**2 + (yd)**2))
    np.fill_diagonal(inverse_dist_sq, 0.0) ## correct 1 / d_ii = 0 / 1 eror
    
    x_rhs = -1*np.nan_to_num( xd*( (1+J*np.cos(theta_d)) - inverse_dist_sq)  ) 
    y_rhs = -1*np.nan_to_num( yd*( (1+J*np.cos(theta_d)) - inverse_dist_sq)  ) 
    theta_rhs = -1*K*np.sin(theta_d)*np.sqrt(inverse_dist_sq)
    
    
    #RHS    
    x_next = np.nan_to_num((1/float(n))*np.sum((1-np.eye(xd.shape[0]))*x_rhs, axis=1))
    y_next = np.nan_to_num((1/float(n))*np.sum((1-np.eye(xd.shape[0]))*y_rhs, axis=1))
    theta_next = np.nan_to_num(omega +  (1/float(n))*np.sum((1-np.eye(xd.shape[0]))*theta_rhs, axis=1))
                                              
    return np.concatenate((x_next, y_next, theta_next))





def rhs_linear_spatial_gaussian_phase(z,t,J,K,sigma,n,omega):
    
    #Instantiate
    x = z[0:n]
    y = z[n:2*n]
    theta = z[2*n:3*n]
    xd = x[:, np.newaxis] - x
    yd = y[:, np.newaxis] - y
    theta_d = theta[:, np.newaxis] - theta
    inverse_dist_sq = np.nan_to_num(1.0/((xd)**2 + (yd)**2))
    np.fill_diagonal(inverse_dist_sq, 0.0) ## correct 1 / d_ii = 1 / 0  "error"
    gaussian_dist = np.exp( -( xd**2 + yd**2) / (2.0*sigma) )
    
    x_rhs = -1*np.nan_to_num( xd*( (1+J*np.cos(theta_d)) - inverse_dist_sq)  ) 
    y_rhs = -1*np.nan_to_num( yd*( (1+J*np.cos(theta_d)) - inverse_dist_sq)  ) 
    theta_rhs = -1*K*np.sin(theta_d)*gaussian_dist
    
    
    #RHS    
    x_next = np.nan_to_num((1/float(n))*np.sum((1-np.eye(xd.shape[0]))*x_rhs, axis=1))
    y_next = np.nan_to_num((1/float(n))*np.sum((1-np.eye(xd.shape[0]))*y_rhs, axis=1))
    theta_next = np.nan_to_num(omega +  (1/float(n))*np.sum((1-np.eye(xd.shape[0]))*theta_rhs, axis=1))
                                              
    return np.concatenate((x_next, y_next, theta_next))



def rhs_linear_spatial_parabolic_phase(z,t,J,K,sigma,n,omega):
    
    #Instantiate
    x = z[0:n]
    y = z[n:2*n]
    theta = z[2*n:3*n]
    xd = x[:, np.newaxis] - x
    yd = y[:, np.newaxis] - y
    theta_d = theta[:, np.newaxis] - theta
    inverse_dist_sq = np.nan_to_num(1.0/((xd)**2 + (yd)**2))
    np.fill_diagonal(inverse_dist_sq, 0.0) ## correct 1 / d_ii = 0 / 1 eror
    parabolic_dist = 1 - ( xd**2 + yd**2) / (2.0*sigma)
    
    x_rhs = -1*np.nan_to_num( xd*( (1+J*np.cos(theta_d)) - inverse_dist_sq)  ) 
    y_rhs = -1*np.nan_to_num( yd*( (1+J*np.cos(theta_d)) - inverse_dist_sq)  ) 
    theta_rhs = -1*K*np.sin(theta_d)*parabolic_dist
    
    
    #RHS    
    x_next = np.nan_to_num((1/float(n))*np.sum((1-np.eye(xd.shape[0]))*x_rhs, axis=1))
    y_next = np.nan_to_num((1/float(n))*np.sum((1-np.eye(xd.shape[0]))*y_rhs, axis=1))
    theta_next = np.nan_to_num(omega +  (1/float(n))*np.sum((1-np.eye(xd.shape[0]))*theta_rhs, axis=1))
                                              
    return np.concatenate((x_next, y_next, theta_next))




def unpack(sols,n):
    """ I store the positions and phases of each swarmalator as a 3d vector z.
        Then the state vector for the systems is z[t][n]. This functions
        "unpacks" z[t][n] into x[t][n], y[t][n], theta[t][n]
    """

    T = len(sols)     #num timesteps
    x = np.array(np.zeros((T,n)))
    y = np.array(np.zeros((T,n)))
    theta = np.array(np.zeros((T,n)))
    
    for t in range(T):
        x[t] = sols[t, 0:n]
        y[t] = sols[t, n:2*n]
        theta[t] = sols[t,2*n:3*n]
    
    return [x,y,theta]



def find_gamma(phi):
    """ gamma = fraction of swarmalators exectuing full cycles """
    tolerance = 0.01
    gamma = 0.0
    for osc in range(phi.shape[1]):
        y = phi[:,osc]
        transient_index = int(0.5*len(y))
        temp = np.sin(y[transient_index:-1])
        temp = (max(temp) - min(temp)) / 2.0
        if temp > 1 - tolerance:
            gamma += 1.0
    return gamma / phi.shape[1]




def cart_to_polar(x,y):
    """ Convert cartesian coordiantes to polar """
    r,phi = np.zeros_like(x), np.zeros_like(x)
    for t in range(x.shape[0]):
        r[t,:] = np.sqrt( x[t,:]**2 + y[t,:]**2)
        phi[t,:] = np.arctan2(y[t,:], x[t,:])
    return r,phi


    
#def scatter_t(x,y,theta,t):
    """ Make a scatter plot of swarmalators at time t in the (x,y) plane 
        where swarmalators are colored according to their phase
        
        x = np.array, x[t][i] = x-coord of i-th swarmalator at time t
        y = np.array, y[t][i] = y-coord of i-th swarmalator at time t
        
    """
    
#    fig = plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
#    norm = mpl.colors.Normalize(vmin=0, vmax=2*np.pi)
#    cmap = cm.gist_rainbow
#    m = cm.ScalarMappable(norm=norm, cmap=cmap)
#    temp = m.to_rgba(np.mod(theta[t,:],2*np.pi))
#    plt.scatter(x[t,:],y[t,:], c = temp, s = 200, alpha = 0.9, marker = 'o',edgecolors='none',cmap = cm.gist_rainbow)


    
    
def find_vel_t(x,y,theta,dt):
    """ Finds the mean population velocity (really the speed I guess) at a given time t.
        Definition: v[t] =  1/N \sum_i  \sqrt  (dx_i / dt)^2 + (dy_i/dt)^2 + (dtheta_i /dt)^2
    """
    
    v = np.zeros((x.shape[0]-1,x.shape[1]))
    for i in range(x.shape[1]):
        v[:,i] = np.sqrt((np.diff(x[:,i])/dt)**2 + (np.diff(y[:,i])/dt)**2 + (np.diff(theta[:,i])/dt)**2)
    return v  

        

def find_W_plus(x,y,theta):
    """ Finds time series of order parameter W^+  = < e^{i*( phi_j + theta_j)} >_j -- see paper """
    numT, num_osc = x.shape
    W_plus = 1j*np.zeros(numT)
    for t in range(numT):
        phi = np.arctan2(x[t,:],y[t,:])
        W_plus[t] = np.sum(np.exp(1j*(phi + theta[t,:]))) / float(num_osc)
    return W_plus



def find_W_minus(x,y,theta):
    """ Finds time series of order parameter W^+  = < e^{i*( phi_j - theta_j)} >_j -- see paper """

    numT, num_osc = x.shape
    W_minus = 1j*np.zeros(numT)
    for t in range(numT):
        phi = np.arctan2(x[t,:],y[t,:])
        W_minus[t] = np.sum(np.exp(1j*(phi - theta[t,:]))) / float(num_osc)
    return W_minus



def find_X_minus(index,x,y,theta):
    numT, num_osc = x.shape
    X = 1j*np.zeros(numT)
    for t in range(numT):
        temp = 0.0+0*1j
        for j in range(num_osc):
            if j != index:
                dist = np.sqrt( (x[t,j]-x[t,index])**2 + (y[t,j]-y[t,index])**2) + 0*1j
                phij = np.arctan2(y[t,j],x[t,j])
                exp = np.exp(1j*(phij - theta[t,j]))
                temp += exp/dist
        X[t] = temp / (float(num_osc) + 0*1j)
    return X


def find_X_plus(index,x,y,theta):
    numT, num_osc = x.shape
    X = 1j*np.zeros(numT)
    for t in range(numT):
        temp = 0.0+0*1j
        for j in range(num_osc):
            if j != index:
                dist = np.sqrt( (x[t,j]-x[t,index])**2 + (y[t,j]-y[t,index])**2) + 0*1j
                phij = np.arctan2(y[t,j],x[t,j])
                exp = np.exp(1j*(phij + theta[t,j]))
                temp += exp/dist
        X[t] = temp / (float(num_osc) + 0*1j)
    return X



def find_Y_minus(index,x,y,theta):
    numT, num_osc = x.shape
    Y = 1j*np.zeros(numT)
    for t in range(numT):
        temp = 0.0+0*1j
        for j in range(num_osc):
            if j != index:
                dist = np.sqrt( (x[t,j]-x[t,index])**2 + (y[t,j]-y[t,index])**2) + 0*1j
                rj, phij = np.sqrt(x[t,j]**2 + y[t,j]**2), np.arctan2(y[t,j],x[t,j])
                exp = np.exp(1j*(phij - theta[t,j]))
                temp += (rj*exp)/dist
        Y[t] = temp / (float(num_osc) + 0*1j)
    return Y



def find_Y_plus(index,x,y,theta):
    numT, num_osc = x.shape
    Y = 1j*np.zeros(numT)
    for t in range(numT):
        temp = 0.0+0*1j
        for j in range(num_osc):
            if j != index:
                dist = np.sqrt( (x[t,j]-x[t,index])**2 + (y[t,j]-y[t,index])**2) + 0*1j
                rj, phij = np.sqrt(x[t,j]**2 + y[t,j]**2), np.arctan2(y[t,j],x[t,j])
                exp = np.exp(1j*(phij + theta[t,j]))
                temp += (rj*exp)/dist
        Y[t] = temp / (float(num_osc) + 0*1j)
    return Y




def find_Z(index,x,y,theta):
    numT, num_osc = x.shape
    Z = 1j*np.zeros(numT)
    for t in range(numT):
        temp = 0.0+0*1j
        for j in range(num_osc):
            if j != index:
                dist = np.sqrt( (x[t,j]-x[t,index])**2 + (y[t,j]-y[t,index])**2) + 0*1j
                temp = np.exp(1j*theta[t,j]) / dist
        Z[t] = temp / (float(num_osc) + 0*1j)
    return Z




def find_X_minus_final(x,y,theta):
    numT, num_osc = x.shape
    X = 1j*np.zeros(num_osc)
    t = -1
    for i in range(num_osc):
        temp = 0.0+0*1j
        for j in range(num_osc):
            if j != i:
                dist = np.sqrt( (x[t,j]-x[t,i])**2 + (y[t,j]-y[t,i])**2) + 0*1j
                phij = np.arctan2(y[t,j],x[t,j])
                exp = np.exp(1j*(phij - theta[t,j]))
                temp += exp/dist
        X[i] = temp / (float(num_osc) + 0*1j)
    return X


def find_X_plus_final(x,y,theta):
    numT, num_osc = x.shape
    X = 1j*np.zeros(num_osc)
    t = -1
    for i in range(num_osc):
        temp = 0.0+0*1j
        for j in range(num_osc):
            if j != i:
                dist = np.sqrt( (x[t,j]-x[t,i])**2 + (y[t,j]-y[t,i])**2) + 0*1j
                phij = np.arctan2(y[t,j],x[t,j])
                exp = np.exp(1j*(phij + theta[t,j]))
                temp += exp/dist
        X[i] = temp / (float(num_osc) + 0*1j)
    return X




def find_Y_minus_final(x,y,theta):
    numT, num_osc = x.shape
    Y = 1j*np.zeros(num_osc)
    t = -1
    for i in range(num_osc):
        temp = 0.0+0*1j
        for j in range(num_osc):
            if j != i:
                dist = np.sqrt( (x[t,j]-x[t,i])**2 + (y[t,j]-y[t,i])**2) + 0*1j
                rj, phij = np.sqrt(x[t,j]**2 + y[t,j]**2), np.arctan2(y[t,j],x[t,j])
                exp = np.exp(1j*(phij - theta[t,j]))
                temp += (rj*exp)/dist
        Y[i] = temp / (float(num_osc) + 0*1j)
    return Y


def find_Y_plus_final(x,y,theta):
    numT, num_osc = x.shape
    Y = 1j*np.zeros(num_osc)
    t = -1
    for i in range(num_osc):
        temp = 0.0+0*1j
        for j in range(num_osc):
            if j != i:
                dist = np.sqrt( (x[t,j]-x[t,i])**2 + (y[t,j]-y[t,i])**2) + 0*1j
                rj, phij = np.sqrt(x[t,j]**2 + y[t,j]**2), np.arctan2(y[t,j],x[t,j])
                exp = np.exp(1j*(phij + theta[t,j]))
                temp += (rj*exp)/dist
        Y[i] = temp / (float(num_osc) + 0*1j)
    return Y



def find_Z_final(x,y,theta):
    numT, num_osc = x.shape
    Z = 1j*np.zeros(num_osc)
    t= -1
    for i in range(num_osc):
        temp = 0.0+0*1j
        for j in range(num_osc):
            if j != i:
                dist = np.sqrt( (x[t,j]-x[t,i])**2 + (y[t,j]-y[t,i])**2) + 0*1j
                temp += np.exp(1j*theta[t,j]) / dist
        Z[i] = temp / (float(num_osc) + 0*1j)
    return Z


def unpack_timestep(sols,n):
    """ Same as unpack, expect at a given timestep in the 
        simualtion
    """
    T = len(sols)     #num timesteps
    x = np.array(np.zeros(n))
    y = np.array(np.zeros(n))
    theta = np.array(np.zeros(n))
    
    x = sols[0:n]
    y = sols[n:2*n]
    theta = sols[2*n:3*n]
    
    return [x.real,y.real,theta.real]


def find_W_plus_timestep(x,y,theta):
    """ Same as find_W_plus, except at a given timestep
        in the simulation """
    
    num_osc = len(x)
    phi = np.arctan2(x,y)
    W_plus = np.sum(np.exp(1j*(phi + theta))) / float(num_osc)
    return W_plus



def find_W_minus_timestep(x,y,theta):
    """ Same as find_W_minus, exceot at a given timestep
            in the simulation 
    """
    
    num_osc = len(x)
    phi = np.arctan2(x,y)
    W_minus = np.sum(np.exp(1j*(phi - theta))) / float(num_osc)
    return W_minus
