import numpy as np
from scipy.stats import norm, cauchy

def rhs_swarmlator(z, J, K, nu, omega):
    """Rhs of the 1D swarmalators model
    
    dot x_i = nu_i + J/N * \sum_j sin(xj-xi) * cos(thetaj-thetai)
    dot theta_i = omega_i + K/N * \sum_j sin(thetaj-thetai) * cos(xj-xi)

    Parameters:
    z (tuple): arrays (x, theta) representing positions and phases
    J (float): coupling term for x
    K (float): coupling term for theta
    nu (array): natural frequencies for x
    omega (array): natural frequencies for theta
    
    Returns:
    tuple: (dot_x, dot_theta) derivatives
    """

    N = len(z) // 2
    x, theta = z[:N], z[N:]

    # Broadcasting differences for all pairwise combinations
    delta_x = np.subtract.outer(x, x)
    delta_theta = np.subtract.outer(theta, theta)

    # Calculate interactions
    interaction_x = np.sin(delta_x) * np.cos(delta_theta)
    interaction_theta = np.sin(delta_theta) * np.cos(delta_x)

    # Summing over interactions and scaling by the coupling constants
    # Need to minus to get it to work right
    sum_x = -J / N * interaction_x.sum(axis=1)
    sum_theta = -K / N * interaction_theta.sum(axis=1)

    # Equations for x and theta
    dot_x = nu + sum_x
    dot_theta = omega + sum_theta
    dot_z = np.concatenate([dot_x, dot_theta])

    return dot_z

def rhs_kuramoto(z, K, omega):
    """Rhs of the kuramoto model
    
    dot theta_i = omega_i + K/N * \sum_j sin(thetaj-thetai)

    Using z as a represenation of the state for consistency with
    above rhs.

    Parameters:
    z (tuple): arrays (x, theta) representing positions and phases
    J (float): coupling term for x
    K (float): coupling term for theta
    nu (array): natural frequencies for x
    omega (array): natural frequencies for theta
    
    Returns:
    tuple: (dot_x, dot_theta) derivatives
    """

    Z = find_sync_order_parameter(z)
    dot_z = omega + (K*Z*np.exp(-1j*z)).imag

    return dot_z

def find_sync_order_parameter(z):
    """ Find the regular Kuramoto model sync parameter:

        Z := 1/N \sum_j exp( i theta_j) ) 

        where the state varible z := theta
        is used for consistency with other code
    """

    Z = np.mean(np.exp(1j*z))
    return Z

def find_rainbow_order_parameters(z):
    """ Find the rainbow order parameters

        Wpm := 1/N \sum_j exp( i (xj pm theta_j) ) 
    """
    n = len(z) // 2
    x, theta = z[:n], z[n:]
    Wp = np.mean(np.exp(1j*(x+theta)))
    Wm = np.mean(np.exp(1j*(x-theta)))
    return Wp, Wm


def euler(dt, z, f, args):
    """Implements one step of the euler method."""
    znext = z + dt * f(z, *args)
    return znext


def rk4(dt, z, f, args):
    """Implements one step of the RK4 method."""
    
    # Calculate the four k values
    k1 = dt * f(z, *args)
    k2 = dt * f(z + 0.5 * k1, *args)
    k3 = dt * f(z + 0.5 * k2, *args)
    k4 = dt * f(z + k3, *args)
    
    # Update z using the weighted average of the k values
    znext = z + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    
    return znext

def make_omega(mu, gamma, n, omega_dist='uniform_deterministic'):
    """
    Create the natural frequencies of the oscillators based on a specified distribution.

    Parameters:
    - omega_dist (str): The type of distribution ('uniform_deterministic', 'uniform_random', 'gaussian_random', 'gaussian_deterministic', 'cauchy_deterministic').
    - mu (float): The mean value of the distribution.
    - gamma (float): A parameter affecting the spread of the omega values.
    - n (int): Number of particles.

    Returns:
    - array_like: An array of natural frequencies for the oscillators.
    """
    
    valid_distributions = ['uniform_deterministic', 'uniform_random', 'gaussian_random', 'gaussian_deterministic', 'cauchy_deterministic']
    
    if omega_dist not in valid_distributions:
        raise ValueError(f"Invalid omega_dist. Expected one of: {valid_distributions}, but got: {omega_dist}")
    
    if gamma < 0:
        raise ValueError(f"Invalid gamma. Gamma should be a non-negative number, but got: {gamma}")
    
    if n <= 0 or not isinstance(n, int):
        raise ValueError(f"Invalid number of particles. Expected a positive integer, but got: {n}")
    
    if omega_dist == 'uniform_deterministic':
        omega = np.linspace(mu - gamma, mu + gamma, n)
    elif omega_dist == 'uniform_random':
        omega = np.random.uniform(mu - gamma, mu + gamma, n)
    elif omega_dist == 'gaussian_random':
        omega = np.random.normal(mu, gamma, n)
    elif omega_dist == 'gaussian_deterministic':
        linear_space = np.linspace(1/(n+1), n/(n+1), n)
        omega = norm.ppf(linear_space, loc=mu, scale=gamma)
    elif omega_dist == 'cauchy_deterministic':
        linear_space = np.linspace(1/(n+1), n/(n+1), n)
        omega = cauchy.ppf(linear_space, loc=mu, scale=gamma)
    else:
        omega = np.zeros(n)  # default to zeros if gamma is 0
                        
    return omega


