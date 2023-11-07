import numpy as np
from scipy.stats import norm, cauchy, uniform

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

    # Find order parameters
    Wp, Wm = find_rainbow_order_parameters(z)

    # Find dot_x and dot_theta using the order parameters
    dot_x = nu + (J/2.0) * ((Wp * np.exp(-1j * (x + theta))).imag + (Wm * np.exp(-1j * (x - theta))).imag)
    dot_theta = omega + (K/2.0) * ((Wp * np.exp(-1j * (x + theta))).imag - (Wm * np.exp(-1j * (x - theta))).imag)
    
    # Concatenate the derivatives into a single array
    dot_z = np.concatenate([dot_x, dot_theta])

    return dot_z


def rhs_2Dswarmlator_torus(z, J, K, u, v, w):
    """Rhs of the 2D swarmalators model on a torus
    
    dot x_i = u_i + J/N * \sum_j sin(xj-xi) * cos(thetaj-thetai)
    dot y_i = v_i + J/N * \sum_j sin(yj-yi) * cos(thetaj-thetai)
    dot theta_i = w_i + K/N * \sum_j sin(thetaj-thetai) * (cos(xj-xi) + cos(yj-yi))

    Parameters:
    z (np.ndarray): concatenated arrays of (x, y, theta) positions and phases
    J (float): coupling term for x and y
    K (float): coupling term for theta
    u (np.ndarray): intrinsic frequencies for x
    v (np.ndarray): intrinsic frequencies for y
    w (np.ndarray): intrinsic frequencies for theta
    
    Returns:
    np.ndarray: Concatenated derivatives (dot_x, dot_y, dot_theta)
    """
    N = len(z) // 3
    x, y, theta = z[:N], z[N:2*N], z[2*N:]

    # Find the 2D rainbow order parameters
    Wp_x, Wm_x, Wp_y, Wm_y = find_2D_rainbow_order_parameters(x, y, theta)

    # Calculate the derivatives using the order parameters
    dot_x = u + (J/2.0) * ((Wp_x * np.exp(-1j * (x + theta))).imag + (Wm_x * np.exp(-1j * (x - theta))).imag)
    dot_y = v + (J/2.0) * ((Wp_y * np.exp(-1j * (y + theta))).imag + (Wm_y * np.exp(-1j * (y - theta))).imag)
    dot_theta = w + (K/2.0) * ((Wp_x * np.exp(-1j * (x + theta))).imag - (Wm_x * np.exp(-1j * (x - theta))).imag) \
                + (K/2.0) * ((Wp_y * np.exp(-1j * (y + theta))).imag - (Wm_y * np.exp(-1j * (y - theta))).imag)

    # Concatenate the derivatives into a single array
    dot_z = np.concatenate([dot_x, dot_y, dot_theta])

    return dot_z


def rhs_2d_swarmalator(z,J,K,omega):
    """
    z0 = [x0, y0, theta0], where x0[i], y0[i], theta0[i] gives the x[t=0], y[y=0], theta[t=0] of the i-th swarmalator
    t = time parameter (need this here to pass to python's odeint function)
    J = float
    K = float
    n = number of swarmalators
    omega = [omega1, omega2, ..., ] natural frequencies of swarmaltors
    """
    
    n = len(z) // 3
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
    - omega_dist (str): The type of distribution ('uniform_deterministic', 'uniform_random', 'gaussian_random', 'gaussian_deterministic', 'cauchy_deterministic', 'cauchy_random').
    - mu (float): The mean value of the distribution.
    - gamma (float): A parameter affecting the spread of the omega values.
    - n (int): Number of particles.

    Returns:
    - array_like: An array of natural frequencies for the oscillators.
    """
    
    valid_distributions = ['uniform_deterministic', 'uniform_random', 
                           'gaussian_random', 'gaussian_deterministic', 
                           'cauchy_deterministic', 'cauchy_random']
    
    if omega_dist not in valid_distributions:
        raise ValueError(f"Invalid omega_dist. Expected one of: {valid_distributions}, but got: {omega_dist}")
    
    if gamma < 0:
        raise ValueError(f"Invalid gamma. Gamma should be a non-negative number, but got: {gamma}")
    
    if n <= 0 or not isinstance(n, int):
        raise ValueError(f"Invalid number of particles. Expected a positive integer, but got: {n}")
    
    # Deterministic Distributions
    if omega_dist == 'uniform_deterministic':
        omega = np.linspace(mu - gamma, mu + gamma, n)
    elif omega_dist == 'gaussian_deterministic':
        linear_space = np.linspace(1/(n+1), n/(n+1), n)
        omega = norm.ppf(linear_space, loc=mu, scale=gamma)
    elif omega_dist == 'cauchy_deterministic':
        linear_space = np.linspace(1/(n+1), n/(n+1), n)
        omega = cauchy.ppf(linear_space, loc=mu, scale=gamma)
        
    # Random Distributions
    elif omega_dist == 'uniform_random':
        omega = np.random.uniform(mu - gamma, mu + gamma, n)
    elif omega_dist == 'gaussian_random':
        omega = np.random.normal(mu, gamma, n)
    elif omega_dist == 'cauchy_random':
        omega = cauchy.rvs(loc=mu, scale=gamma, size=n)
    
    else:
        omega = np.zeros(n)  # default to zeros if gamma is 0
                        
    return omega


def make_omega_nu(n, mu_omega, mu_nu, gamma_omega, gamma_nu, dist_type='cauchy_deterministic'):
    """
    Generate deterministic or random samples from specified distributions (uniform, Gaussian, or Cauchy)
    for omega and nu parameters using the inverse transform sampling method for deterministic samples
    or direct random sampling for random samples.

    The function creates a grid of points for deterministic sampling and applies the percent-point function
    (PPF) of the specified distribution to each point on the grid, ensuring structured coverage of the
    distribution's support. For random sampling, it draws samples directly from the specified distribution.

    Parameters:
    - n (int): Number of samples to generate for each dimension.
    - mu_omega (float): The location parameter for the omega distribution.
    - mu_nu (float): The location parameter for the nu distribution.
    - gamma_omega (float): The scale parameter for the omega distribution.
    - gamma_nu (float): The scale parameter for the nu distribution.
    - dist_type (str): The type of distribution from which to sample ('uniform_deterministic',
                       'gaussian_deterministic', 'cauchy_deterministic', 'uniform_random',
                       'gaussian_random', 'cauchy_random').

    Returns:
    - tuple: A tuple containing two flattened arrays, nu and omega, representing the samples
             from the specified distributions.

    Raises:
    - ValueError: If the dist_type provided is not supported.

    Example usage:
    nu, omega = make_omega_nu_deterministic(n=100, mu_omega=0, mu_nu=0, gamma_omega=1, gamma_nu=1, dist_type='uniform_deterministic')
    """

    if 'deterministic' in dist_type:
        # Calculate the grid size: sqrt(n) x sqrt(n)
        grid_size = int(np.sqrt(n))
        if grid_size ** 2 != n:
            raise ValueError('Deterministic sampling requires n to be a perfect square.')
            
        # Create a linear space for probabilities
        linear_space = np.linspace(0, 1, grid_size + 2)[1:-1]  # exclude 0 and 1 for stability
        
        # Create a meshgrid to have a grid of points
        Omega, Nu = np.meshgrid(linear_space, linear_space)
        flatten = True
    else:
        grid_size = n  # For random sampling, we directly sample n points
        flatten = False


    if dist_type == 'uniform_deterministic':
        # Deterministic sampling for uniform distribution
        omega_samples = uniform.ppf(Omega, loc=mu_omega - gamma_omega/2, scale=gamma_omega)
        nu_samples = uniform.ppf(Nu, loc=mu_nu - gamma_nu/2, scale=gamma_nu)
    elif dist_type == 'gaussian_deterministic':
        # Deterministic sampling for Gaussian distribution
        omega_samples = norm.ppf(Omega, loc=mu_omega, scale=gamma_omega)
        nu_samples = norm.ppf(Nu, loc=mu_nu, scale=gamma_nu)
    elif dist_type == 'cauchy_deterministic':
        # Deterministic sampling for Cauchy distribution
        omega_samples = cauchy.ppf(Omega, loc=mu_omega, scale=gamma_omega)
        nu_samples = cauchy.ppf(Nu, loc=mu_nu, scale=gamma_nu)
    elif dist_type == 'uniform_random':
        # Random sampling for uniform distribution
        omega_samples = uniform.rvs(loc=mu_omega - gamma_omega/2, scale=gamma_omega, size=n)
        nu_samples = uniform.rvs(loc=mu_nu - gamma_nu/2, scale=gamma_nu, size=n)
    elif dist_type == 'gaussian_random':
        # Random sampling for Gaussian distribution
        omega_samples = norm.rvs(loc=mu_omega, scale=gamma_omega, size=n)
        nu_samples = norm.rvs(loc=mu_nu, scale=gamma_nu, size=n)
    elif dist_type == 'cauchy_random':
        # Random sampling for Cauchy distribution
        omega_samples = cauchy.rvs(loc=mu_omega, scale=gamma_omega, size=n)
        nu_samples = cauchy.rvs(loc=mu_nu, scale=gamma_nu, size=n)
    else:
        raise ValueError("Unsupported distribution type")

    if flatten:
        omega_flat = omega_samples.flatten()
        nu_flat = nu_samples.flatten()
    else:
        omega_flat = omega_samples
        nu_flat = nu_samples

    return (nu_flat, omega_flat)

def make_cauchy(mu,gamma,n):
    return cauchy.rvs(loc=mu, scale=gamma, size=n)

def find_2D_rainbow_order_parameters(x, y, theta):
    """ Find the 2D rainbow order parameters

        Wpm_x := 1/N \sum_j exp(i (x_j pm theta_j))
        Wpm_y := 1/N \sum_j exp(i (y_j pm theta_j))
    """
    N = len(theta)
    Wp_x = np.mean(np.exp(1j * (x + theta)))
    Wm_x = np.mean(np.exp(1j * (x - theta)))
    Wp_y = np.mean(np.exp(1j * (y + theta)))
    Wm_y = np.mean(np.exp(1j * (y - theta)))
    return Wp_x, Wm_x, Wp_y, Wm_y


# Define a color class to print colorful text
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def fancy_print(message, color=Colors.OKGREEN, endchar='\n'):
    print(color + message + Colors.ENDC, end=endchar)

def header_print(string):
    length = max(50, len(string))
    print('\n')
    fancy_print('=' * length, Colors.HEADER)
    fancy_print(string, Colors.OKGREEN)
    fancy_print('=' * length, Colors.HEADER)


def rhs_swarmlator_deprecated(z, J, K, nu, omega):
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



