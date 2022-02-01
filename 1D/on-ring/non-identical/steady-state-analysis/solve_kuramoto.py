import numpy as np

from scipy.integrate import quad


def g(omega):
    """ Natural frequency distribution """ 
    return 1.0 / ( np.pi*( 1 + omega**2 ) )

def integrand(omega, K, r):
    """ RHS of integral quation """
    return np.sqrt( 1 - (omega / (K*r))**2 )*g(omega)

def order_parameter(K):
    """ Analytic solution of order parameter 
        when g is Lorentzian
    """
    K_c = 2 / ( np.pi*g(0) )
    return np.sqrt( 1 - (K_c / K))

# Main
K = 2.5
r = order_parameter(K)
rhs = quad(integrand, - K*r, K*r, args=(K, r))[0]
print(f'K_c = { 2 / ( np.pi*g(0) ):.2f}')
print(f"K, r = {K}, {r:.2f}") 
print(f"RHS = {rhs:.2f}")
