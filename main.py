import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

# Speed of light (km/s)
c = 299792.458  

# Example supernova data (replace with real data)
z_data = np.array([0.01, 0.05, 0.1, 0.2, 0.3])
mu_obs = np.array([33.1, 36.7, 38.5, 40.2, 41.5])
sigma = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

# Hubble parameter for flat LCDM
def H(z, H0, Om):
    return H0 * np.sqrt(Om*(1+z)**3 + (1-Om))

# Luminosity distance
def dL(z, H0, Om):
    integral = quad(lambda zp: 1.0/H(zp, H0, Om), 0, z)[0]
    return (1+z) * c * integral

# Distance modulus
def mu_th(z, H0, Om):
    return 5*np.log10(dL(z, H0, Om)) + 25

# Chi-square function
def chi_square(params):
    H0, Om = params
    mu_model = np.array([mu_th(z, H0, Om) for z in z_data])
    return np.sum(((mu_obs - mu_model)/sigma)**2)

# Initial guess
initial_guess = [70, 0.3]

# Minimize chi-square
result = minimize(chi_square, initial_guess)

H0_best, Om_best = result.x

print("Best-fit H0:", H0_best)
print("Best-fit Omega_m:", Om_best)