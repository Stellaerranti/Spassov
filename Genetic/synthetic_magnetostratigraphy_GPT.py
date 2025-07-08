
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit
from numpy import exp
from numpy import sin
from numpy import tanh
from scipy.interpolate import interp1d

def get_lock_depth_from_params(params):
    a1,a2,b1,b2 = params
    lock_in_depth = [(b1-2)/a1,(b1+2)/a1,(b2-2)/a2,(b2+2)/a2]
    return lock_in_depth

def get_params_from_depths(lock_in_depth):
    l0,l1,l2,l3 = lock_in_depth
    params = [4/(l1-l0),4/(l3-l2),2*(l1+l0)/(l1-l0),2*(l3+l2)/(l3-l2)]
    return params

def l (z,s,a1,b1,a2,b2):
    return e(z)/(1+exp(-a1*s+b1)) + (1-e(z))/(1+exp(-a2*s+b2))

def e(z):
    """Lithology ratio at arbitrary depth(s) z (scalar or array)."""
    return _e_interp(z)


def l_diff(z, s, a1, b1, a2, b2):
    term1 = (a1 * np.exp(-a1 * s + b1)) / (1 + np.exp(-a1 * s + b1))**2
    term2 = (a2 * np.exp(-a2 * s + b2)) / (1 + np.exp(-a2 * s + b2))**2
    return e(z) * term1 + (1 - e(z)) * term2

def H(z):
    # Ensure z is treated properly, even if it's an array
    return -np.tanh(((c2 + c1) * (z - (c2 + c1) / 2)) / (c2 - c1))

def integral(s, z, a1, a2, b1, b2):
    # Convert z to float to ensure scalar use in quad
    return H(float(z) - s) * l_diff(float(z), s, a1, b1, a2, b2)

def functional_integration(z, a1, a2, b1, b2):
    # Use quad with scalar z, converting array inputs to floats
    result, _ = quad(lambda s: integral(s, float(z), a1, a2, b1, b2), 0, float(z))
    return result

def get_magnetisation(z, params):
    a1, a2, b1, b2 = params

    # Vectorize the integration function to handle array inputs
    vec_func_integration = np.vectorize(functional_integration)
    M = vec_func_integration(z, a1, a2, b1, b2)

    return np.tanh(M * 10**3)

def get_polarity(z, params, threshold=0.5):
    """Convert magnetization to binary polarity."""
    M = get_magnetisation(z, params)
    return np.where(M > threshold, 1, np.where(M < -threshold, -1, 0))

#z = np.linspace(0, 10, 1000)  # Depth range
#c1 = 5
#c2 = 6
#lock_in_depths = [0.4, 1.0, 1.2, 3.2]  # l0, l1, l2, l3
#params = get_params_from_depths(lock_in_depths)

depth_obs , fraction_data = np.loadtxt('Spassovez.txt', unpack = True)

_e_interp = interp1d(
    depth_obs,
    1.0 - fraction_data,        # your e(z) values
    kind="nearest",             # or "linear" if you prefer
    bounds_error=False,
    fill_value=(1.0 - fraction_data[0],
                1.0 - fraction_data[-1]),
)

z = np.linspace(depth_obs[0],depth_obs[-1],depth_obs.shape[0])

c1 = 59.8
c2 = 60.1

lock_in_depths = [0.4, 1.6, 1.7, 3.2]  # l0, l1, l2, l3
params = get_params_from_depths(lock_in_depths)

polarity = get_polarity(z, params)

data_to_save = np.column_stack((z, fraction_data, polarity))
np.savetxt('synthetic_magnetostratigraphy.txt', data_to_save,
           header='depth(m) e(z) magnetization polarity', fmt='%.6f')

plt.figure(figsize=(10, 4))
plt.step(z, polarity, where='mid', color='k')
plt.xlabel('Depth (m)'); plt.ylabel('Polarity (+1/-1)')
plt.yticks([-1, 0, 1], ['Reversed', 'Inconclusive', 'Normal'])
plt.grid(alpha=0.3)
plt.show()