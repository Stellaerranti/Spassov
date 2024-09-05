import numpy as np
from scipy.integrate import quad
from numpy import exp
from numpy import sin
from numpy import tanh
from scipy.optimize import minimize

def get_lock_depth_from_params(params):
    a1,a2,b1,b2 = params
    lock_in_depth = [(b1-2)/a1,(b1+2)/a1,(b2-2)/a2,(b2+2)/a2]
    return lock_in_depth

def get_params_from_depths(lock_in_depth):
    l0,l1,l2,l3 = lock_in_depth
    params = [4/(l1-l0),4/(l3-l2),2*(l1+l0)/(l1-l0),2*(l3+l2)/(l3-l2)]
    return params

def e(z):
    idx = np.where(np.isclose(depth, z))[0]
    idx = idx[0]

    return 1 - fraction_data[idx]

def H(z):
    # Ensure z is treated properly, even if it's an array
    return -np.tanh(((c2 + c1) * (z - (c2 + c1) / 2)) / (c2 - c1))

def l_diff(z, s, a1, b1, a2, b2):
    term1 = (a1 * np.exp(-a1 * s + b1)) / (1 + np.exp(-a1 * s + b1))**2
    term2 = (a2 * np.exp(-a2 * s + b2)) / (1 + np.exp(-a2 * s + b2))**2
    return e(z) * term1 + (1 - e(z)) * term2

def integral(s, z, a1, a2, b1, b2):
    # Convert z to float to ensure scalar use in quad
    return H(float(z) - s) * l_diff(float(z), s, a1, b1, a2, b2)

def functional_integration(z, a1, a2, b1, b2):
    # Use quad with scalar z, converting array inputs to floats
    result, _ = quad(lambda s: integral(s, float(z), a1, a2, b1, b2), 0, float(z))
    return result

# Vectorized version of get_magnetisation
def get_magnetisation(z, params):
    a1, a2, b1, b2 = params
    
    # Vectorize the integration function to handle array inputs
    vec_func_integration = np.vectorize(functional_integration)
    M = vec_func_integration(z, a1, a2, b1, b2)
    
    return np.tanh(M * 10**3)

def huber_loss(params, z_data, M_obs, delta=1.0):
    # Compute predicted magnetization
    M_pred = get_magnetisation(z_data, params)
    
    # Compute the residuals
    residuals = M_obs - M_pred
    
    # Compute Huber loss
    loss = np.where(np.abs(residuals) <= delta,
                    0.5 * residuals ** 2,
                    delta * (np.abs(residuals) - 0.5 * delta))
    
    return np.mean(loss)

def random_restarts_optimization(loss_function, z_data, M_obs, bounds, n_restarts=10):
    solutions = []
    for i in range(n_restarts):
        # Generate a random initial guess within the bounds
        random_initial = [np.random.uniform(low, high) for low, high in bounds]
        
        # Minimize the loss function
        result = minimize(loss_function, random_initial, args=(z_data, M_obs), method='L-BFGS-B', bounds=bounds)
        solutions.append(result.x)
        
        print(f"Iteration: {i+1}")
    
    return solutions

c1 = 73
c2 = 74.5
'''
depth = np.linspace(0.1,10,70)

params = get_params_from_depths([0.4,1,1.2,3.2])
M_obs = get_magnetisation(depth,params)
'''
_ , fraction_data = np.loadtxt('ez.txt', unpack = True)

polarity = np.loadtxt('Kuldara_polarity for Dima.txt')
M_obs = polarity[np.logical_not(np.logical_and(polarity[:,1]>-45,polarity[:,1]<45))]

depth, M_obs = M_obs[:,0],M_obs[:,1]

M_obs[M_obs > 0] = 1.
M_obs[M_obs < 0] = -1.

M_obs = np.array(M_obs)
depth = np.array(depth)

bounds = [(3., 7.),  # a1
          (0.5, 3.),  # a2
          (2.5, 4.8),    # b1
          (3., 5.)]    # b2

solutions = random_restarts_optimization(huber_loss, depth, M_obs, bounds, n_restarts=10)