# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 17:26:29 2024

@author: Rudko
"""

import numpy as np
from scipy.optimize import minimize

# Define the Huber loss function
def huber_loss(params, z_data, M_obs, delta=1.0):
    # Unpack parameters
    a1, a2, b1, b2 = params
    
    # Compute predicted magnetization
    M_pred = get_magnetisation(z_data, params)
    
    # Compute the residuals
    residuals = M_obs - M_pred
    
    # Compute Huber loss
    loss = np.where(np.abs(residuals) <= delta,
                    0.5 * residuals ** 2,
                    delta * (np.abs(residuals) - 0.5 * delta))
    
    return np.mean(loss)

# Load your data (replace with actual file path)
data = np.loadtxt('your_data_file.csv', delimiter=',')
z_data = data[:, 0]  # Assuming first column is depth (z)
M_obs = data[:, 1]   # Assuming second column is observed magnetization

# Define bounds for parameters
bounds = [(0.1, 10),  # a1: lower bound of 0.1, upper bound of 10
          (0.1, 10),  # a2: lower bound of 0.1, upper bound of 10
          (-5, 5),    # b1: lower bound of -5, upper bound of 5
          (-5, 5)]    # b2: lower bound of -5, upper bound of 5

# Initial guess for parameters within the bounds
initial_params = [1.0, 1.0, 1.0, 1.0]

# Optimize the parameters using Huber loss with bounded optimization
result = minimize(huber_loss, initial_params, args=(z_data, M_obs), 
                  method='L-BFGS-B', bounds=bounds)

# Get optimized parameters
optimized_params = result.x
print('Optimized Parameters:', optimized_params)

# Compute the magnetization using the optimized parameters
M_optimized = get_magnetisation(z_data, optimized_params)

# Visualize the results
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(z_data, M_obs, 'bo', label='Observed Data')  # Plot the observed data
plt.plot(z_data, M_optimized, 'r-', label='Fitted Curve')  # Plot the fitted curve
plt.xlabel('Depth (z)')
plt.ylabel('Magnetization')
plt.legend()
plt.show()