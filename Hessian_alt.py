import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import approx_fprime
from mpl_toolkits.mplot3d import Axes3D

# Define functions for computations
def e(z):
    return np.sin(z)

def H(z, c1, c2):
    return -np.tanh(((c2 + c1) * (z - (c2 + c1) / 2)) / (c2 - c1))

def l_diff(z, s, a1, b1, a2, b2):
    return (e(z) * (a1 * np.exp(-a1 * s + b1)) / (1 + np.exp(-a1 * s + b1))**2 +
            (1 - e(z)) * (a2 * np.exp(-a2 * s + b2)) / (1 + np.exp(-a2 * s + b2))**2)

def integral(s, z, a1, a2, b1, b2, c1, c2):
    return H(z - s, c1, c2) * l_diff(z, s, a1, b1, a2, b2)

def functional_integration(z, a1, a2, b1, b2, c1, c2):
    result, _ = quad(integral, 0, z, args=(z, a1, a2, b1, b2, c1, c2))
    return result

def get_magnetisation_with_z(params):
    z, a1, a2, b1, b2, c1, c2 = params
    M = functional_integration(z, a1, a2, b1, b2, c1, c2)
    return np.tanh(M * 10**3)

# Compute gradient and Hessian
def compute_hessian(func, params):
    n = len(params)
    epsilon = 1e-5
    grad = approx_fprime(params, func, epsilon)
    hessian = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            params_ij = np.copy(params)
            params_ij[i] += epsilon
            params_ij[j] += epsilon
            hessian[i, j] = (func(params_ij) - grad[i] - grad[j] + func(params)) / (epsilon ** 2)
            if i != j:
                hessian[j, i] = hessian[i, j]
    return hessian

# Check if Hessian is positive semi-definite
def is_positive_semi_definite(hessian_matrix):
    try:
        eigenvalues = np.linalg.eigvals(hessian_matrix)
        return np.all(eigenvalues >= 0), eigenvalues
    except np.linalg.LinAlgError:
        return False, []

# Check convexity by region
def check_convexity_by_region(z_grid, a1_grid, a2_grid, b1_grid, b2_grid, c1_grid, c2_grid):
    convex_regions = []
    non_convex_regions = []
    for z in z_grid:
        for a1 in a1_grid:
            for a2 in a2_grid:
                for b1 in b1_grid:
                    for b2 in b2_grid:
                        for c1 in c1_grid:
                            for c2 in c2_grid:
                                params = [z, a1, a2, b1, b2, c1, c2]
                                hessian = compute_hessian(get_magnetisation_with_z, params)
                                is_psd, eigenvalues = is_positive_semi_definite(hessian)
                                if is_psd:
                                    convex_regions.append((z, a1, a2, b1, b2, c1, c2, eigenvalues))
                                else:
                                    non_convex_regions.append((z, a1, a2, b1, b2, c1, c2, eigenvalues))
    return convex_regions, non_convex_regions

# Define parameter grids
z_grid = np.linspace(0.1, 2.0, 5)
a1_grid = np.linspace(0.1, 2.0, 5)
a2_grid = np.linspace(0.1, 2.0, 5)
b1_grid = np.linspace(0.1, 2.0, 5)
b2_grid = np.linspace(0.1, 2.0, 5)
c1_grid = np.linspace(0.1, 4.0, 5)
c2_grid = np.linspace(0.1, 4.0, 5)

# Run convexity check
convex_regions, non_convex_regions = check_convexity_by_region(z_grid, a1_grid, a2_grid, b1_grid, b2_grid, c1_grid, c2_grid)

# Visualize convex and non-convex regions
convex_points = np.array([(region[0], region[1], region[2]) for region in convex_regions])
non_convex_points = np.array([(region[0], region[1], region[2]) for region in non_convex_regions])

# Define function to visualize regions for parameter pairs
def visualize_regions_by_pairs(convex_regions, non_convex_regions, param_pairs, fixed_params):
    for param1, param2 in param_pairs:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Prepare data for plotting
        convex_points = np.array([(region[fixed_params.index(param1)], region[fixed_params.index(param2)], region[0]) for region in convex_regions])
        non_convex_points = np.array([(region[fixed_params.index(param1)], region[fixed_params.index(param2)], region[0]) for region in non_convex_regions])

        # Plot convex regions
        if len(convex_points) > 0:
            ax.scatter(convex_points[:, 0], convex_points[:, 1], convex_points[:, 2], c='green', label='Convex Region')

        # Plot non-convex regions
        if len(non_convex_points) > 0:
            ax.scatter(non_convex_points[:, 0], non_convex_points[:, 1], non_convex_points[:, 2], c='red', label='Non-Convex Region')

        # Set plot labels
        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        ax.set_zlabel('Fixed Parameter')
        ax.set_title(f'Convex and Non-Convex Regions for {param1} and {param2}')

        plt.legend()
        plt.show()



# Define parameter pairs to visualize
#parameter_pairs = [('z', 'a1'), ('a1', 'a2'), ('b1', 'b2'), ('c1', 'c2')]

# Visualize regions for different parameter pairs
#visualize_regions_by_pairs(convex_regions, non_convex_regions, parameter_pairs, ['z', 'a1', 'a2', 'b1', 'b2', 'c1', 'c2'])