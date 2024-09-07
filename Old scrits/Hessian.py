import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import approx_fprime
from mpl_toolkits.mplot3d import Axes3D

# Определение функций
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
    # Численное интегрирование
    result, _ = quad(integral, 0, z, args=(z, a1, a2, b1, b2, c1, c2))
    return result

def get_magnetisation_with_z(params_with_z):
    z, a1, a2, b1, b2, c1, c2 = params_with_z
    M = functional_integration(z, a1, a2, b1, b2, c1, c2)
    return np.tanh(M * 10**3)

# Численное вычисление градиента функции по параметрам
def compute_gradient(func, params, epsilon=1e-5):
    return approx_fprime(params, func, epsilon)

# Функция для вычисления численного Гессиана
def compute_hessian(func, params):
    n = len(params)
    hessian = np.zeros((n, n))
    epsilon = 1e-5

    # Вычисление градиента
    grad = compute_gradient(func, params)

    # Вычисление второй производной по каждой паре параметров
    for i in range(n):
        for j in range(i, n):
            params_ij = np.copy(params)
            params_ij[i] += epsilon
            params_ij[j] += epsilon
            hessian[i, j] = (func(params_ij) - grad[i] - grad[j] + func(params)) / (epsilon ** 2)
            if i != j:
                hessian[j, i] = hessian[i, j]  # Симметричность матрицы Гессиана

    return hessian

# Проверка, является ли Гессиан положительно полуопределенной матрицей
def is_positive_semi_definite(hessian_matrix):
    try:
        # Compute eigenvalues of the Hessian matrix
        eigenvalues = np.linalg.eigvals(hessian_matrix)
        # Check if all eigenvalues are non-negative
        return np.all(eigenvalues >= 0), eigenvalues
    except np.linalg.LinAlgError:
        # If computation fails, the matrix is not valid for this check
        return False, []

# Функция для проверки выпуклости в каждой области путем вычисления Гессиана
def check_convexity_by_region(z_grid, a1_grid, a2_grid, b1_grid, b2_grid, c1_grid, c2_grid):
    convex_regions = []
    non_convex_regions = []

    # Iterate over the grid points for each parameter
    for z in z_grid:
        for a1 in a1_grid:
            for a2 in a2_grid:
                for b1 in b1_grid:
                    for b2 in b2_grid:
                        for c1 in c1_grid:
                            for c2 in c2_grid:
                                # Обновляем параметры для текущей точки сетки
                                params = [z, a1, a2, b1, b2, c1, c2]
                                
                                # Вычисляем Гессиан для текущих параметров
                                hessian = compute_hessian(get_magnetisation_with_z, params)
                                
                                # Проверка, является ли Гессиан положительно полуопределенным
                                is_psd, eigenvalues = is_positive_semi_definite(hessian)
                                
                                # Сохранение региона на основе выпуклости
                                if is_psd:
                                    convex_regions.append((z, a1, a2, b1, b2, c1, c2, eigenvalues))
                                else:
                                    non_convex_regions.append((z, a1, a2, b1, b2, c1, c2, eigenvalues))
    
    return convex_regions, non_convex_regions

# Определение сетки параметров для анализа выпуклости
z_grid = np.linspace(0.1, 2.0, 5)
a1_grid = np.linspace(0.1, 2.0, 5)
a2_grid = np.linspace(0.1, 2.0, 5)
b1_grid = np.linspace(0.1, 2.0, 5)
b2_grid = np.linspace(0.1, 2.0, 5)
c1_grid = np.linspace(0.1, 4.0, 5)
c2_grid = np.linspace(0.1, 4.0, 5)

# Запуск проверки выпуклости для всех параметров
convex_regions, non_convex_regions = check_convexity_by_region(z_grid, a1_grid, a2_grid, b1_grid, b2_grid, c1_grid, c2_grid)

# Визуализация выпуклых и невыпуклых регионов
convex_points = np.array([(region[0], region[1], region[2]) for region in convex_regions])
non_convex_points = np.array([(region[0], region[1], region[2]) for region in non_convex_regions])
