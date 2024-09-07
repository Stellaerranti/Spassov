import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
from numpy import exp, tanh

# Задаем значения параметров c1 и c2
c1 = 73
c2 = 74.5

# Чтение данных из файлов
depth_ez, fraction_data = np.loadtxt('ez.txt', unpack=True)
depth_kuldara, polarity_data = np.loadtxt('Kuldara_polarity for Dima.txt', unpack=True)

# Обработка данных для аппроксимации
def process_polarity_data(polarity_data):
    processed_data = []
    for value in polarity_data:
        if value > 45:
            processed_data.append(1)
        elif value < -45:
            processed_data.append(-1)
        else:
            processed_data.append(np.nan)  # неопределенные значения
    return np.array(processed_data)

processed_polarity_data = process_polarity_data(polarity_data)

# Функция e(z)
def e(z):
    idx = np.where(depth_ez == z)[0]
    if len(idx) > 0:
        return 1 - fraction_data[idx[0]]
    else:
        return 1

# Функция H(z)
def H(z):
    return -tanh(((c2 + c1) * (z - (c2 + c1) / 2)) / (c2 - c1))

# Функция l_diff
def l_diff(z, s, a1, b1, a2, b2):
    return e(z) * (a1 * exp(-a1 * s + b1)) / (1 + exp(-a1 * s + b1)) ** 2 + (1 - e(z)) * (a2 * exp(-a2 * s + b2)) / (1 + exp(-a2 * s + b2)) ** 2

# Интегрируемая функция
def integral(s, z, a1, a2, b1, b2):
    return H(z - s) * l_diff(z, s, a1, b1, a2, b2)

# Функция интегрирования
def functional_integration(z, a1, a2, b1, b2):
    return quad(integral, 0, z, args=(z, a1, a2, b1, b2))[0]

# Определение диапазона параметров
a1_range = np.linspace(0, 5, 10)
a2_range = np.linspace(0, 5, 10)
b1_range = np.linspace(-10, 10, 10)
b2_range = np.linspace(-10, 10, 10)
z_values = np.unique(depth_kuldara)

# Табулирование данных
grid_values = np.zeros((len(z_values), len(a1_range), len(a2_range), len(b1_range), len(b2_range)))

for i, z in enumerate(z_values):
    for j, a1 in enumerate(a1_range):
        for k, a2 in enumerate(a2_range):
            for l, b1 in enumerate(b1_range):
                for m, b2 in enumerate(b2_range):
                    grid_values[i, j, k, l, m] = functional_integration(z, a1, a2, b1, b2)

# Создание интерполятора
interpolator = RegularGridInterpolator((z_values, a1_range, a2_range, b1_range, b2_range), grid_values)

# Финальная функция для получения намагниченности с использованием интерполяции
def get_magnetisation_interpolated(z, params):
    a1, a2, b1, b2 = params
    points = np.array([[zi, a1, a2, b1, b2] for zi in z])
    M = interpolator(points)
    return tanh(M * 10**3)

# Функция ошибки
def objective_function(params):
    mask = ~np.isnan(processed_polarity_data)
    z_values_masked = depth_kuldara[mask]
    observed = processed_polarity_data[mask]
    predicted = get_magnetisation_interpolated(z_values_masked, params)
    return np.sum((observed - predicted) ** 2)

# Начальное предположение параметров
initial_params = [1, 1, 0, 0]

# Оптимизация параметров
result = minimize(objective_function, initial_params, method='Nelder-Mead')

# Оптимальные параметры
optimal_params = result.x
optimal_params
