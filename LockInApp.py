import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt

import numpy as np
from scipy.integrate import quad
from numpy import exp
from numpy import sin
from numpy import tanh
from scipy.optimize import minimize
from scipy.optimize import dual_annealing
from scipy.optimize import differential_evolution
from PIL import Image, ImageTk

blue_color = plt.get_cmap('tab10')(0)

photo = None

depth_obs = None
fraction_data = None
polarity = None

selected_loss_function = "Huber loss"
selected_optimization = "L-BFGS-B"

c1 = 73.4
c2 = 74.


def create_image_label(frame, image_path, row, columnspan):
    global photo  # Используем глобальную переменную для хранения изображения

    # Загружаем изображение с помощью PIL
    image = Image.open(image_path)
    image = image.resize((200, 200))  # При необходимости изменяем размер изображения
    photo = ImageTk.PhotoImage(image)  # Сохраняем объект в глобальной переменной

    # Создаем Label для изображения и размещаем его на Frame с помощью grid
    label = tk.Label(frame, image=photo)
    label.photo = photo  # Сохраняем ссылку на изображение в атрибуте виджета
    label.grid(row=row, column=0, columnspan=columnspan, pady=10)  # Размещаем под кнопкой "Compute"

    return label
def parse_float(value):
    try:
        # Replace commas with periods
        value = value.replace(",", ".")
        return float(value)
    except ValueError:
        raise ValueError(f"Invalid input for float conversion: {value}")

def get_lock_depth_from_params(params):
    a1,a2,b1,b2 = params
    lock_in_depth = [(b1-2)/a1,(b1+2)/a1,(b2-2)/a2,(b2+2)/a2]
    return lock_in_depth

def get_params_from_depths(lock_in_depth):
    l0,l1,l2,l3 = lock_in_depth
    params = [4/(l1-l0),4/(l3-l2),2*(l1+l0)/(l1-l0),2*(l3+l2)/(l3-l2)]
    return params

def e(z):
    #return 0.4
    #return sin(z*16+1.5)/2 + 0.5
    idx = np.where(np.isclose(depth_obs, z))[0]
    idx = idx[0]
    
    return 1-fraction_data[idx]

def H(z):
    return -tanh(2*(z-c1)/(c2-c1) - 1)
    #return -tanh(((c2+c1)*(z-(c2+c1)/2))/(c2-c1))

def l (z,s,a1,b1,a2,b2):
    return e(z)/(1+exp(-a1*s+b1)) + (1-e(z))/(1+exp(-a2*s+b2))

def l_custom (e_value,s,a1,b1,a2,b2):
    return e_value/(1+exp(-a1*s+b1)) + (1-e_value)/(1+exp(-a2*s+b2))

def l_diff(z,s,a1,b1,a2,b2):
    return e(z) * (a1*exp(-a1*s+b1))/(1+exp(-a1*s+b1))**2 + (1-e(z)) * (a2*exp(-a2*s+b2))/(1+exp(-a2*s+b2))**2 

def integral(s,z,a1,a2,b1,b2):
    return H(z-s)*l_diff(z,s,a1,b1,a2,b2)

def functional_integration(z,a1,a2,b1,b2):        
    return quad(integral, 0, z, args=(z,a1,a2,b1,b2))[0]
'''
def functional_integration(z, a1, a2, b1, b2):
    # Use quad with scalar z, converting array inputs to floats
    result, _ = quad(lambda s: integral(s, float(z), a1, a2, b1, b2), 0, float(z))
    return result
'''
def get_magnetisation(z, params):
    a1, a2, b1, b2 = params
    
    # Vectorize the integration function to handle array inputs
    vec_func_integration = np.vectorize(functional_integration)
    M = vec_func_integration(z, a1, a2, b1, b2)
    
    return np.tanh(M * 10**3)

'''
def get_magnetisation(z,params):

    a1,a2,b1,b2 = params
    
    vec_expint = np.vectorize(functional_integration)
    M = vec_expint(z,a1,a2,b1,b2)
    
    return tanh(M*10**3)
'''
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

def mse_loss(params, z_data, M_obs):
    a1, a2, b1, b2 = params
    M_pred = get_magnetisation(z_data, params)
    residuals = M_obs - M_pred
    loss = residuals ** 2
    return np.mean(loss)

def hinge_loss(params, z_data, M_obs):
    a1, a2, b1, b2 = params
    M_pred = get_magnetisation(z_data, params)
    # Hinge loss assumes labels are -1 or 1
    loss = np.maximum(0, 1 - M_obs * M_pred)
    return np.mean(loss)

def on_optimization_change(event):
    selected_option = optimization_combobox.get()
    if selected_option == "Brute-Force":
        entry_calc_times_label.config(text="Step")
        entry_calc_times_label.grid(row=8, column=1, columnspan=2)
    else:
        entry_calc_times_label.config(text="Calculation times")    
        entry_calc_times_label.grid(row=8, column=1, columnspan=2)

# Функция для открытия окна настроек
def open_options():
    options_window = tk.Toplevel(root)
    options_window.title("Настройки")
    
    
    global loss_function_combobox
    tk.Label(options_window, text="Функция потерь:").grid(row=0, column=0)
    loss_function_combobox = ttk.Combobox(options_window, values=["Huber loss", "MSE", "Hinge loss"])
    loss_function_combobox.grid(row=0, column=1)
    loss_function_combobox.set("Huber loss")

    tk.Label(options_window, text="Оптимизация:").grid(row=1, column=0)
    global optimization_combobox
    optimization_combobox = ttk.Combobox(options_window, values=["L-BFGS-B", "Brute-Force", "Dual anneling"])
    optimization_combobox.grid(row=1, column=1)
    optimization_combobox.set("L-BFGS-B")
    
    optimization_combobox.bind("<<ComboboxSelected>>", on_optimization_change)


    def apply_settings():
        global selected_loss_function
        global selected_optimization
        
        selected_loss_function = loss_function_combobox.get()
        selected_optimization = optimization_combobox.get()
        #messagebox.showinfo("Настройки", f"Выбрана функция потерь: {selected_loss_function}\nВыбран метод оптимизации: {selected_optimization}")

    apply_button = tk.Button(options_window, text="Применить", command=apply_settings)
    apply_button.grid(row=2, columnspan=2, pady=10)

def adjust_param_ranges(param_low, param_high, epsilon=1e-1):
    # Проверка и коррекция границ параметров
    if param_low > param_high:
        param_low, param_high = param_high, param_low
    elif param_low == param_high:
        param_high += epsilon  # Раздвинуть границы, если они совпадают
    return param_low, param_high

def convert_depths_to_params_ranges(depth_ranges):
    # Извлечение границ глубин
    l0_low, l0_high, l1_low, l1_high, l2_low, l2_high, l3_low, l3_high = depth_ranges
    
    # Рассчитываем параметры для границ глубин
    params_low = get_params_from_depths([l0_low, l1_low, l2_low, l3_low])
    params_high = get_params_from_depths([l0_high, l1_high, l2_high, l3_high])
    
    # Извлекаем диапазоны параметров
    a1_low, a2_low, b1_low, b2_low = params_low
    a1_high, a2_high, b1_high, b2_high = params_high
    
    # Коррекция диапазонов параметров
    a1_low, a1_high = adjust_param_ranges(a1_low, a1_high)
    a2_low, a2_high = adjust_param_ranges(a2_low, a2_high)
    b1_low, b1_high = adjust_param_ranges(b1_low, b1_high)
    b2_low, b2_high = adjust_param_ranges(b2_low, b2_high)
    
    # Возвращаем скорректированные диапазоны параметров
    return [
         (a1_low, a1_high),
         (a2_low, a2_high),
         (b1_low, b1_high),
         (b2_low, b2_high)]
    

# Функция для открытия и загрузки файла как массив NumPy
def load_file():
    global depth_obs, fraction_data, polarity
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
    if file_path:
        try:
            depth_obs, fraction_data, polarity = np.loadtxt(file_path,unpack = True)
            #fraction_data = 1 - fraction_data
            #messagebox.showinfo("Файл загружен","Файл загружен")
            process_loaded_data(depth_obs, fraction_data, polarity)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {e}")

def process_loaded_data(depth_obs, fraction_data, polarity):
    for ax in axs:
        ax.clear()
        
    axs[0].set_ylim(depth_obs[-1],depth_obs[0])
    axs[0].set_title("Field polarity", fontsize=8)
    axs[0].set_ylim(depth_obs[-1],depth_obs[0])
    axs[0].set_ylabel('Depth')


    axs[1].plot(fraction_data,depth_obs, color = blue_color)
    axs[1].set_title("e(z)", fontsize=8)
    axs[1].set_ylim(depth_obs[-1],depth_obs[0])
    axs[1].set_xlim(0,1.1)

    axs[2].plot(polarity, depth_obs, color = blue_color)
    axs[2].set_title("Observed polarity", fontsize=8)
    axs[2].set_ylim(depth_obs[-1],depth_obs[0])
    
    axs[3].set_title("Modeled polarity", fontsize=8)
    axs[3].set_ylim(depth_obs[-1],depth_obs[0])
    
    axs[4].set_title("Lock-in-function", fontsize=8)
    axs[4].set_ylim(depth_obs[-1],depth_obs[0])
    
    for ax in axs[1:]:
        ax.tick_params(left=False)
        ax.set_yticklabels([])

    figure.subplots_adjust(wspace=0.1)
    canvas.draw()
    
    for ax in axs2:
        ax.clear()
    
    axs2[0].set_ylim(depth_obs[-1],depth_obs[0])
    axs2[0].set_title("Field polarity", fontsize=8)
    axs2[0].set_ylim(depth_obs[-1],depth_obs[0])
    axs2[0].set_ylabel('Depth')


    axs2[1].plot(fraction_data,depth_obs, color = blue_color)
    axs2[1].set_title("e(z)", fontsize=8)
    axs2[1].set_ylim(depth_obs[-1],depth_obs[0])
    axs2[1].set_xlim(0,1.1)

    axs2[2].plot(polarity, depth_obs, color = blue_color)
    axs2[2].set_title("Observed polarity", fontsize=8)
    axs2[2].set_ylim(depth_obs[-1],depth_obs[0])
    
    axs2[3].set_title("Modeled polarity", fontsize=8)
    axs2[3].set_ylim(depth_obs[-1],depth_obs[0])
    
    '''
    axs2[4].set_title("Lock-in-function", fontsize=8)
    #axs2[4].set_ylim(depth_obs[-1],depth_obs[0])
    '''
    
    
    
    for ax in axs2[1:]:
        ax.tick_params(left=False)
        ax.set_yticklabels([])

    figure2.subplots_adjust(wspace=0.1)
    canvas2.draw()


def update_graphs(params):

    a1,a2,b1,b2 = params    

    for ax in axs:
        ax.clear()

    axs[0].plot(H(depth_obs), depth_obs, color = blue_color)
    axs[0].set_title("Field polarity", fontsize=8)
    axs[0].set_ylim(depth_obs[-1],depth_obs[0])
    axs[0].set_xlim(-1.1,1.1)
    axs[0].set_ylabel('Depth')


    axs[1].plot(fraction_data,depth_obs, color = blue_color)
    axs[1].set_title("e(z)", fontsize=8)
    axs[1].set_ylim(depth_obs[-1],depth_obs[0])
    axs[1].set_xlim(0,1.1)

    axs[2].plot(polarity, depth_obs, color = blue_color)
    axs[2].set_title("Observed polarity", fontsize=8)
    axs[2].set_ylim(depth_obs[-1],depth_obs[0])
    
    axs[3].plot(get_magnetisation(depth_obs, params),depth_obs, color = blue_color)
    axs[3].set_title("Modeled polarity", fontsize=8)
    axs[3].set_ylim(depth_obs[-1],depth_obs[0])


    #params = get_params_from_depths([d0,d1,d2,d3])
    
    axs[4].clear()
    axs[4].plot(l_custom(0.9,np.linspace(0,10,50),params[0],params[2],params[1],params[3]),np.linspace(0,10,50), label = 'e(z) = 0.9')
    axs[4].plot(l_custom(0.5,np.linspace(0,10,50),params[0],params[2],params[1],params[3]),np.linspace(0,10,50), label = 'e(z) = 0.5')
    axs[4].plot(l_custom(0.1,np.linspace(0,10,50),params[0],params[2],params[1],params[3]),np.linspace(0,10,50), label = 'e(z) = 0.1')
    axs[4].legend(loc = 'lower left')
    axs[4].set_title("Lock-in-function", fontsize=8)

    if axs[1].get_lines():
        yticks = axs[0].get_yticks() 
        
        tick_interval = yticks[1] - yticks[0]
        axs[4].yaxis.set_major_locator(MultipleLocator(tick_interval))
        
        set_visual_scale([axs[4]], axs[0], tick_interval)
        
        
        axs[4].invert_yaxis()
        #axs2[4].set_yticks()
        
    else:
        axs[4].set_ylim(10,0)

    for ax in axs[1:]:
        ax.tick_params(left=False)
        ax.set_yticklabels([])

    figure.subplots_adjust(wspace=0.1)
    canvas.draw()

def random_restarts_optimization(loss_function, initial_guess, bounds, depth, M_obs, n_restarts=10):
    solutions = []
    for i in range(n_restarts):
        
        # Generate a random initial guess within the bounds
        random_initial = [np.random.uniform(low, high) for low, high in bounds]
        result = minimize(loss_function, random_initial, args=(depth, M_obs), method='L-BFGS-B', bounds=bounds)
        solutions.append(result.x)
        print(f"Iteration: {i+1}")
    return solutions

def random_restarts_optimization_dual_anneling(loss_function, initial_guess, bounds, depth, M_obs, n_restarts=10):
    solutions = []
    for i in range(n_restarts):
        
        # Generate a random initial guess within the bounds
        random_initial = [np.random.uniform(low, high) for low, high in bounds]
        result = dual_annealing(loss_function, random_initial, args=(depth, M_obs))
        
        solutions.append(result.x)
        print(f"Iteration: {i+1}")
    return solutions

def compute():
    global c1,c2
    
    d0_low = parse_float(entry_d0_low.get())
    d0_high = parse_float(entry_d0_high.get())
    d1_low = parse_float(entry_d1_low.get())
    d1_high = parse_float(entry_d1_high.get())
    d2_low = parse_float(entry_d2_low.get())
    d2_high = parse_float(entry_d2_high.get())
    d3_low = parse_float(entry_d3_low.get())
    d3_high = parse_float(entry_d3_high.get())
    c1 = parse_float(entry_с1.get())
    c2 = parse_float(entry_с2.get())
    calculation_times = int(entry_calc_times.get())
    initial_params = [1.0, 1.0, 1.0, 1.0]
    
    #bounds = np.column_stack((get_params_from_depths([d0_low,d1_low,d2_low,d3_low]),get_params_from_depths([d0_high,d1_high,d2_high,d3_high])))
    depth_ranges = (d0_low,d0_high,d1_low,d1_high,d2_low,d2_high,d3_low,d3_high)
    
    bounds = convert_depths_to_params_ranges(depth_ranges)
    '''
    bounds = [(d0_low, d0_high),  # a1
          (d1_low, d1_high),  # a2
          (d2_low, d2_high),    # b1
          (d3_low, d3_high)] 
    '''
    try:
        #selected_optimization = optimization_combobox.get()
        #selected_loss_function = loss_function_combobox.get()
        
        if(selected_loss_function == "Huber loss"):
            loss_function = huber_loss
        elif(selected_loss_function == "Hinge loss"):
            loss_function = hinge_loss
        elif(selected_loss_function == "MSE"):
            loss_function = mse_loss
            
        
        
        if (selected_optimization == 'L-BFGS-B'):
            
            solutions = random_restarts_optimization(loss_function, initial_params, bounds, depth_obs, polarity, n_restarts=calculation_times)
            
            # Очистка предыдущих решений
            for widget in solution_frame.winfo_children():
                widget.destroy()
    
            # Замена placeholder на кнопки решений
            for i, (sols) in enumerate(solutions):
                d0,d1,d2,d3 = get_lock_depth_from_params(sols)
                
                button = tk.Button(solution_frame, text=f"Решение {i+1}: d0={d0:.2f}, d1={d1:.2f}, d2={d2:.2f}, d3={d3:.2f}",
                                   command=lambda d0=d0, d1=d1, d2=d2, d3=d3: update_graphs(sols))
                button.pack()
                
        elif(selected_optimization == "Dual anneling"):
            
            solutions = random_restarts_optimization_dual_anneling(loss_function, initial_params, bounds, depth_obs, polarity, n_restarts=calculation_times)
            
            # Очистка предыдущих решений
            for widget in solution_frame.winfo_children():
                widget.destroy()
    
            # Замена placeholder на кнопки решений
            for i, (sols) in enumerate(solutions):
                d0,d1,d2,d3 = get_lock_depth_from_params(sols)
                
                button = tk.Button(solution_frame, text=f"Решение {i+1}: d0={d0:.2f}, d1={d1:.2f}, d2={d2:.2f}, d3={d3:.2f}",
                                   command=lambda d0=d0, d1=d1, d2=d2, d3=d3: update_graphs(sols))
                button.pack()
        
        elif(selected_optimization == "Brute-Force"):
            
            val_step = calculation_times
            
            d0,d1,d2,d3 = d0_low,d1_low,d2_low,d3_low
            
            res = []
            
            while d0 < d0_high:
                while d1 < d1_high:
                    while d2 < d2_high:
                        while d3 < d3_high:
                            res.append([d0,d1,d2,d3,mse_loss(get_params_from_depths([d0,d1,d2,d3],depth_obs,polarity))])                      
    
                            d3 = d3 + val_step
                        d2 = d2 + val_step
                    d1 = d1 + val_step
                d0 = d0 + val_step
                
            res = np.asarray(res)
            
            res_sorted = np.sort(res)
            
            for i in range(20):
                d0,d1,d2,d3,_ = res_sorted[i]
                
                sols = get_params_from_depths([d0,d1,d2,d3])
                
                button = tk.Button(solution_frame, text=f"Решение {i+1}: d0={d0:.2f}, d1={d1:.2f}, d2={d2:.2f}, d3={d3:.2f}",
                                   command=lambda d0=d0, d1=d1, d2=d2, d3=d3: update_graphs(sols))
                button.pack()
            
    except ValueError:
        messagebox.showerror("Ошибка", "Введите корректные числовые параметры.")

def direct_comptue():
    
    global c1,c2
    
    d0 = parse_float(entry_d0.get())
    d1 = parse_float(entry_d1.get())
    d2 = parse_float(entry_d2.get())
    d3 = parse_float(entry_d3.get())
    
    c1 = parse_float(entry_c1_d.get())
    c2 = parse_float(entry_c2_d.get())
    
    params = get_params_from_depths([d0,d1,d2,d3])
    
    try:
        #for ax in axs2:
         #   ax.clear()
        #M_modeled = get_magnetisation(depth_obs, params)
        
        #H_obs = H(depth_obs)
        
        axs2[0].clear()
        axs2[0].plot(H(depth_obs), depth_obs, color = blue_color)
        axs2[0].set_title("Field polarity", fontsize=8)
        axs2[0].set_ylim(depth_obs[-1],depth_obs[0])
        axs2[0].set_xlim(-1.1,1.1)
        axs2[0].set_ylabel('Depth')
        
        #axs2[1].plot(fraction_data,depth_obs)
        #axs2[1].set_title("e(z)", fontsize=8)
        #axs2[1].set_ylim(depth_obs[-1],depth_obs[0])
        
        axs2[3].clear()
        axs2[3].plot(get_magnetisation(depth_obs, params),depth_obs, color = blue_color)
        axs2[3].set_title("Modeled polarity", fontsize=8)
        axs2[3].set_ylim(depth_obs[-1],depth_obs[0])
        
        '''
        axs2[4].clear()
        axs2[4].plot(l_custom(0.9,np.linspace(0,10,50),params[0],params[2],params[1],params[3]),np.linspace(0,10,50), label = 'e(z) = 0.9')
        axs2[4].plot(l_custom(0.5,np.linspace(0,10,50),params[0],params[2],params[1],params[3]),np.linspace(0,10,50), label = 'e(z) = 0.5')
        axs2[4].plot(l_custom(0.1,np.linspace(0,10,50),params[0],params[2],params[1],params[3]),np.linspace(0,10,50), label = 'e(z) = 0.1')
        
        axs2[4].set_title("Lock-in-function", fontsize=8)
        #axs2[3].set_ylim(depth_obs[-1],depth_obs[0])
        axs2[4].set_ylim(10,0)
        axs2[4].legend(loc = 'lower left')        
        
        if axs2[1].get_lines():
            yticks = axs2[0].get_yticks() 
            
            tick_interval = yticks[1] - yticks[0]
            axs2[4].yaxis.set_major_locator(MultipleLocator(tick_interval))
            
            set_visual_scale([axs2[4]], axs2[0], tick_interval)
            
            
            axs2[4].invert_yaxis()
            #axs2[4].set_yticks()
            
        else:
            axs2[4].set_ylim(10,0)
        '''
        for ax in axs2[1:]:
            ax.tick_params(left=False)
            ax.set_yticklabels([])

        figure2.subplots_adjust(wspace=0.1)
        canvas2.draw()
        
        
    except ValueError:
        messagebox.showerror("Ошибка", "Введите корректные числовые параметры.")
        
def field_change_forvard(*args):
    
    global c1,c2  
    
    
    try:
        c1 = parse_float(entry_c1_d.get())
        c2 = parse_float(entry_c2_d.get())
        
        axs2[0].clear()
        axs2[0].plot(H(depth_obs), depth_obs, color = blue_color)
        axs2[0].set_title("Field polarity", fontsize=8)
        axs2[0].set_ylim(depth_obs[-1],depth_obs[0])
        axs2[0].set_xlim(-1.1,1.1)
        axs2[0].set_ylabel('Depth')
        
        for ax in axs2[1:]:
            ax.tick_params(left=False)
            ax.set_yticklabels([])
        
        figure2.subplots_adjust(wspace=0.1)
        canvas2.draw()
        
    except (ValueError, ZeroDivisionError) as e:
        return   

def field_change_inverse(*args):
    
    global c1,c2  
    
    
    try:
        c1 = parse_float(entry_с1.get())
        c2 = parse_float(entry_с2.get())
        
        axs[0].clear()
        axs[0].plot(H(depth_obs), depth_obs, color = blue_color)
        axs[0].set_title("Field polarity", fontsize=8)
        axs[0].set_ylim(depth_obs[-1],depth_obs[0])
        axs[0].set_xlim(-1.1,1.1)
        axs[0].set_ylabel('Depth')        
        
        for ax in axs[1:]:
            ax.tick_params(left=False)
            ax.set_yticklabels([])
        
        figure.subplots_adjust(wspace=0.1)
        canvas.draw()
        
    except (ValueError, ZeroDivisionError) as e:
        return   

def depth_changed_forvard(*args):
    
    try:        
                
        d0 = parse_float(entry_d0.get())
        d1 = parse_float(entry_d1.get())
        d2 = parse_float(entry_d2.get())
        d3 = parse_float(entry_d3.get())
        
        if d3 < d2:
            return 
        
        params = get_params_from_depths([d0,d1,d2,d3])
        
        axs2[4].clear()
        axs2[4].plot(l_custom(0.9,np.linspace(0,10,50),params[0],params[2],params[1],params[3]),np.linspace(0,10,50), label = 'e(z) = 0.9')
        axs2[4].plot(l_custom(0.5,np.linspace(0,10,50),params[0],params[2],params[1],params[3]),np.linspace(0,10,50), label = 'e(z) = 0.5')
        axs2[4].plot(l_custom(0.1,np.linspace(0,10,50),params[0],params[2],params[1],params[3]),np.linspace(0,10,50), label = 'e(z) = 0.1')
        axs2[4].legend(loc = 'lower left')
        axs2[4].set_title("Lock-in-function", fontsize=8)
        #axs2[3].set_ylim(depth_obs[-1],depth_obs[0])
        
        if axs2[1].get_lines():
            yticks = axs2[0].get_yticks() 
            
            tick_interval = yticks[1] - yticks[0]
            axs2[4].yaxis.set_major_locator(MultipleLocator(tick_interval))
            
            set_visual_scale([axs2[4]], axs2[0], tick_interval)
            
            
            axs2[4].invert_yaxis()
            #axs2[4].set_yticks()
            
        else:
            axs2[4].set_ylim(10,0)
        

        for ax in axs2[1:]:
            ax.tick_params(left=False)
            ax.set_yticklabels([])
        
        figure2.subplots_adjust(wspace=0.1)
        canvas2.draw()
        
    except (ValueError, ZeroDivisionError) as e:
        return 
    
def set_visual_scale(axes, ref_ax, ref_interval):
    """
    Adjusts only the ymax of the given axes to have identical visual tick intervals based on the reference axis.
    The ymin remains unchanged. Supports reversed scale on the reference axis.
    
    Parameters:
    - axes: List of axes to adjust.
    - ref_ax: The reference axis to base the visual interval on.
    - ref_interval: The tick interval on the reference axis.
    """
    # Get the pixel height of one interval on the reference axis
    ref_figure_height = ref_ax.get_window_extent().height
    ref_ylim = ref_ax.get_ylim()
    ref_range = abs(ref_ylim[1] - ref_ylim[0])  # Use absolute value to handle reversed axes
    ref_pixels_per_unit = ref_figure_height / ref_range
    
    # Calculate the pixel height of one tick interval on the reference axis
    ref_pixel_interval = ref_pixels_per_unit * ref_interval
    
    for ax in axes:
        if ax == ref_ax:
            continue
        
        # Get the current axis height in pixels
        ax_figure_height = ax.get_window_extent().height
        
        # Calculate the new ymax value to match the reference visual scale
        new_range = ax_figure_height / ref_pixels_per_unit
        
        # Calculate the new ymax, considering if the reference axis is reversed
        ax_ymin, ax_ymax = ax.get_ylim()
        if ref_ylim[0] < ref_ylim[1]:  # Reference axis is not reversed
            new_ymax = ax_ymin + new_range
        else:  # Reference axis is reversed
            new_ymax = ax_ymin - new_range
        
        # Ensure new_ymax is positive if required
        if new_ymax < 0:
            new_ymax = abs(new_ymax)
        
        # Set new y-limits with unchanged ymin and adjusted ymax
        ax.set_ylim(0, new_ymax)

# Создание главного окна
root = tk.Tk()
root.title("Loess - paleosol magnitization")

menu_bar = tk.Menu(root)
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Open", command=load_file)
menu_bar.add_cascade(label="File", menu=file_menu)

options_menu = tk.Menu(menu_bar, tearoff=0)
options_menu.add_command(label="Settings", command=open_options)
menu_bar.add_cascade(label="Options", menu=options_menu)
root.config(menu=menu_bar)

# Create Notebook (tabs)
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill='both')

# First Tab
tab1 = tk.Frame(notebook)
notebook.add(tab1, text='Inverse')

left_frame = tk.Frame(tab1)
left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

tk.Label(left_frame, text = "Depths ranges:").grid(row = 0, column= 0, columnspan = 2)

tk.Label(left_frame, text="d0 low:").grid(row=1, column=0)
entry_d0_low = tk.Entry(left_frame)
entry_d0_low.grid(row=1, column=1)
entry_d0_low.insert(0, "1.0")

tk.Label(left_frame, text="d0 high:").grid(row=1, column=2)
entry_d0_high = tk.Entry(left_frame)
entry_d0_high.grid(row=1, column=3)
entry_d0_high.insert(0, "1.0")

tk.Label(left_frame, text="d1 low:").grid(row=2, column=0)
entry_d1_low = tk.Entry(left_frame)
entry_d1_low.grid(row=2, column=1)
entry_d1_low.insert(0, "1.0")

tk.Label(left_frame, text="d1 high:").grid(row=2, column=2)
entry_d1_high = tk.Entry(left_frame)
entry_d1_high.grid(row=2, column=3)
entry_d1_high.insert(0, "1.0")

tk.Label(left_frame, text="d2 low:").grid(row=3, column=0)
entry_d2_low = tk.Entry(left_frame)
entry_d2_low.grid(row=3, column=1)
entry_d2_low.insert(0, "1.0")

tk.Label(left_frame, text="d2 high:").grid(row=3, column=2)
entry_d2_high = tk.Entry(left_frame)
entry_d2_high.grid(row=3, column=3)
entry_d2_high.insert(0, "1.0")

tk.Label(left_frame, text="d3 low:").grid(row=4, column=0)
entry_d3_low = tk.Entry(left_frame)
entry_d3_low.grid(row=4, column=1)
entry_d3_low.insert(0, "1.0")

tk.Label(left_frame, text="d3 high:").grid(row=4, column=2)
entry_d3_high = tk.Entry(left_frame)
entry_d3_high.grid(row=4, column=3)
entry_d3_high.insert(0, "1.0")

ez_button = tk.Button(left_frame, text="Import e(z)", command=load_file)
ez_button.grid(row = 5, column = 0, columnspan = 2 )

tk.Label(left_frame, text = "Field parametrs:").grid(row=6, column=0, columnspan = 2)


tk.Label(left_frame, text="с1").grid(row=7, column=0)
c1_var_inverse = tk.StringVar()
entry_с1 = tk.Entry(left_frame, textvariable = c1_var_inverse)
entry_с1.grid(row=7, column=1)
entry_с1.insert(0, "1.0")

tk.Label(left_frame, text="с2").grid(row=7, column=2)
c2_var_inverse = tk.StringVar()
entry_с2 = tk.Entry(left_frame, textvariable = c2_var_inverse)
entry_с2.grid(row=7, column=3)
entry_с2.insert(0, "1.0")

c1_var_inverse.trace_add("write", field_change_inverse)
c2_var_inverse.trace_add("write", field_change_inverse)

entry_calc_times_label = tk.Label(left_frame, text="Calculation times:")
entry_calc_times_label.grid(row=8, column=1, columnspan=2)

entry_calc_times = tk.Entry(left_frame)
entry_calc_times.grid(row=8, column=3)
entry_calc_times.insert(0, "10")

compute_button = tk.Button(left_frame, text="Compute", command=compute)
compute_button.grid(row=9, columnspan=4, pady=10)

# Рамка для отображения решений
solution_frame = tk.Frame(tab1)
solution_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

# Placeholder для области решений
placeholder_label = tk.Label(solution_frame, text="Выберите решение", font=("Arial", 12), fg="grey")
placeholder_label.pack()

image_path = "ez.png" 

tk.Label(left_frame, text = "e(z) - CRM/PDRM ratio").grid(row = 10, column= 1, columnspan = 2)

image_label_tab1 = create_image_label(left_frame, image_path, 11, 4)

param1 = 1.0
param2 = 1.0

figure = Figure(figsize=(12, 4))
ax1 = figure.add_subplot(1, 5, 1)

axs = [ax1,
       figure.add_subplot(1, 5, 2),
       figure.add_subplot(1, 5, 3),
       figure.add_subplot(1, 5, 4),
       figure.add_subplot(1, 5, 5)]




axs[0].set_title("Field polarity", fontsize=8)
axs[0].set_ylabel('Depth')
#axs[0].tick_params(axis='y', which='both', left=True, labelleft=True)  # Включаем метки и подписи по оси Y



axs[1].set_title("e(z)", fontsize=8)



axs[2].set_title("Observed polarity", fontsize=8)


axs[3].set_title("Modeled polarity", fontsize=8)


axs[4].set_title("Lock-in-function", fontsize=8)

figure.subplots_adjust(wspace=0.1)


for ax in axs[1:]:
    ax.tick_params(left=False)
    ax.set_yticklabels([])

canvas = FigureCanvasTkAgg(figure, tab1)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Second Tab
tab2 = tk.Frame(notebook)
notebook.add(tab2, text='Forward')

# Second Tab Left Frame
left_frame2 = tk.Frame(tab2)
left_frame2.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

# Textboxes and Button in Second Tab
tk.Label(left_frame2, text = "Depths (filter parameters):").grid(row = 0, column= 0, columnspan = 2)

tk.Label(left_frame2, text="d0:").grid(row=1, column=0)
d0_val_forvard = tk.StringVar()
entry_d0 = tk.Entry(left_frame2, textvariable = d0_val_forvard)
entry_d0.grid(row=1, column=1)
entry_d0.insert(0, "1.0")

tk.Label(left_frame2, text="d1:").grid(row=2, column=0)
d1_val_forvard = tk.StringVar()
entry_d1 = tk.Entry(left_frame2, textvariable = d1_val_forvard)
entry_d1.grid(row=2, column=1)
entry_d1.insert(0, "1.0")

tk.Label(left_frame2, text="d2:").grid(row=3, column=0)
d2_val_forvard = tk.StringVar()
entry_d2 = tk.Entry(left_frame2, textvariable = d2_val_forvard)
entry_d2.grid(row=3, column=1)
entry_d2.insert(0, "1.0")

tk.Label(left_frame2, text="d3:").grid(row=4, column=0)
d3_val_forvard = tk.StringVar()
entry_d3 = tk.Entry(left_frame2, textvariable = d3_val_forvard)
entry_d3.grid(row=4, column=1)
entry_d3.insert(0, "1.0")

d0_val_forvard.trace_add("write", depth_changed_forvard)
d1_val_forvard.trace_add("write", depth_changed_forvard)
d2_val_forvard.trace_add("write", depth_changed_forvard)
d3_val_forvard.trace_add("write", depth_changed_forvard)

ez_button_forvard = tk.Button(left_frame2, text = "Import e(z)", command = load_file)
ez_button_forvard.grid(row = 5, column = 0, columnspan = 2)

tk.Label(left_frame2, text = "Field parametrs:").grid(row=6, column=0, columnspan = 2)

tk.Label(left_frame2, text="c1:").grid(row=7, column=0)
c1_var_forvard = tk.StringVar()
entry_c1_d = tk.Entry(left_frame2, textvariable=c1_var_forvard)
entry_c1_d.grid(row=7, column=1)
entry_c1_d.insert(0, "1.0")

tk.Label(left_frame2, text="c2:").grid(row=8, column=0)
c2_var_forvard = tk.StringVar()
entry_c2_d = tk.Entry(left_frame2, textvariable=c2_var_forvard)
entry_c2_d.grid(row=8, column=1)
entry_c2_d.insert(0, "1.0")

#entry_c2_d.bind("<FocusOut>", field_change_forvard)
#entry_c1_d.bind("<FocusOut>", field_change_forvard)

c1_var_forvard.trace_add("write", field_change_forvard)
c2_var_forvard.trace_add("write", field_change_forvard)

compute_button2 = tk.Button(left_frame2, text="Compute", command=direct_comptue)
compute_button2.grid(row=9, columnspan=2, pady=10)

image_path = 'ez.png'

#image_canvas_tab2 = create_image_on_canvas(left_frame2, 'ez.png', 10, 2)

tk.Label(left_frame2, text = "e(z) - CRM/PDRM ratio").grid(row = 10, column= 0, columnspan = 2)

image_label_tab2 = create_image_label(left_frame2, image_path, 11 , 4)

#image_label_tab2 = tk.Label(left_frame2, text="Image Placeholder")
#mage_label_tab2.grid(row=8, columnspan=2, pady=10)

# Second Tab Figure and Canvas
figure2 = Figure(figsize=(12, 4))
ax2_1 = figure2.add_subplot(1, 5, 1)
axs2 = [ax2_1,
        figure2.add_subplot(1, 5, 2),
        figure2.add_subplot(1, 5, 3),
        figure2.add_subplot(1, 5, 4),
        figure2.add_subplot(1, 5, 5)]

axs2[0].set_title("Field polarity", fontsize=8)
axs2[0].set_ylabel('Depth')
#axs[0].tick_params(axis='y', which='both', left=True, labelleft=True)  # Включаем метки и подписи по оси Y


#axs[1].plot(y2, x)
axs2[1].set_title("e(z)", fontsize=8)

#axs[3].plot(y4, x)

axs2[2].set_title("Observed polarity", fontsize=8)

axs2[3].set_title("Modeled polarity", fontsize=8)

#axs[4].plot(y5, x)
axs2[4].set_title("Lock-in-function", fontsize=8)

for ax in axs2[1:]:
    ax.tick_params(left=False)
    ax.set_yticklabels([])

canvas2 = FigureCanvasTkAgg(figure2, tab2)
canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Load and display the provided image in both placeholders
image_path = "ez.png"
#load_and_display_image(image_path, image_label_tab1)
#load_and_display_image(image_path, image_label_tab2)


root.mainloop()
