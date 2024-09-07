import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

# Функция для открытия и загрузки файла как массив NumPy
def load_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
    if file_path:
        try:
            # Загрузка файла как массив NumPy
            data = np.loadtxt(file_path)
            messagebox.showinfo("Файл загружен", f"Данные из файла:\n{data}")
            # Действия с загруженными данными
            process_loaded_data(data)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {e}")

# Пример функции для обработки загруженных данных
def process_loaded_data(data):
    # Пример обработки данных: просто выводим их в консоль
    print("Загруженные данные:")
    print(data)

# Функция для открытия окна настроек
def open_options():
    # Окно настроек
    options_window = tk.Toplevel(root)
    options_window.title("Настройки")

    # Параметры для настройки
    tk.Label(options_window, text="Функция потерь:").grid(row=0, column=0)
    loss_function_combobox = ttk.Combobox(options_window, values=["MSE", "MAE", "Cross-Entropy"])
    loss_function_combobox.grid(row=0, column=1)
    loss_function_combobox.set("MSE")

    tk.Label(options_window, text="Оптимизация:").grid(row=1, column=0)
    optimization_combobox = ttk.Combobox(options_window, values=["SGD", "Adam", "RMSprop"])
    optimization_combobox.grid(row=1, column=1)
    optimization_combobox.set("SGD")

    def apply_settings():
        selected_loss_function = loss_function_combobox.get()
        selected_optimization = optimization_combobox.get()
        messagebox.showinfo("Настройки", f"Выбрана функция потерь: {selected_loss_function}\nВыбран метод оптимизации: {selected_optimization}")

    apply_button = tk.Button(options_window, text="Применить", command=apply_settings)
    apply_button.grid(row=2, columnspan=2, pady=10)

# Функция для обновления графиков
def compute():
    try:
        # Получение параметров
        d0_low = float(entry_d0_low.get())
        d0_high = float(entry_d0_high.get())
        d1_low = float(entry_d1_low.get())
        d1_high = float(entry_d1_high.get())
        d2_low = float(entry_d2_low.get())
        d2_high = float(entry_d2_high.get())
        d3_low = float(entry_d3_low.get())
        d3_high = float(entry_d3_high.get())
        calculation_times = int(entry_calc_times.get())

        # Пример данных
        x = np.linspace(0, 10, 100)
        y1 = d0_low * np.sin(d0_high * x)
        y2 = d1_low * np.cos(d1_high * x)
        y3 = d2_low * np.sin(d2_high * x) * np.cos(d3_high * x)
        y4 = np.sin(x) + np.cos(x)

        # Очистка графиков
        for ax in axs:
            ax.clear()

        # Построение графиков с общей вертикальной осью Y
        axs[0].plot(y1, x)
        axs[0].set_title("y = d0_low * sin(d0_high * x)", fontsize=8)
        axs[0].set_ylabel('Common X')  # Подпись оси Y только у левого графика

        axs[1].plot(y2, x)
        axs[1].set_title("y = d1_low * cos(d1_high * x)", fontsize=8)

        axs[2].plot(y3, x)
        axs[2].set_title("y = d2_low * sin(d2_high * x) * cos(d3_high * x)", fontsize=8)

        axs[3].plot(y4, x)
        axs[3].set_title("y = sin(x) + cos(x)", fontsize=8)

        # Установка общей вертикальной оси Y для всех графиков
        for ax in axs[1:]:
            ax.tick_params(left=False)  # Отключаем ticks по оси Y для всех графиков, кроме левого
            ax.set_yticklabels([])  # Убираем подписи значений вертикальной оси

        # Уменьшение отступов между графиками
        figure.subplots_adjust(wspace=0.1)  # Небольшое расстояние между графиками

        canvas.draw()
    except ValueError:
        messagebox.showerror("Ошибка", "Введите корректные числовые параметры.")

# Создание главного окна
root = tk.Tk()
root.title("Приложение с графиками")

# Создание панели инструментов
menu_bar = tk.Menu(root)
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Open", command=load_file)  # Изменено на load_file
menu_bar.add_cascade(label="File", menu=file_menu)

options_menu = tk.Menu(menu_bar, tearoff=0)
options_menu.add_command(label="Settings", command=open_options)
menu_bar.add_cascade(label="Options", menu=options_menu)
root.config(menu=menu_bar)

# Левые элементы управления
left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

# Ввод параметров
tk.Label(left_frame, text="d0 low:").grid(row=0, column=0)
entry_d0_low = tk.Entry(left_frame)
entry_d0_low.grid(row=0, column=1)
entry_d0_low.insert(0, "1.0")

tk.Label(left_frame, text="d0 high:").grid(row=0, column=2)
entry_d0_high = tk.Entry(left_frame)
entry_d0_high.grid(row=0, column=3)
entry_d0_high.insert(0, "1.0")

tk.Label(left_frame, text="d1 low:").grid(row=1, column=0)
entry_d1_low = tk.Entry(left_frame)
entry_d1_low.grid(row=1, column=1)
entry_d1_low.insert(0, "1.0")

tk.Label(left_frame, text="d1 high:").grid(row=1, column=2)
entry_d1_high = tk.Entry(left_frame)
entry_d1_high.grid(row=1, column=3)
entry_d1_high.insert(0, "1.0")

tk.Label(left_frame, text="d2 low:").grid(row=2, column=0)
entry_d2_low = tk.Entry(left_frame)
entry_d2_low.grid(row=2, column=1)
entry_d2_low.insert(0, "1.0")

tk.Label(left_frame, text="d2 high:").grid(row=2, column=2)
entry_d2_high = tk.Entry(left_frame)
entry_d2_high.grid(row=2, column=3)
entry_d2_high.insert(0, "1.0")

tk.Label(left_frame, text="d3 low:").grid(row=3, column=0)
entry_d3_low = tk.Entry(left_frame)
entry_d3_low.grid(row=3, column=1)
entry_d3_low.insert(0, "1.0")

tk.Label(left_frame, text="d3 high:").grid(row=3, column=2)
entry_d3_high = tk.Entry(left_frame)
entry_d3_high.grid(row=3, column=3)
entry_d3_high.insert(0, "1.0")

tk.Label(left_frame, text="Calculation times:").grid(row=4, column=0, columnspan=2)
entry_calc_times = tk.Entry(left_frame)
entry_calc_times.grid(row=4, column=2, columnspan=2)
entry_calc_times.insert(0, "10")

compute_button = tk.Button(left_frame, text="Compute", command=compute)
compute_button.grid(row=5, columnspan=4, pady=10)

# Глобальные параметры
param1 = 1.0
param2 = 1.0

# Создание фигуры и графиков с общей вертикальной осью Y
figure = Figure(figsize=(10, 4))  # Увеличена ширина для горизонтального размещения
ax1 = figure.add_subplot(1, 4, 1)
axs = [ax1,
       figure.add_subplot(1, 4, 2, sharey=ax1),
       figure.add_subplot(1, 4, 3, sharey=ax1),
       figure.add_subplot(1, 4, 4, sharey=ax1)]

# Заполнение графиков начальными данными
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * np.cos(x)
y4 = np.sin(x) + np.cos(x)

# Построение графиков с начальными данными
axs[0].plot(y1, x)
axs[0].set_title("y = sin(x)", fontsize=8)
axs[0].set_ylabel('Common X')

axs[1].plot(y2, x)
axs[1].set_title("y = cos(x)", fontsize=8)

axs[2].plot(y3, x)
axs[2].set_title("y = sin(x) * cos(x)", fontsize=8)

axs[3].plot(y4, x)
axs[3].set_title("y = sin(x) + cos(x)", fontsize=8)

# Установка параметров для отступов
figure.subplots_adjust(wspace=0.1)  # Устанавливаем небольшое расстояние между графиками

# Убираем метки и значения вертикальной оси для графиков, кроме первого
for ax in axs[1:]:
    ax.tick_params(left=False)  # Убираем ticks по оси Y
    ax.set_yticklabels([])  # Убираем подписи значений вертикальной оси

canvas = FigureCanvasTkAgg(figure, root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

root.mainloop()
