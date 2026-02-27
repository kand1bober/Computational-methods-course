import numpy as np
import matplotlib.pyplot as plt

def generate_wiener_process(N, T=1.0, delta=0.0001):
    """
    Генерирует N траекторий винеровского процесса
    
    Параметры:
    N - количество траекторий
    T - временной горизонт (по умолчанию 1)
    delta - шаг по времени (по умолчанию 0.0001)
    
    Возвращает:
    time_grid - временная сетка
    trajectories - массив траекторий размерности (N, M+1), 
                   где M = int(T/delta)
    """
    # Количество шагов
    M = int(T / delta)
    
    # Создаем временную сетку
    time_grid = np.linspace(0, T, M + 1)
    
    # Генерируем приращения винеровского процесса
    # dW ~ N(0, delta) - нормально распределенные приращения
    dW = np.random.normal(0, np.sqrt(delta), size=(N, M))
    
    # Интегрируем приращения для получения траекторий
    # Начинаем с W(0)=0 для всех траекторий
    trajectories = np.zeros((N, M + 1))
    trajectories[:, 1:] = np.cumsum(dW, axis=1)
    
    return time_grid, trajectories

def plot_wiener_trajectories(time_grid, trajectories, num_trajectories_to_plot=5):
    """
    Визуализирует несколько траекторий винеровского процесса
    """
    plt.figure(figsize=(12, 6))
    
    # Отображаем указанное количество траекторий
    for i in range(min(num_trajectories_to_plot, trajectories.shape[0])):
        plt.plot(time_grid, trajectories[i], linewidth=1, 
                label=f'Траектория {i+1}')
    
    plt.title(f'Винеровский процесс (T={T}, delta={delta})')
    plt.xlabel('Время t')
    plt.ylabel('W(t)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def plot_statistics(time_grid, trajectories):
    """
    Отображает статистические характеристики ансамбля траекторий
    """
    plt.figure(figsize=(12, 8))
    
    # Вычисляем статистики
    mean = np.mean(trajectories, axis=0)
    std = np.std(trajectories, axis=0)
    var = np.var(trajectories, axis=0)
    
    # Теоретические значения
    theoretical_std = np.sqrt(time_grid)
    theoretical_var = time_grid
    
    # График среднего
    plt.subplot(2, 2, 1)
    plt.plot(time_grid, mean, 'b-', label='Эмпирическое среднее', linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--', label='Теоретическое среднее (0)')
    plt.xlabel('Время t')
    plt.ylabel('E[W(t)]')
    plt.title('Среднее значение')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # График дисперсии
    plt.subplot(2, 2, 2)
    plt.plot(time_grid, var, 'b-', label='Эмпирическая дисперсия', linewidth=2)
    plt.plot(time_grid, theoretical_var, 'r--', label='Теоретическая дисперсия (t)')
    plt.xlabel('Время t')
    plt.ylabel('Var[W(t)]')
    plt.title('Дисперсия')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # График стандартного отклонения
    plt.subplot(2, 2, 3)
    plt.plot(time_grid, std, 'b-', label='Эмпирическое std', linewidth=2)
    plt.plot(time_grid, theoretical_std, 'r--', label='Теоретическое std (√t)')
    plt.xlabel('Время t')
    plt.ylabel('Std[W(t)]')
    plt.title('Стандартное отклонение')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Гистограмма конечных значений
    plt.subplot(2, 2, 4)
    final_values = trajectories[:, -1]
    plt.hist(final_values, bins=30, density=True, alpha=0.7, 
             label='Эмпирическое распределение')
    
    # Теоретическое нормальное распределение для W(1) ~ N(0,1)
    x = np.linspace(-3, 3, 100)
    theoretical = 1/np.sqrt(2*np.pi) * np.exp(-x**2/2)
    plt.plot(x, theoretical, 'r--', linewidth=2, 
             label='Теоретическое N(0,1)')
    
    plt.xlabel('W(1)')
    plt.ylabel('Плотность')
    plt.title('Распределение конечных значений')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Параметры задачи
N = 1000  # Количество траекторий
T = 1.0   # Временной горизонт
delta = 0.0001  # Шаг по времени

print(f"Генерация {N} траекторий винеровского процесса...")
print(f"T = {T}, delta = {delta}")
print(f"Количество временных шагов: {int(T/delta)}")

# Генерация траекторий
time_grid, trajectories = generate_wiener_process(N, T, delta)

print("Готово!")
print(f"Размерность массива траекторий: {trajectories.shape}")

# Визуализация нескольких траекторий
plot_wiener_trajectories(time_grid, trajectories, num_trajectories_to_plot=10)

# Проверка статистических свойств
plot_statistics(time_grid, trajectories)

# Дополнительная информация
print(f"\nСтатистика конечных значений W(1):")
print(f"Среднее: {np.mean(trajectories[:, -1]):.4f} (теоретическое: 0)")
print(f"Дисперсия: {np.var(trajectories[:, -1]):.4f} (теоретическая: 1)")
print(f"Стандартное отклонение: {np.std(trajectories[:, -1]):.4f} (теоретическое: 1)")

# Сохранение данных (опционально)
np.savez('wiener_trajectories.npz', 
         time_grid=time_grid, 
         trajectories=trajectories)
print("\nДанные сохранены в файл 'wiener_trajectories.npz'")

