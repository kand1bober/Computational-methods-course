import numpy as np
import matplotlib.pyplot as plt

def gen_wiener_proc(T, delta):

    time_steps_num = int(T / delta)

    time_grid = np.linspace(0, T, time_steps_num)

    dW = np.random.normal(0, delta, time_steps_num) 

    trajectorie = np.zeros(time_steps_num) # including t= 0
    
    trajectorie = np.cumsum(dW) # integration
    
    return time_grid, trajectorie

def plot_trajectorie(time_grid, trajectorie):
    plt.plot(time_grid, trajectorie, linewidth= 0.1)
    return

def main(): 
    trajectories_num = 1000
    T = 1.0
    delta = 0.0001

    plt.figure(figsize=(12, 6))
    
    for i in range(trajectories_num):
        time_grid, trajectorie = gen_wiener_proc(T, delta)    
        plot_trajectorie(time_grid, trajectorie)

    plt.title(f'Винеровский процесс (T={T}, delta={delta})')
    plt.xlabel('Время t')
    plt.ylabel('W(t)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


# direct execution
if __name__ == "__main__":
    main()
