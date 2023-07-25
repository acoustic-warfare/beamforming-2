import numpy as np
import matplotlib.pyplot as plt
import config

def calc_r_prime(d):
    half = d/2
    r_prime = np.zeros((2, config.N_MICROPHONES))
    element_index = 0
    for array in range(config.ACTIVE_ARRAYS):
        for row in range(config.rows):
            for col in range(config.columns):
                r_prime[0,element_index] = col * d + half + array*config.columns*d + array*config.ARRAY_SEPARATION - config.columns* config.ACTIVE_ARRAYS * half
                r_prime[1, element_index] = row * d - config.rows * half + half
                element_index += 1
    r_prime[0,:] -= config.ACTIVE_ARRAYS*config.ARRAY_SEPARATION/2

    if config.plot_setup:
        plt.figure()#figsize=(config.columns*config.active_arrays, config.rows))
        ax = plt.gca() #you first need to get the axis handle
        ax.set_aspect(16/9) #sets the height to width ratio to 1.5.
        element = 0
        color_arr = ['r', 'b', 'g','m']
        dx = 0
        dy = 0
        for array in range(config.ACTIVE_ARRAYS):
            plt.title('Array setup')
            for mic in range(config.rows*config.columns):
                x = r_prime[0,element]
                y = r_prime[1,element]
                plt.scatter(x, y, color=color_arr[array])
                plt.text(x-dx, y+dy, str(element+1))
                element += 1
        plt.show()
    return r_prime