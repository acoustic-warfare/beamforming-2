import numpy as np
import matplotlib.pyplot as plt
import config
import active_microphones as am

from matplotlib import rc # only for plot

def calc_r_prime(d):
    half = d/2
    r_prime = np.zeros((2, config.elements))
    element_index = 0
    for array in range(config.active_arrays):
        for row in range(config.rows):
            for col in range(config.columns):
                r_prime[0,element_index] = col * d + half + array*config.columns*d + array*config.sep - config.columns* config.active_arrays * half
                r_prime[1, element_index] = row * d - config.rows * half + half
                element_index += 1
    r_prime[0,:] -= config.active_arrays*config.sep/2
    active_mics = am.active_microphones()

    r_prime_all = r_prime
    r_prime = r_prime[:,active_mics]


    if config.plot_setup:
        fig, ax = plt.subplots(figsize=(config.columns*config.active_arrays/2, config.rows/2))
        ax = plt.gca() #you first need to get the axis handle
        ax.set_aspect(0.5* config.columns*config.active_arrays / config.rows) #sets the height to width ratio to 1.5.
        element = 0
        color_arr = ['r', 'b', 'g','m']
        dx = 0
        dy = 0
        for array in range(config.active_arrays):
            mics_array = [x for x in active_mics if ((0+array*config.rows*config.columns) <= x & x < ((array+1)*config.rows*config.columns))]
            plt.title('Array setup')
            for mic in range(len(mics_array)): #range(r_prime.shape[1]): #range(int(len(active_mics)/config.active_arrays)):
                #if active_mics[mic] in range(0,64)
                x = r_prime[0,element]
                y = r_prime[1,element]
                ax.scatter(x, y, color = color_arr[array])
                plt.text(x-dx, y+dy, str(active_mics[element]))
                element += 1
            
    if config.plot_setup:
        fig, ax = plt.subplots(figsize=(config.columns*config.active_arrays/2, config.rows/2))
        ax = plt.gca() #you first need to get the axis handle
        ax.set_aspect(0.5* config.rows / config.columns*config.active_arrays) #sets the height to width ratio to 1.5.
        element = 0
        color_arr = ['r', 'b', 'g','m']
        dx = 0
        dy = 0
        for array in range(config.active_arrays):
            plt.title('Array setup')
            for mic in range(config.rows*config.columns):
                mic += array*config.rows*config.columns
                x = r_prime_all[0,element]
                y = r_prime_all[1,element]
                if mic in active_mics:
                    ax.scatter(x, y, color = color_arr[array])
                else:
                    ax.scatter(x, y, color = 'none', edgecolor=color_arr[array], linewidths = 1)
                plt.text(x-dx, y+dy, str(element))
                element += 1

    return r_prime

