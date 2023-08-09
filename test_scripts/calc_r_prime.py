import numpy as np
import matplotlib.pyplot as plt
import config_test as config
import active_microphones as am

camera_offset = 0.11      # [m]

def calc_r_prime(d):
    # d: distance between microphones (m)
    half = d/2
    r_prime = np.zeros((2, config.N_MICROPHONES))
    element_index = 0

    # give each microphone the correct coordinate
    for array in range(config.ACTIVE_ARRAYS):
        array *= -1
        for row in range(config.ROWS):
            for col in range(config.COLUMNS):
                r_prime[0,element_index] = - col*d - half + array*config.COLUMNS*d + array*config.ARRAY_SEPARATION \
                                        + config.COLUMNS*config.ACTIVE_ARRAYS * half
                r_prime[1, element_index] = row*d - config.ROWS * half + half - camera_offset
                element_index += 1
    r_prime[0,:] += (config.ACTIVE_ARRAYS-1)*config.ARRAY_SEPARATION/2
    
    # obtain vector with index of active microphones
    active_mics = am.active_microphones(config.mode, config.mics_used)

    r_prime_all = r_prime               # coordiantes of all microphones, independent of chosen mode
    r_prime = r_prime[:,active_mics]    # coordinates of active microphones, dependent of chosen mode

    # Plots the array setup
    if config.plot_setup:
        # Plots the whole array setup, do not take mode into account
        fig, ax = plt.subplots(figsize=(config.columns*config.active_arrays/2, config.rows/2))
        ax.set_box_aspect(int(config.rows)/int(config.columns*config.active_arrays))    # aspect ratio
        plt.tight_layout()
        ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)      # remove top and right axis lines
        color_arr = ['r', 'b', 'g','m'] # colors of elements in the different arrays
        dx = config.distance*0.1        # displacement of mic index text
        dy = config.distance*0.1        # displacement of mic index text
        
        element = 0
        for array in range(config.ACTIVE_ARRAYS):
            mics_array = [x for x in active_mics if ((0+array*config.ROWS*config.COLUMNS) <= x & x < ((array+1)*config.ROWS*config.COLUMNS))]
            for mic in range(len(mics_array)): #range(r_prime.shape[1]): #range(int(len(active_mics)/config.active_arrays)):
                #if active_mics[mic] in range(0,64)
                x = r_prime[0,element] * 10**2
                y = r_prime[1,element] * 10**2
                ax.scatter(x, y, color = color_arr[array])
                plt.text(x-dx, y+dy, str(active_mics[element]))
                element += 1
        #plt.title('Array setup')
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')

    if config.plot_setup:
        # Plots the whole array setup, and indicating which microphones that are active or not
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set_box_aspect(int(config.ROWS)/int(config.COLUMNS*config.ACTIVE_ARRAYS))    # aspect ratio of figure
        plt.tight_layout()
        ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)      # remove top and right axis lines

        color_arr = ['r', 'b', 'g','m'] # colors of elements in the different arrays
        dx = config.distance*0.1 * 10**2    # displacement of mic index text
        dy = config.distance*0.2 * 10**2    # displacement of mic index text
        S = 200 # size of the scatter
        FS = 22     # font size of plot texts
        element = 0
        for array in range(config.active_arrays):
            for mic in range(config.ROWS*config.COLUMNS):
                mic += array*config.ROWS*config.COLUMNS
                x = r_prime_all[0,element] * 10**2
                y = r_prime_all[1,element] * 10**2
                if mic in active_mics:
                    ax.scatter(x, y, color = color_arr[array], s=S)
                else:
                    ax.scatter(x, y, color = 'none', edgecolor=color_arr[array], linewidths = 1, s=S)
                #plt.text(x-dx, y+dy, str(element), fontsize = 12)
                element += 1
        #plt.title('Array setup')
        ax.set_xlabel('x (cm)',fontsize = FS)
        ax.set_ylabel('y (cm)', fontsize = FS)
        plt.xticks(fontsize= FS), plt.yticks(fontsize= FS)

    return r_prime_all, r_prime

