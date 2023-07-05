import numpy as np
import matplotlib as plt
import config
def calc_r_prime(d):
    half = d/2
    r_prime = np.zeros((2, config.elements))
    element_index = 0
    for row in range(config.rows):
        for col in range(config.columns*config.active_arrays):
            #print(str(row) + ' ' + str(col))
            r_prime[0,element_index] = col * d - config.columns * config.active_arrays * half + half
            r_prime[1,element_index] = row * d - config.rows * half + half
            element_index += 1

    #plt.figure()
    #plt.title('Array setup')
    #plt.scatter(r_prime[0,:], r_prime[1,:].T)
    #plt.xlim([-(d*config.columns * config.active_arrays/2 + d) , d*config.columns * config.active_arrays/2 + d])
    #print(r_prime)
    return r_prime