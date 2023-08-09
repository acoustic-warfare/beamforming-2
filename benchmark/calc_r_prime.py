# CALCULATE R_PRIME
#
# Calculates the position of each microphone in the array configuration
# and stores the coordinates in the matrix r_prime.
# Each column holds the cordinate of one microphone, and the rows are according to:
#   x-coord on row 0
#   y-coord on row 1
#   z-coord on row 2
#
# Assuming that the array setup lies in the xy-plane at z=0 
# with the middle of the setup being in origo,
# and that arrays are stacked in the horizontal direction.

import numpy as np
import matplotlib.pyplot as plt
import config_other as config


def calc_r_prime(d):
    # d = distance between microphones
    half = d/2      # half distance between microphones
    r_prime = np.zeros((2, config.elements))
    element_index = 0

    # give each microphone the correct coordinate
    for array in range(config.active_arrays):
        array *= -1
        for row in range(config.rows):
            for col in range(config.columns):
                r_prime[0,element_index] = - col * d - half + array*config.columns*d + array*config.sep + config.columns* config.active_arrays * half
                r_prime[1, element_index] = row * d - config.rows * half + half
                element_index += 1
    r_prime[0,:] += (config.active_arrays-1)*config.sep/2

        #filename = 'array_setup_mode'+ str(config.mode)
        #plt.savefig('plots/setup/'+filename+'.png', dpi = 500, format = 'png')

    
    return r_prime
