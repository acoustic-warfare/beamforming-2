import numpy as np
np.set_printoptions(threshold=np.inf)
import config
import active_microphones_modes as amm

c = config.PROPAGATION_SPEED
fs = config.fs
N = config.N_SAMPLES
f = np.linspace(0,int(fs/2),int(N/2)+1) # frequencies after FFT
d = config.ELEMENT_DISTANCE

def mode_matrix(matrix):
    mode_matrix = np.zeros_like(matrix)
    print(np.shape(mode_matrix))
    distances = c/(2*f + 0.0001)
    for mode in range(1, config.modes+1):
        if mode == 1:
            mode_idxs = (distances>0)*(distances<=(mode+1)*d)
        elif mode == config.modes:
            mode_idxs = (distances>mode*d)
        else:
            mode_idxs = (distances>mode*d)*(distances<=(mode+1)*d)

        mode_interval = np.where(mode_idxs)[0]
        #print('mode', mode, 'interval:', mode_interval)
        active_mics_mode = amm.active_microphones(mode)
        mode_matrix[mode_interval[0]:mode_interval[-1]+1,active_mics_mode,:,:] = matrix[mode_interval[0]:mode_interval[-1]+1,active_mics_mode,:,:]
    
    return mode_matrix