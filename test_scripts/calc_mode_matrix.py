import numpy as np
np.set_printoptions(threshold=np.inf)
import config_test as config
import active_microphones as am

c = config.PROPAGATION_SPEED
fs = config.fs
N = config.N_SAMPLES
f_FFT = np.linspace(0,int(fs/2),int(N/2)+1) # frequencies after FFT
d = config.ELEMENT_DISTANCE

def mode_matrix(matrix, f=f_FFT, matrix_type = 'FFT'):
    mode_matrix = np.zeros_like(matrix)
    n_active_mics = np.zeros((len(f),1,1))
    distances = c/(2*f + 0.0001)
    for mode in range(1, config.modes+1):
        if mode == 1:
            mode_idxs = (distances>0)*(distances<=(mode+1)*d)
        elif mode == config.modes:
            mode_idxs = (distances>mode*d)
        else:
            mode_idxs = (distances>mode*d)*(distances<=(mode+1)*d)

        mode_interval = np.where(mode_idxs)[0]
        
        if mode == 4:       # mode 4 gives bad results. replaced by mode 3
            active_mics_mode = am.active_microphones(mode-1,config.mics_used)
        else:
            active_mics_mode = am.active_microphones(mode,config.mics_used)
        
        if matrix_type == 'FFT':
            mode_matrix[mode_interval[0]:mode_interval[-1]+1,active_mics_mode,:,:] = matrix[mode_interval[0]:mode_interval[-1]+1,active_mics_mode,:,:]
        elif matrix_type == 'AF':
            mode_matrix[:,:,mode_interval[0]:mode_interval[-1]+1,active_mics_mode] = matrix[:,:,mode_interval[0]:mode_interval[-1]+1,active_mics_mode]
        else:
            print('Unknown matrix type')
        n_active_mics[mode_interval[0]:mode_interval[-1]+1,0,0] = len(active_mics_mode)

    return mode_matrix, n_active_mics