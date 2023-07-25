import numpy as np
np.set_printoptions(threshold=np.inf)
import time
import config
import config_beamforming
import calc_phase_shift_cartesian

n_elements = config_beamforming.elements
n_samples = config.N_SAMPLES      
f = calc_phase_shift_cartesian.f      
theta = calc_phase_shift_cartesian.theta
print('Scanning window in horizontal direction:\n' + 'theta:', -int(np.rad2deg(theta)[0,0,0,round(config_beamforming.y_res/2)]), 'to', int(np.rad2deg(theta)[0,0,config_beamforming.x_res-1,round(config_beamforming.y_res/2)]), 'deg')

x_scan = calc_phase_shift_cartesian.x_scan
y_scan = calc_phase_shift_cartesian.y_scan
x_res = config_beamforming.x_res
y_res = config_beamforming.y_res

phase_shift = calc_phase_shift_cartesian.phase_shift
threshold = 0.01       # threshold value for detecting peak values, set between 0 and 1
heatmap_empty = np.zeros((config_beamforming.x_res,config_beamforming.y_res))


def frequency_phase_shift(signal, phase_shift):
    FFT = np.fft.rfft(signal,axis=0) # Fourier transform each signal
    FFT = np.reshape(FFT, (int(n_samples/2)+1,n_elements,1,1)) # reshape into a 4D array from 2D
    FFT_shifted = FFT*phase_shift    # apply phase shift to every signal
    return FFT_shifted

def remove_neighbors(heatmap, x_index, y_index):
    n_neighbors = 5 # numbers of neighbors to remove in x- and y-direction
    x_start = x_index-n_neighbors
    x_end = x_index+n_neighbors+1
    y_start = y_index-n_neighbors
    y_end = y_index+n_neighbors+1
    if x_start < 0:
        x_start = 0
    if y_start < 0:
        y_start = 0
    
    heatmap[x_start:x_end, y_start:y_end] = 0


def peak_detection(FFT, threshold):
    #FFT_copy = np.copy(FFT)
    #heatmap = np.zeros((config_beamforming.x_res,config_beamforming.y_res))
    heatmap = np.copy(heatmap_empty)
    for f_ind in range(4, int(n_samples/2)+1):
        if np.max(FFT[f_ind,:,:]) > threshold*np.max(FFT):
            (x_max,y_max) = np.unravel_index(FFT[f_ind,:,:].argmax(), np.shape(FFT[f_ind,:,:])) #indexes for x- and y-position of the maximum peak
            #heatmap[x_max,y_max] += 1
            heatmap[x_max,y_max] += FFT[f_ind,x_max,y_max]

            ## Handling sources at the same frequency, doesn't work very well
            #remove_neighbors(FFT_copy[f_ind,:,:],x_max,y_max)
            #threshold_freq = 0.99    # set between 0 and 1
            #P = 10
            #while np.max(FFT_copy[f_ind,:,:])**P > threshold_freq*np.max(FFT)**P:
            #    (x_max,y_max) = np.unravel_index(FFT_copy[f_ind,:,:].argmax(), np.shape(FFT_copy[f_ind,:,:]))
            ##    #heatmap[x_max,y_max] += 1
            #    heatmap[x_max,y_max] += FFT_copy[f_ind,x_max,y_max]
            ##    #print('Found peak value:', round(FFT[f_ind,x_max,y_max]), 'at approx.', int(f[f_ind]), 'Hz')
            #    remove_neighbors(FFT_copy[f_ind,:,:],x_max,y_max)
    return heatmap


def main(signal):
    FFT_shifted = frequency_phase_shift(signal,phase_shift)
    FFT_power = (np.abs(np.sum(FFT_shifted,axis=1))**2)         # Power of FFT summed over all array elements
    FFT_power = FFT_power/n_elements                            # normalise with number of array elements

    heatmap = peak_detection(FFT_power, threshold)
    heatmap = (heatmap/np.max(heatmap))**10
    return heatmap
