import numpy as np
np.set_printoptions(threshold=np.inf)
import time
import config
import config_beamforming
import calc_phase_shift_cartesian

n_elements = config_beamforming.elements
n_samples = config.N_SAMPLES            
theta = calc_phase_shift_cartesian.theta
print('Scanning window in horizontal direction:\n' + 'theta:', -int(np.rad2deg(theta)[0,0,0,round(config_beamforming.y_res/2)]), 'to', int(np.rad2deg(theta)[0,0,config_beamforming.x_res-1,round(config_beamforming.y_res/2)]), 'deg')

x_scan = calc_phase_shift_cartesian.x_scan
y_scan = calc_phase_shift_cartesian.y_scan

phase_shift = calc_phase_shift_cartesian.phase_shift

def frequency_phase_shift(signal, phase_shift):
    FFT = np.fft.rfft(signal,axis=0) # Fourier transform each signal
    FFT = np.reshape(FFT, (int(n_samples/2)+1,n_elements,1,1)) # reshape into a 4D array from 2D
    FFT_shifted = FFT*phase_shift     # apply phase shift to every signal
    return FFT_shifted

def main(signal):
    #start = time.time()
    FFT_shifted = frequency_phase_shift(signal,phase_shift)
    FFT_power = (np.abs(np.sum(FFT_shifted,axis=1))**2)         # Power of FFT summed over all array elements
    FFT_power = FFT_power/n_elements                            # normalise with number of array elements
    FFT_power_summed = np.sum(FFT_power, axis=0)                # Power of the FFT summed over all elements and frequencies
    #end = time.time()
    #print('Simulation time:', round((end - start), 4), 's')
    return FFT_power_summed
