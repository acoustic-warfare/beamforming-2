import numpy as np
np.set_printoptions(threshold=np.inf)
import config_test as config
import calc_r_prime
import calc_mode_matrix
import calc_mode_matrices
import active_microphones as am

print('\nCalculate phase shift matrix')
c = config.PROPAGATION_SPEED
fs = int(config.fs)
N = config.N_SAMPLES
d = config.ELEMENT_DISTANCE
theta_max = config.VIEW_ANGLE/2
active_mics = am.active_microphones(config.SKIP_N_MICS, config.mics_used)

# microphone coordinates
r_prime_all, r_prime = calc_r_prime.calc_r_prime(d)
x_i = r_prime_all[0,:]
y_i = r_prime_all[1,:]
x_i = np.reshape(x_i, (1,len(x_i),1,1))
y_i = np.reshape(y_i, (1,len(y_i),1,1))
# scanning window
x_scan_max = config.Z*np.tan(np.deg2rad(theta_max))
x_scan_min = -x_scan_max
y_scan_max = x_scan_max/config.ASPECT_RATIO
y_scan_min = -y_scan_max

x_scan = np.linspace(x_scan_min,x_scan_max,config.MAX_RES_X)
y_scan = np.linspace(y_scan_min,y_scan_max,config.MAX_RES_Y)
x_scan = np.reshape(x_scan, (1,1,len(x_scan),1))    # reshape into 4D arrays
y_scan = np.reshape(y_scan, (1,1,1,len(y_scan)))    # reshape into 4D arrays
r_scan = np.sqrt(x_scan**2 + y_scan**2 + config.Z**2) # distance between middle of array to the xy-scanning coordinate

f = np.linspace(0,int(fs/2),int(N/2)+1) # frequencies after FFT
f = np.reshape(f, (len(f),1,1,1))
k = 2*np.pi*f/c                         # wave number

theta = np.arccos(config.Z/r_scan)
phi = np.arctan2(y_scan,x_scan)

# calc of phase shift based on scanning window in cartesian coordinates instead of angles
phase_shift_matrix_full = -k*((x_scan*x_i + y_scan*y_i) / r_scan) # rows = frequencies, columns = array elements, depth = theta, fourth dimension = phi
phase_shift_full = np.exp(1j*phase_shift_matrix_full)


phase_shift = phase_shift_full[:,active_mics,:,:]
#print(np.shape(phase_shift))


phase_shift_modes, n_active_mics = calc_mode_matrix.mode_matrix(phase_shift_full)
phase_shift_modes = phase_shift_modes[:,active_mics,:,:]
#print(np.shape(mode_matrix))

mode_matrices, mode_intervals, active_mics_mode_list = calc_mode_matrices.calc_mode_matrices(phase_shift_full)