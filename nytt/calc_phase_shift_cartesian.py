import numpy as np
np.set_printoptions(threshold=np.inf)
import config
import calc_r_prime

print('\nCalculate phase shift matrix')
c = config.c
fs = config.f_sampling
N = config.samples
d = config.distance
theta_max = config.alpha/2

# microphone coordinates
r_prime = calc_r_prime.calc_r_prime(d)
x_i = r_prime[0,:]
y_i = r_prime[1,:]
x_i = np.reshape(x_i, (1,len(x_i),1,1))
y_i = np.reshape(y_i, (1,len(y_i),1,1))

# scanning window
x_scan_max = config.z_scan*np.tan(np.deg2rad(theta_max))
x_scan_min = -x_scan_max
y_scan_max = x_scan_max/config.aspect_ratio
y_scan_min = -y_scan_max

x_scan = np.linspace(x_scan_min,x_scan_max,config.x_res)
y_scan = np.linspace(y_scan_min,y_scan_max,config.y_res)
x_scan = np.reshape(x_scan, (1,1,len(x_scan),1))    # reshape into 4D arrays
y_scan = np.reshape(y_scan, (1,1,1,len(y_scan)))    # reshape into 4D arrays
r_scan = np.sqrt(x_scan**2 + y_scan**2 + config.z_scan**2) # distance between middle of array to the xy-scanning coordinate

f = np.linspace(0,int(fs/2),int(N/2)+1) # frequencies after FFT
f = np.reshape(f, (len(f),1,1,1))
k = 2*np.pi*f/c                         # wave number

theta = np.arccos(config.z_scan/r_scan)
phi = np.arctan2(y_scan,x_scan)
#sin_theta = np.sin(theta)
#cos_phi = np.cos(phi)
#sin_phi = np.sin(phi)
#phase_shift_matrix = -k*(x_i*sin_theta*cos_phi + y_i*sin_theta*sin_phi) # rows = frequencies, columns = array elements, depth = theta, fourth dimension = phi

# calc of phase shift based on scanning window in cartesian coordinates instead of angles
phase_shift_matrix = -k*((x_scan*x_i + y_scan*y_i) / r_scan) # rows = frequencies, columns = array elements, depth = theta, fourth dimension = phi
phase_shift = np.exp(1j*phase_shift_matrix)
