import numpy as np
np.set_printoptions(threshold=np.inf)
import config
import config_beamforming
import calc_r_prime

c = config_beamforming.c
fs = config.fs
N = config.N_SAMPLES
d = config_beamforming.distance
r_prime = calc_r_prime.calc_r_prime(d)
x_i = r_prime[0,:]
y_i = r_prime[1,:]

# Scanning window
theta_max = config_beamforming.alpha/2
z_scan = config_beamforming.z_scan
x_scan_max = z_scan*np.tan(np.deg2rad(theta_max))
x_scan_min = -x_scan_max
y_scan_max = x_scan_max/config_beamforming.aspect_ratio
y_scan_min = -y_scan_max
x_scan = np.linspace(x_scan_min,x_scan_max,config_beamforming.x_res)
y_scan = np.linspace(y_scan_min,y_scan_max,config_beamforming.y_res)
x_scan = np.reshape(x_scan, (1,1,len(x_scan),1))
y_scan = np.reshape(y_scan, (1,1,1,len(y_scan)))

theta = np.arccos(z_scan/(np.sqrt(x_scan**2 + y_scan**2 + z_scan**2)))
phi = np.arctan2(y_scan,x_scan)

f = np.linspace(0,int(fs/2),int(N/2)+1)
f = np.reshape(f, (len(f),1,1,1))
x_i = np.reshape(x_i, (1,len(x_i),1,1))
y_i = np.reshape(y_i, (1,len(y_i),1,1))
sin_theta = np.sin(theta)
cos_phi = np.cos(phi)
sin_phi = np.sin(phi)
k = 2*np.pi*f/c      # wave number

phase_shift_matrix = -k*(x_i*sin_theta*cos_phi + y_i*sin_theta*sin_phi) # rows = frequencies, columns = array elements, depth = x, fourth dimension = y
phase_shift = np.exp(1j*phase_shift_matrix)
#print(np.shape(phase_shift))