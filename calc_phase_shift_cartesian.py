import numpy as np
np.set_printoptions(threshold=np.inf)
import config
import generate_signals

def find_nearest(array, value):
    idx = np.argmin((np.abs(array - value)))
    return idx

c = config.c
fs = config.f_sampling
N = config.samples
d = config.distance
r_prime = generate_signals.calc_r_prime(d)
x_i = r_prime[0,:]
y_i = r_prime[1,:]
theta_max = config.alpha/2
x_scan_max = config.z_scan*np.tan(np.deg2rad(theta_max))
x_scan_min = -x_scan_max
y_scan_max = x_scan_max/config.aspect_ratio
y_scan_min = -y_scan_max
x_scan = np.linspace(x_scan_min,x_scan_max,config.x_res)
y_scan = np.linspace(y_scan_min,y_scan_max,config.y_res)
x_scan = np.reshape(x_scan, (1,1,len(x_scan),1))    # reshape into 4D arrays
y_scan = np.reshape(y_scan, (1,1,1,len(y_scan)))    # reshape into 4D arrays


theta = np.arctan(x_scan/config.z_scan)
phi = np.arctan(y_scan/x_scan)

theta_source1 = config.theta_deg1    # degrees
phi_source1 = config.phi_deg1        # degrees
theta_source1_indx = find_nearest(theta,np.deg2rad(theta_source1))
phi_source1_indx = find_nearest(phi,np.deg2rad(phi_source1))


theta_source2 = config.theta_deg2    # degrees
phi_source2 = config.phi_deg2        # degrees
theta_source2_indx = find_nearest(theta,np.deg2rad(theta_source2))
phi_source2_indx = find_nearest(phi,np.deg2rad(phi_source2))

f = np.linspace(0,int(fs/2),int(N/2)+1)
f = np.reshape(f, (len(f),1,1,1))
x_i = np.reshape(x_i, (1,len(x_i),1,1))
y_i = np.reshape(y_i, (1,len(y_i),1,1))
sin_theta = np.sin(theta)
cos_phi = np.cos(phi)
sin_phi = np.sin(phi)
k = 2*np.pi*f/c      # wave number

phase_shift_matrix = -k*(x_i*sin_theta*cos_phi + y_i*sin_theta*sin_phi) # rows = frequencies, columns = array elements, depth = theta, fourth dimension = phi
phase_shift = np.exp(1j*phase_shift_matrix)
#print(np.shape(phase_shift))