import numpy as np
np.set_printoptions(threshold=np.inf)
import config
import generate_signals

def find_nearest(array, value):
    idx = np.argmin((np.abs(array - value)))
    return idx

c = 343
fs = config.f_sampling
N = config.samples
d = config.distance
r_prime = generate_signals.calc_r_prime(d)
x_i = r_prime[0,:]
y_i = r_prime[1,:]
theta = config.theta
phi = config.phi
x_scan = np.linspace(-8,8,config.x_res)
y_scan = np.linspace(-4.5,4.5,config.y_res)
z_scan = 10
r_scan = np.sqrt(x_scan**2 + y_scan**2 + z_scan**2)



#x_scan = np.linspace(-config.a0/2,config.a0/2,config.x_res)
#y_scan = np.linspace(-config.b0/2,config.b0/2,config.y_res)
#r_scan = config.r_scan
#z_0 = np.sqrt(r_scan**2 - x_scan**2 - y_scan**2)
#theta = np.abs(np.arccos(z_0/(np.sqrt(x_scan**2 + y_scan**2 + z_0**2))))
#phi = 

theta_source1 = config.theta_deg1    # degrees
phi_source1 = config.phi_deg1        # degrees
theta_source1_indx = find_nearest(theta,np.deg2rad(theta_source1))
phi_source1_indx = find_nearest(phi,np.deg2rad(phi_source1))

theta_source2 = config.theta_deg2    # degrees
phi_source2 = config.phi_deg2        # degrees
theta_source2_indx = find_nearest(theta,np.deg2rad(theta_source2))
phi_source2_indx = find_nearest(phi,np.deg2rad(phi_source2))


#print(r_prime)
#print(np.shape(r_prime))
f = np.linspace(0,int(fs/2),int(N/2)+1)
f = np.reshape(f, (len(f),1,1,1))
x_i = np.reshape(x_i, (1,len(x_i),1,1))
y_i = np.reshape(y_i, (1,len(y_i),1,1))
sin_theta = np.reshape(np.sin(theta), (1,1,len(theta),1))
cos_phi = np.reshape(np.cos(phi), (1,1,1,len(phi)))
sin_phi = np.reshape(np.sin(phi), (1,1,1,len(phi)))
k = 2*np.pi*f/c      # wave number

phase_shift_matrix = -k*(x_i*sin_theta*cos_phi + y_i*sin_theta*sin_phi) # rows = frequencies, columns = array elements, depth = theta, fourth dimension = phi
phase_shift = np.exp(1j*phase_shift_matrix)
#phase_shift_summed = np.sum(phase_shift,axis=1)
#print(np.shape(phase_shift_matrix))
#print(np.shape(phase_shift_summed))