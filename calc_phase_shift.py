import numpy as np
np.set_printoptions(threshold=np.inf)
import config
import calc_r_prime

c = 343
fs = config.f_sampling
N = config.samples
d = config.distance
r_prime = calc_r_prime.calc_r_prime(d)
#r_prime = np.array([[3, 1, 2], [6, 4, 2], [0, 0, 0]])
x_i = r_prime[0,:]
y_i = r_prime[1,:]
theta = np.linspace(0,np.pi,56)
phi = np.linspace(0,2*np.pi,56)
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