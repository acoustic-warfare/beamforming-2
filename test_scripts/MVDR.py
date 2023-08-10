import numpy as np
import matplotlib.pyplot as plt

import time
import math
import config_test as config
import generate_signals
import calc_r_prime
import active_microphones as am
from scipy import signal


def generate_filename():
    if config.sources == 1:
        filename ='emul_'+ 'samples='+str(config.N_SAMPLES) + '_start'+ str(config.f_start1)+'Hz'+'_end'+str(config.f_end1)+'Hz'+\
            '_theta='+str(config.theta_deg1)+'_phi='+str(config.phi_deg1)+ '_AWD=' + str(config.away_distance) + \
            '_E'+ str(config.ROWS*config.COLUMNS) + '_A'+str(config.ACTIVE_ARRAYS)
    elif config.sources == 2:
        filename ='emul_'+'samples='+str(config.N_SAMPLES) + \
        '_start'+str(config.f_start1)+str(config.f_start2)+'Hz'+\
        '_end'+str(config.f_end1)+str(config.f_end2)+'Hz'+ \
        '_theta='+str(config.theta_deg1)+str(config.theta_deg2)+ \
        '_phi='+str(config.phi_deg1)+str(config.phi_deg2)+\
        '_AWD=' + str(config.away_distance) +\
        '_E'+ str(config.ROWS*config.COLUMNS) + '_A'+str(config.ACTIVE_ARRAYS)
    return filename

def validation_check(y_scan, x_scan):
    # Validation check
    xy_val_check = np.zeros((config.MAX_RES_X,config.MAX_RES_Y))
    z = config.Z
    theta_s = np.array([config.theta_deg1, config.theta_deg2])*math.pi/180
    phi_s = np.array([config.phi_deg1, config.phi_deg2])*math.pi/180
    r_scan = z/np.cos(theta_s)

    for source_ind in range(config.sources):
            x_s = r_scan[source_ind] * math.sin(theta_s[source_ind]) * math.cos(phi_s[source_ind])
            y_s = r_scan[source_ind] * math.sin(theta_s[source_ind]) * math.sin(phi_s[source_ind])
            x_ind = (np.abs(x_scan[:,0,0] - x_s)).argmin()
            y_ind = (np.abs(y_scan[0,:,0] - y_s)).argmin()
            xy_val_check[x_ind,y_ind] = 1

    fig, ax = plt.subplots() #figsize = (16,9))
    plt.title('Actual location of sources')
    pic = ax.pcolormesh(x_scan[:,0,0], y_scan[0,:,0], xy_val_check.T )#,shading='gouraud') # heatmap summed over all frequencies
    fig.colorbar(pic)

### GENERATE SIGNALS
filename = 'emulated_data/' + generate_filename()
try:
    signal = np.float32(np.load(filename+'.npy',allow_pickle=True))
    print('Loading from memory: ' + filename)
except:
    generate_signals.main(generate_filename())
    signal = np.float32(np.load(filename+'.npy',allow_pickle=True))


# source with noise
signal += np.random.uniform(0,np.max(signal)/5, (config.ELEMENTS, config.N_SAMPLES)).T    # add noise to signals
signal = signal[:,am.active_microphones(config.mode, 'all')]                                              # take out signals from active microphones
signal = signal/np.max(signal)                                                          # normalize so signals are between -1 and 1

# No source, only noise
# signal[:,:] = 0 + np.random.uniform(0,np.max(signal)/2, (config.elements, config.samples)).T
# signal = signal/np.max(signal)

# Figure of the signal
#plt.figure()
#plt.plot(signal)
#print('max_sig',np.max(signal))

z_scan = config.Z
AS = config.ASPECT_RATIO
alpha = config.VIEW_ANGLE #config.alpha  # total scanning angle (bildvinkel) in theta-direction [degrees], from config

x_res = config.MAX_RES_X #config.x_res  # resolution in x, from config
y_res = config.MAX_RES_Y  # resolution in y, from config

# Calculations for scanning window
r_prime_ALL, r_prime = calc_r_prime.calc_r_prime(config.ELEMENT_DISTANCE)  # matrix holding the xy positions of each microphone
x_i = r_prime[0,:]                      # x-coord for microphone i
y_i = r_prime[1,:]                      # y-coord for microphone i

# outer limits of scanning window in xy-plane
x_scan_max = z_scan*np.tan((alpha/2)*math.pi/180)
x_scan_min = - x_scan_max
y_scan_max = x_scan_max/AS
y_scan_min = -y_scan_max

# scanning window
x_scan = np.linspace(x_scan_min, x_scan_max, x_res).reshape(x_res,1,1)
y_scan = np.linspace(y_scan_min, y_scan_max, y_res).reshape(1, y_res,1)
r_scan = np.sqrt(x_scan**2 + y_scan**2 + z_scan**2) # distance between middle of array to the xy-scanning coordinate


fs = config.fs
N = 512                         # number of samples
M = config.ELEMENTS             # number of array elements
P = config.sources              # number of sources
c = config.PROPAGATION_SPEED    
f = 1.5 * 10**3                 # frequency that the music algorithm is based on
k = 2*math.pi*f / c             # wavenumber
d = config.ELEMENT_DISTANCE             # distance between elements
N_elements = signal.shape[1]

psi =  (x_scan*x_i + y_scan*y_i) / r_scan
S = np.exp(1j*k*psi).reshape(x_res, y_res, N_elements,1)   # steering matrix

# MVDR ALGORITHM STARTS HERE
start = time.time()
print(signal.shape)
x = np.fft.rfft(signal.T,axis=1)    # FFT each signal (signal for music algorithm is assumed to be complex)

R = (x @ x.conj().transpose()) / N   # calculate covariance matrix and normalize with number of samples


#https://dsp.stackexchange.com/questions/60091/implementing-mvdr-beamformer-in-the-stft-domain
# https://www.mathworks.com/help/phased/ref/phased.mvdrbeamformer-system-object.html
invR = np.linalg.pinv(R)
MVDR = np.zeros((x_res,y_res))     # heatmap for music algorithm
#for ff in range(10):
for jj in range(y_res):
    for ii in range(x_res): 
        SS = S[ii,jj,:,:]                                     # stearing matrix for a specific direction
        denominator = SS.conj().transpose() @ invR @ SS
        P = 1/denominator[0,0]
        MVDR[ii,jj] = abs(P)
#MVDR = 10*np.log10(MVDR)
end = time.time()
print('Simulation time:', round((end - start), 4), 's')

# --- PLOT
# of performance in xy-plane
fig, ax = plt.subplots() #figsize = (16,9))
pic = ax.pcolormesh(x_scan[:,0,0], y_scan[0,:,0], np.transpose(MVDR), cmap = plt.get_cmap('plasma')) 
fig.colorbar(pic)

validation_check(y_scan, x_scan)

plt.show()