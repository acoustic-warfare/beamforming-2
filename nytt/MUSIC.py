import numpy as np
import matplotlib.pyplot as plt

import time
import math
import config
import generate_signals
import calc_r_prime
import active_microphones as am

def generate_filename():
    if config.sources == 1:
        filename ='emul_'+ 'samples='+str(config.samples) + '_start'+ str(config.f_start1)+'Hz'+'_end'+str(config.f_end1)+'Hz'+\
            '_theta='+str(config.theta_deg1)+'_phi='+str(config.phi_deg1)+ '_AWD=' + str(config.away_distance) + \
            '_E'+ str(config.rows*config.columns) + '_A'+str(config.active_arrays)
    elif config.sources == 2:
        filename ='emul_'+'samples='+str(config.samples) + \
        '_start'+str(config.f_start1)+str(config.f_start2)+'Hz'+\
        '_end'+str(config.f_end1)+str(config.f_end2)+'Hz'+ \
        '_theta='+str(config.theta_deg1)+str(config.theta_deg2)+ \
        '_phi='+str(config.phi_deg1)+str(config.phi_deg2)+\
        '_AWD=' + str(config.away_distance) +\
        '_E'+ str(config.rows*config.columns) + '_A'+str(config.active_arrays)
    return filename

def validation_check(y_scan, x_scan):
    # Validation check
    xy_val_check = np.zeros((config.x_res,config.y_res))
    z = config.z_scan
    theta_s = np.array([config.theta_deg1, config.theta_deg2])*math.pi/180
    phi_s = np.array([config.phi_deg1, config.phi_deg2])*math.pi/180
    r_scan = z/np.cos(theta_s)

    for source_ind in range(config.sources):
            x_s = r_scan[source_ind] * math.sin(theta_s[source_ind]) * math.cos(phi_s[source_ind])
            y_s = r_scan[source_ind] * math.sin(theta_s[source_ind]) * math.sin(phi_s[source_ind])
            x_ind = (np.abs(x_scan[:,0,0] - x_s)).argmin()
            y_ind = (np.abs(y_scan[0,:,0] - y_s)).argmin()
            xy_val_check[x_ind,y_ind] = 1
            print(x_scan.shape)
            print(y_scan.shape)

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
    signal = generate_signals.emulated_signals


# source with noise
signal += np.random.uniform(0,np.max(signal)/5, (config.elements, config.samples)).T    # add noise to signals
signal = signal[:,am.active_microphones()]                                              # take out signals from active microphones
signal = signal/np.max(signal)                                                          # normalize so signals are between -1 and 1

# No source, only noise
# signal[:,:] = 0 + np.random.uniform(0,np.max(signal)/2, (config.elements, config.samples)).T
# signal = signal/np.max(signal)

# Figure of the signal
#plt.figure()
#plt.plot(signal)
#print('max_sig',np.max(signal))

z_scan = config.z_scan
AS = config.aspect_ratio
alpha = config.alpha #config.alpha  # total scanning angle (bildvinkel) in theta-direction [degrees], from config

x_res = config.x_res #config.x_res  # resolution in x, from config
y_res = config.y_res  # resolution in y, from config

# Calculations for scanning window
r_prime = calc_r_prime.calc_r_prime(config.distance)  # matrix holding the xy positions of each microphone
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


# --- code based on matlab for DOA MUSIC --
fs = config.f_sampling
N = 512                         # number of samples
M = config.elements             # number of array elements
P = config.sources              # number of sources
c = config.c    
f = 1.5 * 10**3                 # frequency that the music algorithm is based on
k = 2*math.pi*f / c             # wavenumber
d = config.distance             # distance between elements

psi =  (x_scan*x_i + y_scan*y_i) / r_scan 
S = np.exp(-1j*k*psi)   # steering matrix


# MUSIC ALGORITHM STARTS HERE
start = time.time()
x = np.fft.rfft(signal.T,axis=1)    # FFT each signal (signal for music algorithm is assumed to be complex)
R = (x @ x.conj().transpose())/ N   # calculate covariance matrix and normalize with number of samples
N, V = np.linalg.eig(R)             # calculate eigenvalues (N) and eigenvectors (V) of R
idx = N.argsort()[::-1]             # sorted indexes based on eigenvalues
V = V[:, idx]                       # sort eigenvectors in correct order
VV = V[:,P:M]                       # take out the used eigenvectors

music = np.zeros((x_res,y_res))     # heatmap for music algorithm
for jj in range(y_res):
    for ii in range(x_res):
        #SS = np.exp(-1j*k*psi[ii,jj,:])                     # stearing matrix for a specific direction 
        SS = S[ii,jj,:]                                     # stearing matrix for a specific direction 
        PP1 = (SS @ VV)                                     # matrix multiplication 
        PP2 = VV.conj().transpose() @ SS.conj().transpose() # matrix multiplication 
        P = PP1 @ PP2                                       # matrix multiplication 
        music[ii,jj] = abs(1/P)                             # music value for a specific direction
#music = 10*np.log10(music)


## tests
#lmax = np.max(music)    # maximum music value
#print('lmax:', lmax)
#if np.max(music)<1:
#    print('\t lmax<1')
#    music += 1
#music **= 2
##print(music>(np.max(music)*0.7))
#music = music*(music>(np.max(music)*0.5))
#lmax = np.max(music)
#print('\t lmax after:', lmax)

end = time.time()
print('Simulation time:', round((end - start), 4), 's')


# --- PLOT
#   showing the difference of theta between:
#        linearly spaced angels (speharical scanning)
#        non-linearly spaced angels ("rectangular" scanning)
#theta1 = np.linspace(-alpha/2, alpha/2, x_res)/180*math.pi
#theta_x = np.arccos(z_scan/np.sqrt(x_scan**2 + z_scan**2))[:,0,:]
#theta_x[0:round(x_res/2)] *= -1
#theta_y = np.arccos(z_scan/np.sqrt(y_scan**2 + z_scan**2))[0,:,:]
#theta_y[0:round(y_res/2)] *= -1
#print('x_scan', x_scan)
#print('y_scan',y_scan)
#print('theta_x', theta_y)
#print('theta_y', theta_y)
#plt.figure()
#plt.plot(theta1*180/math.pi, label = 'theta1')
#plt.plot(theta_x*180/math.pi, label = 'theta2')
#plt.plot(theta_y*180/math.pi, label = 'theta3')
#plt.legend()

## --- PLOT
##   showing psi2 for only x-axis and mic
##   vs psi3 for x-axis, y-axis and mic
#plt.figure()
#plt.plot(psi2[:,:], label='psi2')
#plt.plot(psi3[:,round(y_res/2),:],'--',label = 'psi3')
#plt.legend()
#
## --- PLOT
##   showing psi2 for only x-axis and mic
##   vs psi3 for x-axis, y-axis and mic
#plt.figure()
#plt.plot(psi3[round(x_res/2),:,:],'--',label = 'psi3')
#plt.legend()

plot_x_y = 0
if plot_x_y:
    # --- PLOT
    # of performance along the x-axis
    plt.figure()
    plt.plot(x_scan[:,0,0], music[:,round(y_res/2)])
    #plt.plot(theta*180/math.pi, music)
    plt.title('performance along x-axis, when y=0')

    # --- PLOT
    # of performance along the y-axis
    plt.figure()
    plt.plot(y_scan[0,:,0], music[round(x_res/2),:])
    #plt.plot(theta*180/math.pi, music)
    plt.title('performance along y-axis, when x=0')


# --- PLOT
# of performance in xy-plane
fig, ax = plt.subplots() #figsize = (16,9))
pic = ax.pcolormesh(x_scan[:,0,0], y_scan[0,:,0], np.transpose(music), cmap = plt.get_cmap('plasma'))#,shading='gouraud') # heatmap summed over all frequencies
fig.colorbar(pic)

validation_check(y_scan, x_scan)

plt.show()