import math
import numpy as np
import matplotlib.pyplot as plt
import calc_r_prime


import config


def weight_index(f):
    # calculates what mode to use, depending on the wavelength of the signal
    d = config.distance              # distance between elements
    wavelength_rel = f*d/c    # relative wavelength to distance between microphone elements
    #print('f: ' + str(f))
    if wavelength_rel>0.1581:
        mode = 1
    elif (wavelength_rel <= 0.1581) and (wavelength_rel > 0.156):
        mode = 2
    elif (wavelength_rel <= 0.156) and (wavelength_rel > 0.0986):
        mode = 3
    elif (wavelength_rel <= 0.0986) and (wavelength_rel > 0.085):
        mode = 5
    elif (wavelength_rel <= 0.085) and (wavelength_rel > 0.07):
        mode = 6
    else:
        mode = 7
    return mode

def adaptive_matrix(rows, columns):
    # Creates the weight matrix
    try:
            # array_audio_signals = np.load(filename)
            weight_matrix = np.load('adaptive_matrix'+'.npy', allow_pickle=True)
            #print("Loading from Memory: " + filename)
    except:
        weight_matrix = np.zeros((7, rows*columns))
        for mode in range(1,7+1):
            weight = np.zeros((1,rows*columns))
            row_lim = math.ceil(rows/mode)
            column_lim = math.ceil(columns/mode)
            for i in range(row_lim):
                for j in range(column_lim):
                    element_index = (mode*i*rows + mode*j) # this calculation could be wrong thanks to matlab and python index :))
                    weight[0,element_index] = 1
            weight_matrix[mode-1,:] = weight
        np.save('adaptive_matrix', weight_matrix)
    return weight_matrix

def plot_single_freq(f_vec, f_single, AF):
        # plots |AF|² for a set of frequencies given by f_single
        freqs = []
        for f in f_single:
            freqs.append(np.argmax(f_vec>f))
        plt.figure()
        for fig in range(len(freqs)):
            freq = f_vec[0,0,freqs[fig],0]
            plt.plot(theta[:,0,0,0]*180/math.pi, AF[:,0,freqs[fig]], linewidth=2, label=str(round(freq))+' Hz')
        plt.xlabel('Theta (deg)')
        plt.ylabel('|AF|²')
        plt.title('Array factor')
        #plt.ylim(np.max(AF3ad_3D_dBi[:,0,freq])-40, np.max(AF3ad_3D_dBi[:,0,freq])+3)
        plt.grid(True)
        plt.legend()

def plot_contour(theta, i_phi, f_vec, AF, min_dB):
    # plots |AF|² for all frequencies in f_vec, and all angles theta
    #   for a set angle phi, where i_phi = 0 gives phi=0deg and i_phi=1 gives phi=90deg
    X, Y = np.meshgrid(theta*180/math.pi, f_vec*10**(-3))
    plt.figure()
    levels = np.linspace(np.max(AF[:,i_phi,:])-min_dB, np.max(AF[:,i_phi,:]), 25)
    plt.contourf(X, Y, np.transpose(AF[:,i_phi,:]), levels, cmap=plt.get_cmap('coolwarm'), extend='min' )
    plt.colorbar(label='|AF|² (dBi)', ticks = None)
    plt.ylabel('Frequency (kHz)')
    plt.xlabel('Theta (deg)')
    plt.title('phi = '+str(round(phi[0,i_phi,0,0])))
    filename = 'AF_'+str(config.active_arrays) + '_array_d=' + str(config.distance*10**(2)) + '_not_adaptive'
    #plt.savefig('plots/'+filename+'.png', dpi = 500, format = 'png')


c = config.c         # speed of sound

F = 501                     # number of points in frequency vector
P = 501                     # number of points in angle vectors
d = config.distance         # distance between elements

fs = config.f_sampling             # sampling frequency


theta0 = 0 * math.pi/180    # scanning angle
phi0 = 0 * math.pi/180      # scanning angle
f_max = 8 * 10**3           # maxium frequency to calc AF for
f_vec = np.linspace(100, f_max, F).reshape(1,1, F,1) 
k_vec = 2*math.pi*f_vec/c

theta = np.linspace(-90,90,P) * math.pi/180     # scan in theta direcion
phi = np.array([1*10**(-14), 90])           
theta = np.reshape(theta, (len(theta),1,1,1))
phi = np.reshape(phi,(1,len(phi),1,1))

r_prime = calc_r_prime.calc_r_prime(config.distance)
x_i = r_prime[0,:].reshape(1,1,1,len(r_prime[0,:]))
y_i = r_prime[1,:].reshape(1,1,1,len(r_prime[1,:]))


n_active_elements = r_prime.shape[1]
print('distance between elements: ' + str(d*10**2) + ' cm')
print('number of elements:', n_active_elements)

#adaptive_weight_matrix = adaptive_matrix(config.rows, config.columns)


# --- 3D matrix of AF, without adaptive ---
AF_mat = np.exp(1j*k_vec*(np.sin(theta)*np.sin(phi) - math.sin(theta0)*np.sin(phi0))*y_i) \
            * np.exp(1j*k_vec*(np.sin(theta)*np.cos(phi) - math.sin(theta0)*np.cos(phi0))*x_i)
AF_mat = np.square(np.absolute(np.sum(AF_mat,axis=3)/n_active_elements))
AF_mat_dB = 10*np.log10(AF_mat)


# --- PLOTS OF SINGLE FREQUENCIES ---
#   the individual frequencies to plot are listed in single_freqs
#   for plot, set plot_single_f = 1 in config.py
if config.plot_single_f:
    single_freqs = [300, 500, 1000, 3000, 7000]
    plot_single_freq(f_vec, np.array(single_freqs), AF_mat)
    plot_single_freq(f_vec, np.array(single_freqs), AF_mat_dB)

min_dB = 20
if config.plot_contourf:
    plot_contour(theta, 1, f_vec, AF_mat_dB, min_dB)
    plot_contour(theta, 0, f_vec, AF_mat_dB, min_dB)


plt.show()