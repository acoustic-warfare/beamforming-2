# ARRAY FACTOR
#
# Program that calculates and plots the array factor for a span of frequencies and angles
#
# for a microphone array with properties specified in config.py
#

import math
import numpy as np
import matplotlib.pyplot as plt
import calc_r_prime
import config_test as config
import calc_mode_matrix
import active_microphones as am

def calc_HPBW2(AF):
    # calculate half-power beam-width
    # assumes AF is normalized and in dB
    HPBW = AF+3 > 0     # set power values larger than half-power to 1, all lower values are 0

    idx_phi0 = np.argmax(HPBW[round(P/2):,0,:]==0, axis = 0)    # index of HP values when phi=0
    idx_phi90 = np.argmax(HPBW[round(P/2):,1,:]==0, axis = 0)   # index of HP values when phi=90
    deg180_0 = np.where(idx_phi0 == 0)
    deg180_90= np.where(idx_phi90 == 0)
    idx_phi0[deg180_0] = len(HPBW[round(P/2):,1,:])
    idx_phi90[deg180_90] = len(HPBW[round(P/2):,1,:])

    HPWB_phi0 = theta[round(P/2)-idx_phi0,0,0,0]*180/math.pi
    HPWB_phi90 = theta[round(P/2)-idx_phi90+1,0,0,0]*180/math.pi
    HPWB_phi0 = abs(HPWB_phi0)*2
    HPWB_phi90 = abs(HPWB_phi90)*2
    return HPBW, HPWB_phi0, HPWB_phi90

def plot_single_freq(f_vec, f_single, AF):
    # plots |AF|² for a set of frequencies given by f_single
    freqs = []
    for f in f_single:
        freqs.append(np.argmax(f_vec>f))
    plt.figure()
    for fig in range(len(freqs)):
        freq = f_vec[0,0,freqs[fig],0]
        plt.plot(theta[:,0,0,0]*180/math.pi, AF[:,0,freqs[fig]], linewidth=2, label=str(round(freq))+' Hz')
    plt.xlabel('Theta (deg)', fontsize = FS_label)
    plt.ylabel('|AF|²', fontsize = FS_label)
    plt.title('Array factor')
    plt.xticks(fontsize=FS_ticks), plt.yticks(fontsize=FS_ticks)
    plt.grid(True)
    plt.legend()

def AF_mode(AF, mode):
    # calculates AF for a given mode
    active_mics = am.active_microphones(mode, 'all')
    AF = np.square(np.absolute(np.sum(AF[:,:,:,active_mics],axis=3)/len(active_mics)))
    return AF

def plot_mode_single_freq(f_vec, f, AF, min_dB = False):
    # plots |AF|² for several modes at one frequency
    f_ind = np.argmax(f_vec>=f)
    plt.figure()
    for A, AF_mode in enumerate(AF):
        AF_f = AF_mode[:,0,f_ind]
        plt.plot(theta[:,0,0,0]*180/math.pi, AF_f, linewidth=2, label='mode '+str(A+1),  c =cmap(A/len(AF)))
    plt.xlabel('Theta (deg)', fontsize = FS_label)
    plt.ylabel('|AF|²', fontsize = FS_label)
    plt.title('Array factor')
    plt.xticks(fontsize=FS_ticks), plt.yticks(fontsize=FS_ticks)
    if min_dB:
        plt.ylim(-20, 3)
    plt.xlim(np.min(theta)*180/math.pi, np.max(theta)*180/math.pi)
    plt.grid(True)
    plt.legend()

def plot_contour(theta, i_phi, f_vec, AF, min_dB, filename_extra = ''):
    # plots |AF|² for all frequencies in f_vec, and all angles theta
    #   for a set angle phi, where i_phi = 0 gives phi=0deg and i_phi=1 gives phi=90deg
    X, Y = np.meshgrid(theta*180/math.pi, f_vec*10**(-3))
    plt.figure()
    levels = np.linspace(np.max(AF[:,i_phi,:])-min_dB, np.max(AF[:,i_phi,:]), 25)
    plt.contourf(X, Y, np.transpose(AF[:,i_phi,:]), levels, cmap=plt.get_cmap(color_map_type), extend='min' )
    cb = plt.colorbar() #ticklocation='left')
    cb.ax.tick_params(labelsize=FS_label)
    cb.ax.set_title('|AF|² (dBi)', size = FS_label)
    plt.ylabel('Frequency (kHz)', fontsize = FS_label)
    plt.xlabel('Theta (deg)', fontsize = FS_label)
    plt.xticks(fontsize=FS_ticks), plt.yticks(fontsize=FS_ticks)
    #plt.title('phi = '+str(round(phi[0,i_phi,0,0])))
    #filename = 'AF_'+'A' + str(config.active_arrays) + filename_extra
    #plt.savefig('plots/array_factor/'+filename+'.png', dpi = 500, format = 'png')


def plot_HPBW(HPBW, theta, f_vec):
    # Plot Half-Power Beam-Width in a colormap,
    # for a specific mode
    fig, ax = plt.subplots()
    pic = ax.pcolormesh(theta*180/math.pi, f_vec*10**(-3), np.transpose(HPBW[:,0,:]) , cmap=plt.get_cmap('Greys'))
    plt.ylabel('Frequency (kHz)', fontsize = FS_label)
    plt.xlabel('Theta (deg)', fontsize = FS_label)

def plot_HPBW_angles(HPBW_angles, labels=None):
    # Plot the total Half-Power Beam-Width angle in a regular plot
    # for several modes, given by HPBW_angles
    plt.figure()
    for i, angles in enumerate(HPBW_angles):
        plt.plot(angles, f_vec[0,0,:,0]*10**(-3), c=cmap(i/(len(HPBW_angles))))
    plt.ylabel('Frequency (kHz)', fontsize=FS_label)
    plt.xlabel('Half power beam-width (deg)', fontsize=FS_label)
    plt.xticks(fontsize=FS_ticks), plt.yticks(fontsize=FS_ticks)
    plt.xlim(0,180)
    plt.ylim(100*10**(-3), f_max*10**(-3))
    if labels != None:
        plt.legend(labels, fontsize = FS_label)
    plt.grid(True)
    #filename = 'AF_A3_HPBW_all_modes'
    #plt.savefig('plots/array_factor/'+filename+'.png', dpi = 500, format = 'png')



theta0 = 0 * math.pi/180    # main beam angle
phi0 = 0 * math.pi/180       # main beam angle
f_max = 15 * 10**3           # maxium frequency to calc AF for
F = 101                      # number of points in frequency vector
P = 501                      # number of points in angle vectors

# Plot settings
color_map_type = 'coolwarm'  # type of colormap for contour plot of AF
cmap = plt.get_cmap('jet')   # colormap for plotting HPBW angles
# Plot text fontsize
FS_label = 13
FS_ticks = 13

c = config.PROPAGATION_SPEED            # speed of sound
d = config.ELEMENT_DISTANCE             # distance between elements
fs = config.SAMPLE_RATE                 # sampling frequency
f_vec = np.linspace(100, f_max, F).reshape(1,1, F,1) # frequency vector
k_vec = 2*math.pi*f_vec/c                            # wave number vector

theta = np.linspace(-90,90,P) * math.pi/180     # scaning angles of main beam in theta direcion
phi = np.array([0+1*10**(-14), 90])             # scaning angles of main beam in phi direcion
theta = theta.reshape(len(theta),1, 1,1)       
phi = np.reshape(phi,(1,len(phi),1,1))

r_prime, r_prime_ = calc_r_prime.calc_r_prime(config.ELEMENT_DISTANCE)
x_i = r_prime[0,:].reshape(1,1,1,len(r_prime[0,:]))
y_i = r_prime[1,:].reshape(1,1,1,len(r_prime[1,:]))

n_active_elements = r_prime.shape[1]
print('distance between elements:', str(d*10**2) + ' cm')
print('number of active elements:', n_active_elements)

bad_mics = np.array([70,  71,  73,  74,  75,  76,  77,  78,  79, 153, 154, 155, 169, 170, 171, 172, 184, 185,
186, 187, 188, 189, 190])

# --- AF ---
#   Calculate the base matrix for the array factor
#   AF_mat [theta, phi, f_vec, mics]
AF_mat = np.exp(1j*k_vec*(np.sin(theta)*np.sin(phi) - math.sin(theta0)*np.sin(phi0))*y_i) \
            * np.exp(1j*k_vec*(np.sin(theta)*np.cos(phi) - math.sin(theta0)*np.cos(phi0))*x_i)


## AF for the real setup (including mics deleted and mics sending zeros)
#   where the mode is decided in the config file,
#   and bad microphones stored in unused_mics.npy
try:
    real_used_mics = am.active_microphones(config.mode, 'only good') # load microphones that are used in real setup
    AF_real = AF_mat[:,:,:,real_used_mics].copy()
    n_used_mics = len(real_used_mics)
    AF_real = np.square(np.absolute(np.sum(AF_real,axis=3)/n_used_mics))
    AF_real = 10*np.log10(AF_real)
    print(real_used_mics)
except:
    print('No infomration about real microphone setup')


## Adaptive, different modes for different frequencies
AF_ad, n_active_mics = calc_mode_matrix.mode_matrix(AF_mat, f_vec[0,0,:,0], matrix_type = 'AF')
n_mics_ad = n_active_mics.reshape(1,1,F,1)
AF_ad = np.square(np.absolute(np.sum(AF_ad/n_mics_ad,axis=3)))
AF_ad_dB = 10*np.log10(AF_ad)


calc_modes = 1
calc_HPBW = 1
plotHPBW = 1
plot_contourf = 0
print(AF_mat.shape)
if calc_modes:
    ## Not adaptive to frequency, but one mode for all frequencies
    AF_m1 = 10*np.log10(AF_mode(AF_mat, 1))                     # mode 1
    AF_m2 = 10*np.log10(AF_mode(AF_mat, 2))                     # mode 2
    AF_m3 = 10*np.log10(AF_mode(AF_mat, 3))                     # mode 3
    AF_m4 = 10*np.log10(AF_mode(AF_mat, 4))                     # mode 4
    AF_m5 = 10*np.log10(AF_mode(AF_mat, 5))                     # mode 5
    AF_m6 = 10*np.log10(AF_mode(AF_mat, 6))                     # mode 6
    AF_m7 = 10*np.log10(AF_mode(AF_mat, 7))                     # mode 7

# Calculate half power beamwidth
if calc_HPBW:
    HPBW_m1, HPBW_0_m1, HPBW_90_m1 = calc_HPBW2(AF_m1)          # mode 1
    HPBW_m2, HPBW_0_m2, HPBW_90_m2 = calc_HPBW2(AF_m2)          # mode 2
    HPBW_m3, HPBW_0_m3, HPBW_90_m3 = calc_HPBW2(AF_m3)          # mode 3
    HPBW_m4, HPBW_0_m4, HPBW_90_m4 = calc_HPBW2(AF_m4)          # mode 4
    HPBW_m5, HPBW_0_m5, HPBW_90_m5 = calc_HPBW2(AF_m5)          # mode 5
    HPBW_m6, HPBW_0_m6, HPBW_90_m6 = calc_HPBW2(AF_m6)          # mode 6
    HPBW_m7, HPBW_0_m7, HPBW_90_m7 = calc_HPBW2(AF_m7)          # mode 7
    HPBW_ad, HPBW_0_ad, HPBW_90_ad = calc_HPBW2(AF_ad_dB)       # adaptive

    HPBW_real, HPWB_0_real, HPWB_90_real = calc_HPBW2(AF_real)   # real configuration


if plotHPBW:
    plot_HPBW(HPBW_real, theta[:,0,0,0], f_vec[0,0,:,0])
    plot_HPBW(HPBW_m1, theta[:,0,0,0], f_vec[0,0,:,0])
    plot_HPBW(HPBW_m2, theta[:,0,0,0], f_vec[0,0,:,0])
    plot_HPBW(HPBW_m3, theta[:,0,0,0], f_vec[0,0,:,0])
    plot_HPBW(HPBW_m4, theta[:,0,0,0], f_vec[0,0,:,0])
    plot_HPBW(HPBW_m5, theta[:,0,0,0], f_vec[0,0,:,0])
    plot_HPBW(HPBW_m6, theta[:,0,0,0], f_vec[0,0,:,0])
    plot_HPBW(HPBW_m7, theta[:,0,0,0], f_vec[0,0,:,0])
    plot_HPBW(HPBW_ad, theta[:,0,0,0], f_vec[0,0,:,0])


# --- PLOTS OF SINGLE FREQUENCIES ---
#   the individual frequencies to plot are listed in single_freqs
#   for plot, set plot_single_f = 1 in config.py
plot_single_f = 0
if plot_single_f:
    single_freqs = [300, 500, 1000, 3000, 7000]
    plot_single_freq(f_vec, np.array(single_freqs), AF_m1)



min_dB = 20
if plot_contourf:
    #plot_contour(theta, 0, f_vec, AF_m1, min_dB, filename_extra = '_phi=' + str(theta0*180/math.pi) + '_m1')
    #plot_contour(theta, 0, f_vec, AF_m2, min_dB, filename_extra = '_phi=' + str(theta0*180/math.pi) + '_m2')
    plot_contour(theta, 0, f_vec, AF_m1, min_dB)
    plot_contour(theta, 0, f_vec, AF_real, min_dB)
    #plot_contour(theta, 0, f_vec, AF_m4, min_dB, filename_extra = '_phi=' + str(theta0*180/math.pi) + '_m4')
    #plot_contour(theta, 0, f_vec, AF_m7, min_dB, filename_extra = '_phi=' + str(theta0*180/math.pi) + '_m7')
    #plot_contour(theta, 0, f_vec, AF_ad_dB, min_dB, filename_extra = '_phi=' + str(theta0*180/math.pi) + '_ad')
    #plot_contour(theta, 0, f_vec, AF_real, min_dB)


plt.show()