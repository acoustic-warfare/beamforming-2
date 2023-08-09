# VISUALIZE MICROPHONE ARRAY SIGNALS
#

import numpy as np
import matplotlib.pyplot as plt
import math
import config_test as config
import active_microphones as am
import calc_r_prime as crp
#def calc_r_prime(d):
#    half = d/2
#    r_prime = np.zeros((2, config.ELEMENTS))
#    element_index = 0
#    for array in range(config.ACTIVE_ARRAYS):
#        array *= -1
#        for row in range(config.R):
#            for col in range(config.columns):
#                r_prime[0,element_index] = - col * d - half + array*config.columns*d + array*config.sep + config.columns* config.active_arrays * half
#                r_prime[1, element_index] = row * d - config.rows * half + half
#                element_index += 1
#    r_prime[0,:] += (config.active_arrays-1)*config.sep/2
#    active_mics = am.active_microphones()
#
#    r_prime_all = r_prime
#    r_prime = r_prime[:,active_mics]
#
#    if config.plot_setup:
#        fig, ax = plt.subplots(figsize=(config.columns*config.active_arrays/2, config.rows/2))
#        ax.set_box_aspect(int(config.rows)/int(config.columns*config.active_arrays))            # aspect ratio
#        plt.tight_layout()
#        ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)              # remove top and right axis lines
#        color_arr = ['r', 'b', 'g','m']
#        dx = config.distance*0.1
#        dy = config.distance*0.1
#        element = 0
#        for array in range(config.active_arrays):
#            mics_array = [x for x in active_mics if ((0+array*config.rows*config.columns) <= x & x < ((array+1)*config.rows*config.columns))]
#            for mic in range(len(mics_array)): #range(r_prime.shape[1]): #range(int(len(active_mics)/config.active_arrays)):
#                #if active_mics[mic] in range(0,64)
#                x = r_prime[0,element] * 10**2
#                y = r_prime[1,element] * 10**2
#                ax.scatter(x, y, color = color_arr[array])
#                plt.text(x-dx, y+dy, str(active_mics[element]))
#                element += 1
#        #plt.title('Array setup')
#        ax.set_xlabel('x (cm)')
#        ax.set_ylabel('y (cm)')
#
#    if config.plot_setup:
#        fig, ax = plt.subplots(figsize=(12, 5))
#        ax.set_box_aspect(int(config.rows)/int(config.columns*config.active_arrays))                            # aspect ratio
#        plt.tight_layout()
#        ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)      # remove top and right axis lines
#        element = 0
#        color_arr = ['r', 'b', 'g','m']
#        dx = config.distance*0.1 * 10**2
#        dy = config.distance*0.2 * 10**2
#        S = 200
#        for array in range(config.active_arrays):
#            for mic in range(config.rows*config.columns):
#                mic += array*config.rows*config.columns
#                x = r_prime_all[0,element] * 10**2
#                y = r_prime_all[1,element] * 10**2
#                if mic in active_mics:
#                    ax.scatter(x, y, color = color_arr[array], s=S)
#                else:
#                    ax.scatter(x, y, color = 'none', edgecolor=color_arr[array], linewidths = 1, s=S)
#                #plt.text(x-dx, y+dy, str(element), fontsize = 12)
#                element += 1
#        #plt.title('Array setup')
#        FS = 22 # fontsize
#        ax.set_xlabel('x (cm)',fontsize = FS)
#        ax.set_ylabel('y (cm)', fontsize = FS)
#        plt.xticks(fontsize= FS), plt.yticks(fontsize= FS)
#
#        if save_fig:
#            filename = 'array_setup_mode'+ str(config.mode)
#            plt.savefig('plots/setup/'+filename+'.png', dpi = 500, format = 'png')
#    return r_prime_all, r_prime
#

def delete_mic_data(signal, mic_to_delete):
    #   sets signals from selected microphones in mic_to_delete to 0
    new_signal = signal.copy()
    new_signal[:,mic_to_delete] = 0
    return new_signal

def plot_all_mics(data, N_mics = 4*64, amp_lim=0, samp_lim = 0):
    #   plot of data from all microphones
    plt.figure()
    for i in range(N_mics): 
        plt.plot(data[:,i], c=cmap(i/N_mics))
    if amp_lim:
        #plt.suptitle('All microphones', fontsize = FS_title)
        plt.ylim([-max_value*1.1, max_value*1.1]); filename = 'all_sigs' + '_del_mics'
    else:
        #plt.title('All microphones, no amplitude limit', fontsize = FS_title)
        filename = 'all_sigs_nolim'
    if samp_lim: plt.xlim([0, plot_samples])
    plt.xlabel('Samples', fontsize = FS_label), plt.ylabel('Amplitude', fontsize = FS_label)
    plt.xticks(fontsize= FS_tics), plt.yticks(fontsize= FS_tics)
    plt.tight_layout()
    if save_fig:
        plt.savefig('plots/signal_analyzis/'+filename+'.png', dpi = 500, format = 'png')

def plot_all_individual(data, start_val, a):
    #   plot of all individual signals in subplots, two periods
    fig, axs = plt.subplots(rows, cols, figsize=(5,7))
    fig.tight_layout(pad = 0.1)
    plt.subplots_adjust(left=0.03, bottom=0, right=0.97,
                        top=0.9, wspace=0.1, hspace=0.7)
    fig.suptitle("Individual signals, A"+str(a+1), fontsize=FS_title)
    for j in range(rows):
        for i in range(cols):
            axs[7-j,i].plot(data[start_val:start_val+plot_samples, \
                                 i+j*cols+array_elements*a], \
                                 c=cmap((i+j*cols)/(array_elements)))
            axs[7-j,i].set_title(str(i+j*cols+array_elements*a), fontsize=FS_mics)
            axs[7-j,i].set_ylim(-max_value*1.1, max_value*1.1)
            axs[7-j,i].axis('off')
    if save_fig:
        filename = 'indi_sig_A'+str(a+1)
        plt.savefig('plots/signal_analyzis/'+filename+'.png', dpi = 500, format = 'png')

def plot_selected(data, plot_mics, amp_lim=0, samp_lim = 0, subtitle=''):
    # --- PLOT ---
    #   of selected microphones, given by plot_mics
    arr_plot_mics = np.array(plot_mics)     # convert plot_mics to numpy array with correct index
    plt.figure()
    for i in range(len(arr_plot_mics)):
        plt.plot(data[:,int(arr_plot_mics[i])], label=f"{arr_plot_mics[i]}", c=cmap(i/(len(plot_mics)-1+0.1)))
    if amp_lim: # amplitude limitation
        #plt.suptitle('Selected microphones', fontsize = FS_title)
        plt.ylim([-max_value*1.1, max_value*1.1])
    else: # no amplitude limitation
        #plt.suptitle('Selected microphones, no amplitude limit', fontsize = FS_title)
        filename = 'sel_sig_nolim_M'+str(plot_mics)
    if samp_lim: # sample limitation
        plt.xlim([0, plot_samples])
    
    if subtitle != '': plt.title(subtitle, fontsize = FS_title*0.75) # extra subtitle
     
    if len(plot_mics) < 12:
        plt.legend(loc = 4, fontsize = FS_label)
    plt.xlabel('Samples',fontsize = FS_label),  plt.ylabel('Amplitude',fontsize = FS_label)
    plt.xticks(fontsize=FS_tics),               plt.yticks(fontsize=FS_tics)
    plt.tight_layout()
    if save_fig:
        filename = 'sel_sig_M'+str(plot_mics)
        plt.savefig('plots/signal_analyzis/'+filename+'.png', dpi = 500, format = 'png')

def plot_energy(data, mics_FFT):
    #   plot of FFT of several signals
    samples = len(data[:,0])
    t_stop = samples/fs
    t = np.linspace(0,t_stop,samples)
    arr_mics_FFT = np.array(mics_FFT,dtype=int)
    plt.figure()
    for i in range(len(arr_mics_FFT)):
        data_FFT = np.fft.fft(data[:,int(arr_mics_FFT[i])])
        energy = abs(data_FFT)**2
        freq = np.fft.fftfreq(t.shape[-1])
        plt.plot(fs*freq, energy, label=f"{arr_mics_FFT[i]}", c=cmap(i/(len(mics_FFT))))
    
    plt.suptitle('Energy of selected microphones signals', fontsize = FS_title)
    plt.xlabel('Frequency [Hz]', fontsize = FS_label)
    plt.xticks(fontsize= FS_tics), plt.yticks(fontsize= FS_tics)
    plt.legend()
    plt.tight_layout()

def plot_array(ignore_mic = [], mode=1):
    #   plot array setup
    #   displays mode, and if microphones are ignored or not
    active_mics = am.active_microphones(mode, 'all')  # load active microphones
    r_prime_all, r_prime = crp.calc_r_prime(config.ELEMENT_DISTANCE)

    fig, ax = plt.subplots(figsize=(config.COLUMNS*config.ACTIVE_ARRAYS/2, config.ROWS/2))
    ax.set_box_aspect(int(config.ROWS)/int(config.COLUMNS*config.ACTIVE_ARRAYS))            # aspect ratio
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)              # remove top and right axis lines
    plt.tight_layout()

    color_arr = ['r', 'b', 'g', 'orange']   # element colors for the separate arrays
    dx = 0.001                              # offset of element index text
    dy = 0.001                              # offset of element index text

    element = 0
    for array in range(config.ACTIVE_ARRAYS):
        plt.title('Array setup')
        for mic in range(config.ROWS*config.COLUMNS):
            x = r_prime_all[0,element]
            y = r_prime_all[1,element]
            if element in zero_mics: #(np.sum(data[element], axis = 0)):
                ax.scatter(x, y, color = 'none', edgecolor='k', linewidths = 1)
            elif element in ignore_mic:
                ax.scatter(x, y, color = '#fdff38', edgecolor='#fdff38', linewidths = 1)
            #elif element in other_spikes:
            #    ax.scatter(x, y, color = 'none', edgecolor='r', linewidths = 1)
            #elif element in spike_05_mics:
                #ax.scatter(x, y, color = 'none', edgecolor='r', linewidths = 1)
            #elif element in spike_025_mics:
                #ax.scatter(x, y, color = 'none', edgecolor='r', linewidths = 1)
            elif element in active_mics:
                ax.scatter(x, y, color = color_arr[array])
            else:
                ax.scatter(x, y, color = 'none', edgecolor=color_arr[array], linewidths = 1)
            plt.text(x-dx, y+dy, str(element))
            element += 1


# name of file of recordings, should be .npy file
filename = 'sine1k.npy' 

# Plot options
show_plots = 1              # if show_plots = 1, then plots will be shown, if = 0 no plots will be shown
                            #   show (=1) or hide (=0) different type of plots
all_mics = 0                # plots of all microphones
selected = 1                # plots of several selected microphones
all_individual = 1          # plot of individual microphones
energy = 0                  # plot of energy of signals
plot_array_setup = 1        # plot the array setup
save_fig = 0                # if 1, save figure

color_map_type = 'jet'      # 'plasma', 'cool', 'inferno'
plot_period = 3             # periods to plot
f0 = 1000                   # frequency of recorded sinus signal
fs = 48828                  # sampling frequency

plot_samples = math.floor(plot_period*(fs/f0))  # number of samples to plot, to use for axis scaling

# Plot text sizes
FS_title = 18
FS_mics = 15
FS_label = 15
FS_tics = 15

rows = config.ROWS                      # number of rows in one array
cols = config.COLUMNS                   # number of columns in one array
array_elements = rows*cols              # total elements in one array

cmap = plt.get_cmap(color_map_type)     # type of color map to use for coloring the different signals

# load data, from file
directory = 'recordings'
data = np.load(directory+'/'+ filename)
data = data.T
data = data[2000:,:]                # take out selected samples

# find the mics that sends zeros
data_sum = np.sum(data,axis=0)
zero_mics = np.where(data_sum[np.arange(0,64*3)]==0)[0]

# Different max values of the microphone amplitudes
max_value = np.max(np.abs(data))                    # maximum value of all microphones
max_each_mic = np.max(data[:,:rows*cols*config.ACTIVE_ARRAYS], axis=0)
max_value_mean = np.mean(max_each_mic)
max_value_median = np.median(np.sort(max_each_mic))
print('Max value:', max_value)
print('Mean of maximum value of all microphones',max_value_mean)
print('Median of maximum value of all microphones',max_value_median)
max_value = max_value_median*1.5 # use the median of the max value of each microphone as an amplitude limit for plots

# Find mics giving higher values than normal
high_val_mics = np.where(np.argmax(np.abs(data) > max_value_median*2, axis = 0)>0)[0] # all microphones with values over max_value_median*2
#spike_05_mics = np.where(np.argmax(np.abs(data)  > 0.45, axis = 0)>0)[0]        # microphones with values around 0.5 (related to bit errors)
#spike_025_mics = np.where(np.argmax(np.abs(data-0.25) < 0.02, axis = 0)>0)[0]   # microphones with values around 0.25 (related to bit errors)

## microphones in spike_05_mics high values that are not in spike_05_mics or spike_025_mics
#other_spikes = np.setdiff1d(high_val_mics, spike_05_mics)
#other_spikes = np.setdiff1d(other_spikes, spike_025_mics)

# manually selected bad mics where the signals should be set to zero
delete_mics = [23, 153, 154, 155, 169, 170, 171, 172, 188, 189, 190]
delete_mics = np.sort(delete_mics)

# collect all microphones that should be ignored in the beamforming algorithm
ignored_mics = np.append(delete_mics, zero_mics)
ignored_mics = np.sort(ignored_mics)
np.save('unused_mics', ignored_mics)        # save ignored microphones to .npy file, to be loaded into the beamformer program

#print('ignored microphones:', ignored_mics)
print('high val mics', high_val_mics)
#print('0.25 spike', spike_025_mics)
#print('0.5 spike', spike_05_mics)
#print('other', other_spikes)
print('zero mics,', zero_mics)

# delete data (set data to 0) for the specified microphones
data_ignored = delete_mic_data(data, ignored_mics)

# plot array setup
if plot_array_setup:
    plot_array(delete_mics)

if all_mics:
    #plot_all_mics(data, amp_lim=1, samp_lim=1)
    plot_all_mics(data_ignored, N_mics = 3*64, amp_lim=1, samp_lim=1)

if all_individual:
    plot_all_individual(data, 0, 0)
    plot_all_individual(data, 0, 1)
    plot_all_individual(data, 0, 2)

# plot selected microphones
if selected:
    #plot_selected(data, np.sort([153, 169, 170, 190, 191, 171, 172, 155, 154]), amp_lim=1, samp_lim=1)
    #plot_selected(data, high_val_mics, amp_lim=0, samp_lim=0)
    #plot_selected(data,[188,189,190])
    #plot_selected(data,[154])
    #plot_selected(data,[169,170,171])
    #plot_selected(data, np.arange(1,64))
    #plot_selected(data, [23])
    #plot_selected(data,np.arange(64,128))
    #plot_selected(data,np.arange(128,192))
    #plot_selected(data, [33, 42, 43, 49])
    plot_selected(data, [33, 42, 43, 49])
    plot_selected(data, [173, 149])

# plot energy spectrum
if energy:
    plot_energy(data, high_val_mics)

# show all plots
if show_plots:
    plt.show()