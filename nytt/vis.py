import numpy as np
import matplotlib.pyplot as plt
import math

def delete_mic_data(signal, mic_to_delete):
    #   FUNCTION THAT SETS SIGNALS FROM 'BAD' MICROPHONES TO 0
    signal[:,mic_to_delete] = 0
    return signal

def plot_all_mics(data, N_mics, limitation):
    N_mics = 64 # data.shape[1]

    plt.figure()
    for i in range(N_mics): 
        #plt.plot(data[:,i+2], c=cmap(i/data.shape[1]))
        plt.plot(data[:,i], c=cmap(i/N_mics))
    plt.xlim([0, plot_samples])
    if limitation:
        plt.suptitle('All microphones')
        plt.ylim([-max_value_ok*1.1, max_value_ok*1.1])
        filename = 'all_sigs'
    else:
        plt.title('All microphones, no amplitude limit')
        filename = 'all_sigs_nolim'
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    if save_fig:
        plt.savefig('plots/signal_analyzis/'+filename+'.png', dpi = 500, format = 'png')

def plot_all_individual(data, start_val, a):
    # --- PLOT ---
    #   of all individual signals in subplots, two periods
    #fig, axs = plt.subplots(4,16)
    fig, axs = plt.subplots(8, 8, figsize=(5,7))
    fig.tight_layout(pad = 0.1)
    plt.subplots_adjust(left=0.03,
                bottom=0,
                right=0.97,
                top=0.9,
                wspace=0.1,
                hspace=0.7)
    fig.suptitle("Individual signals, A"+str(a+1), fontsize=FS_title)
    for j in range(8):
        for i in range(8):
            axs[7-j,i].plot(data[start_val:start_val+plot_samples,i+j*8+64*a])
            axs[7-j,i].set_title(str(i+j*8+64*a), fontsize=FS_mics)
            #axs[j,i].plot(data[start_val:start_val+plot_samples,i+j*16+64*a])
            #axs[j,i].set_title(str(i+j*16+64*a), fontsize=8)
            axs[7-j,i].set_ylim(-max_value_ok*1.1, max_value_ok*1.1)
            axs[7-j,i].axis('off')
    if save_fig:
        filename = 'indi_sig_A'+str(a+1)
        plt.savefig('plots/signal_analyzis/'+filename+'.png', dpi = 500, format = 'png')

def plot_selected(plot_mics):
    # --- PLOT ---
    #   of selected microphones, given by plot_mics

    arr_plot_mics = np.array(plot_mics)     # convert plot_mics to numpy array with correct index
    plt.figure()
    for i in range(len(arr_plot_mics)):
        plt.plot(data[:,int(arr_plot_mics[i])], label=f"{arr_plot_mics[i]}", c=cmap(i/(len(plot_mics))))
    plt.xlim([0, plot_samples])
    #plt.ylim([-max_value_ok*1.1, max_value_ok*1.1])
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.suptitle('Selected microphones')
    plt.legend(loc = 4)
    if save_fig:
        filename = 'sel_sig_M'+str(plot_mics)
        plt.savefig('plots/signal_analyzis/'+filename+'.png', dpi = 500, format = 'png')

def plot_energy(mics_FFT):
    # --- PLOT ---
    #   of FFT of several signals
    samples = len(data[:,0])
    t_stop = samples/fs
    t = np.linspace(0,t_stop,samples)
    arr_mics_FFT = np.array(mics_FFT,dtype=int)
    plt.figure()
    for i in range(len(arr_mics_FFT)):
        data_FFT = np.fft.fft(data[:,int(arr_mics_FFT[i])])
        energy = abs(data_FFT)**2
        freq = np.fft.fftfreq(t.shape[-1])
        plt.plot(fs*freq,energy, label=f"{arr_mics_FFT[i]}", c=cmap(i/(len(mics_FFT))))
        #FFT_mic_legend = np.append(FFT_mic_legend,str(arr_mics_FFT[i]))
    plt.suptitle('Energy of selected microphones signals')
    plt.xlabel('Frequency [Hz]')
    plt.legend()


# Plot options
show_plots = 1          # if show_plots = 1, then plots will be shown, if = 0 no plots will be shown
                        #   show (=1) or hide (=0) different type of plots
all_mics = 1       # plots of all microphones
selected = 1       # plots of several selected microphones
all_individual = 0     # plot of individual microphones
energy = 0         # plot of energy of signals
save_fig = 0

FS_title = 20
FS_mics = 15

plot_period = 3         # periods to plot
f0 = 1000               # frequency of recorded sinus signal
fs = 48828              # sampling frequency
samples = fs            # recorded samples

cmap = plt.get_cmap('plasma_r')   # type of color map to use for coloring the different signals

# load data, from file
filename = "recording_up.npy"
data = np.load(filename)
data = data.reshape((3*64, samples))

# ONLY FOR SPECIFIC FILE, SINCE FIRST 2 MICS ARE NOT MIC DATA
#data[0:-2,] = data[2:,]

data = data.T
data = data[4000:,:]    # take out first 5000 samples

plot_samples = math.floor(plot_period*(fs/f0))  # number of samples to plot, to use for axis scaling
max_value_ok = np.max(data)                    # maximum value of data, to use for axis scaling in plots



if all_mics:
    plot_all_mics(data, 64, 1)
    #plot_all_mics(data, 64, 0)

if all_individual:
    plot_all_individual(data, 4000, 0)
    #plot_all_individual(data, 4000, 1)
    #plot_all_individual(data, 4000, 2)

if selected:
    #plot_selected(range(0,8))   # plot mics of row 0
    #plot_selected(range(1*8, 2*8))   # plot mics of row 1
    #plot_selected(range(2*8, 3*8))   # plot mics of row 2
    #plot_selected(range(3*8, 3*8 + 8))   # plot mics of row 3
    #plot_selected(range(4*8, 4*8 + 8))   # plot mics of row 4
    #plot_selected(np.arange(0,63-4+1,8)+4)   # plot mics of row 4'
    plot_selected([61,62])

if energy:
    plot_energy(range(0,8))

# Show all plots
if show_plots:
    plt.show()

