import numpy as np
import matplotlib.pyplot as plt
import math


def delete_mic_data(signal, mic_to_delete):
    #   FUNCTION THAT SETS SIGNALS FROM 'BAD' MICROPHONES TO 0
    signal[:,mic_to_delete] = 0
    return signal

def main():
    # Choose the mics that might give bad values
    bad_mics = [68, 70, 71, 72]

    # Plot options
    show_plots = 1      # if show_plots = 1, then plots will be shown, if = 0 no plots will be shown
    #   show (=1) or hide (=0) different type of plots
    plot_all_mics = 1       # plots of all microphones
    plot_selected = 1       # plots of several selected microphones
    plot_individual = 1     # plot of individual microphones
    plot_energy = 1         # plot of energy of signals

    plot_period = 5         # periods to plot
    f0 = 1000               # frequency of recorded sinus signal
    fs = 48828
    cmap = plt.get_cmap('plasma')   # type of color map to use for coloring the different signals

    # load data, from file
    filename = "out_recording2.npy"
    data = np.load(filename)
    data = data.reshape((3*64, fs))
    data = data.T 

    data = data[:5000,:]    # take out first 5000 samples

    #data_sum = np.sum(data,axis=0)
    #print(data_sum.shape)
    #data_sum = 1*(abs(data_sum)>0)
    #non_zero = np.where(data_sum==1)[0]

    plot_samples = math.floor(plot_period*(fs/f0))  # number of samples to plot, to use for axis scaling
    max_value_ok = 1.25*10**5                       # maximum value of data, to use for axis scaling in plots

    if plot_all_mics:
        # --- PLOT ---
        #   plot of all microphones, with amplitude limit of max_value
        plt.figure()
        # ONLY microphones from the first array
        for i in range(64): #range(data.shape[1]):
            #plt.plot(data[:,i+2], c=cmap(i/data.shape[1]))
            plt.plot(data[:,i+2], c=cmap(i/64))
        plt.xlim([0, plot_samples])
        plt.ylim([-max_value_ok*1.1, max_value_ok*1.1])
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.suptitle('All microphones')


        # --- PLOT ---
        #   plot of all microphones, without any amplitude limit
        plt.figure()
        for i in range(data.shape[1]):
            plt.plot(data[:,i], c=cmap(i/data.shape[1]))
        plt.xlim([0, plot_samples])
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.title('All microphones, no amplitude limit')


    # --- PLOT ---
    #   of all individual signals in subplots, two periods
    if plot_individual:
        a = 0               # first array
        fig, axs = plt.subplots(4,16)
        fig.suptitle("Individual signals", fontsize=16)
        start_val = 4000
        for j in range(4):
            for i in range(16):
                axs[j,i].plot(data[start_val:start_val+plot_samples,i+j*16+64*a])
                axs[j,i].set_title(str(i+j*16+64*a), fontsize=8)
                axs[j,i].set_ylim(-max_value_ok*1.1, max_value_ok*1.1)
                axs[j,i].axis('off')

        # --- PLOT ---
        #   of all individual signals in subplots, two periods
        a = 1               # second array
        fig, axs = plt.subplots(4,16)
        fig.suptitle("Individual signals", fontsize=16)
        start_val = 4000
        for j in range(4):
            for i in range(16):
                axs[j,i].plot(data[start_val:start_val+plot_samples,i+j*16+64*a])
                axs[j,i].set_title(str(i+j*16+64*a), fontsize=8)
                axs[j,i].set_ylim(-max_value_ok*1.1, max_value_ok*1.1)
                axs[j,i].axis('off')

        # --- PLOT ---
        #   of all individual signals in subplots, two periods
        a = 2               # third array
        fig, axs = plt.subplots(4,16)
        fig.suptitle("Individual signals", fontsize=16)
        start_val = 400
        for j in range(4):
            for i in range(16):
                axs[j,i].plot(data[start_val:start_val+plot_samples,i+j*16+64*a])
                axs[j,i].set_title(str(i+j*16+64*a), fontsize=8)
                axs[j,i].set_ylim(-max_value_ok*1.1, max_value_ok*1.1)
                axs[j,i].axis('off')


    if plot_selected:
        # --- PLOT ---
        #   of bad microphones
        plt.figure()
        for mic in range(len(bad_mics)):
            plt.plot(data[:,bad_mics[mic]], label=f"{bad_mics[mic]}")
        plt.xlim([0, plot_samples])
        #plt.ylim([-max_value_ok*1.1, max_value_ok*1.1])
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.title('Bad microphones')

    
        # --- PLOT ---
        #   of selected microphones, given by plot_mics
        plot_mics = [8, 18, 60, 64]             # choose what microphones to plot
        arr_plot_mics = np.array(plot_mics)     # convert plot_mics to numpy array with correct index
        plt.figure()
        for i in range(len(arr_plot_mics)):
            plt.plot(data[:,int(arr_plot_mics[i])], label=f"{arr_plot_mics[i]}")
        plt.xlim([0, plot_samples])
        plt.ylim([-max_value_ok*1.1, max_value_ok*1.1])
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.suptitle('Selected microphones microphones')
        plt.legend()

        ## --- PLOT ---
        ##   of all microphones with non-zero amplitude
        #plt.figure()
        #for mic in non_zero:
        #    plt.plot(data[:,mic], label=f"{mic}")
        #plt.xlim([0, plot_samples])
        ##plt.ylim([-max_value_ok*1.1, max_value_ok*1.1])
        #plt.xlabel('Samples')
        #plt.ylabel('Amplitude')
        #plt.legend()
        #plt.title('Non-zero microphones, set to 0')


    
    # --- PLOT ---
    #   of FFT of one signal
    if plot_energy:
        mic = 1         # mic signals of FFT
        samples = len(data[:,0])
        t_stop = samples/fs
        t = np.linspace(0,t_stop,samples)
        data_FFT = np.fft.fft(data[:,mic])
        energy = abs(data_FFT)**2
        freq = np.fft.fftfreq(t.shape[-1])
        plt.figure()
        plt.plot(fs*freq,energy)
        plt.title('Energy of signal')
        plt.xlabel('Frequency [Hz]')
        plt.legend(str(mic))

        # --- PLOT ---
        #   of FFT of several signals
        mics_FFT = [1,18, 64]
        arr_mics_FFT = np.array(mics_FFT,dtype=int)
        FFT_mic_legend = []                         # empty list that should hold legends for plot
        plt.figure()
        for i in range(len(arr_mics_FFT)):
            data_FFT = np.fft.fft(data[:,int(arr_mics_FFT[i])])
            energy = abs(data_FFT)**2
            freq = np.fft.fftfreq(t.shape[-1])
            plt.plot(fs*freq,energy)
            FFT_mic_legend = np.append(FFT_mic_legend,str(arr_mics_FFT[i]))
        plt.suptitle('Energy of selected microphones signals')
        plt.xlabel('Frequency [Hz]')
        plt.legend(FFT_mic_legend)

    # Show all plots
    if show_plots:
        plt.show()

main()
