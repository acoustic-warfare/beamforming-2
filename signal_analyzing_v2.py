import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
import config
import sounddevice
import soundfile as sf


def load_data_FPGA(filename):
    #   FUNCTION TO LOAD DATA FROM .TXT FILE INTO NUMPY ARRAY 
    #   (RECORDED BY FPGA)

    path = Path('/home/batman/github/micarray-gpu/FPGA_data/1108/' + filename + '.txt')

    # Load recorded data from file
    data = np.loadtxt(open(path,'rb').readlines()[:-1],delimiter=',')
    f_sampling = config.f_sampling  # get sampling frequency
    data = data[:,4:]       # take out data from microphones only

    return data, int(f_sampling)

def delete_mic_data(signal, mic_to_delete):
    #   FUNCTION THAT SETS SIGNALS FROM 'BAD' MICROPHONES TO 0
    signal[:,mic_to_delete] = 0
    #for mic in range(len(mic_to_delete)):
    #    for samp in range(len(signal[:,0])):
    #        signal[samp,mic_to_delete[mic]] = 0
    return signal

def write_to_txt_file(filename, signals):
    #   FUNCTION THAT WRITES VALUES TO .TXT FILE
    np.savetxt(filename, signals, delimiter=',\t ', newline='\n', header='', footer='', comments='# ', encoding=None)

def write_to_npy_file(filename, signals):
    array_signals = np.zeros((1), dtype=object)
    array_signals[0] = signals
    np.save(filename+'.npy', array_signals)

def play_sound(sound_signal, f_sampling):
    scaled = sound_signal/np.max(np.abs(sound_signal))
    sounddevice.play(scaled, f_sampling, blocking=True)

    #sf.write("test.wav", sound_signal, int(f_sampling), 'PCM_24')


def main():
    recording_device = 'FPGA' # choose between 'FPGA' and 'BB' (BeagelBone) 
    #filename = '0908_440Hz_0deg'
    filename = config.filename
    print(filename)
    #initial_samples = 30000                 # initial samples, at startup phase of Beaglebone recording

    # Choose the mic signals that should be set to zero
    mics_to_delete = [18, 64]
    arr_mics_to_delete = np.array(mics_to_delete, dtype = int) # converts mic_to_delete to numpy array with correct index

    # Plot options
    show_plots = 1      # if show_plots = 1, then plots will be shown
    plot_period = 1     # periods to plot
    f0 = 800            # frequency of recorded sinus signal

    # Load data from .txt file
    data, fs = load_data_FPGA(filename)
        #write_to_npy_file(filename,data)
    total_samples = len(data[:,0])          # Total number of samples
    #initial_data = data[0:initial_samples,] # takes out initial samples of signals 
    #
    #
    #if recording_device == 'FPGA':
    #    ok_data = data # all data is ok
    #elif recording_device == 'BB':
    #    ok_data = data[initial_samples:,] # initial startup values are ignored


    plot_samples = math.floor(plot_period*(fs/f0))                     # number of samples to plot, to use for axis scaling
    max_value_ok = np.max(np.max(data[0:4000,],axis=0)) # maximum value of data, to use for axis scaling in plots

    #play_sound(data[:,21],fs)

    print('f_sampling: '+ str(int(fs)))

    # --- PLOT ---
    #   of bad microphones
    plt.figure()
    for mic in range(len(arr_mics_to_delete)):
        plt.plot(data[:,arr_mics_to_delete[mic]])
    plt.xlim([0, plot_samples])
    plt.ylim([-max_value_ok*1.1, max_value_ok*1.1])
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title('Bad microphones')


    # --- PLOT ---
    #   of all individual signals in subplots, two periods
    fig, axs = plt.subplots(4,16)
    fig.suptitle("Individual signals", fontsize=16)
    start_val = 4000
    for j in range(4):
        for i in range(16):
            axs[j,i].plot(data[start_val:start_val+plot_samples,i+j*16])
            axs[j,i].set_title(str(i+j*16+1), fontsize=8)
            axs[j,i].set_ylim(-max_value_ok*1.1, max_value_ok*1.1)
            axs[j,i].axis('off')


    # Set microphone signals of bad mics to zero
    #clean_data = delete_mic_data(ok_data, arr_mics_to_delete)
    #clean_initial_data = delete_mic_data(initial_data, arr_mics_to_delete)

    # --- PLOT ---
    #   plot of all microphones, after bad signals have been set to 0
    plt.figure()
    plt.plot(data)
    plt.xlim([0, plot_samples])
    plt.ylim([-max_value_ok*1.1, max_value_ok*1.1])
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.suptitle('All microphones')

    
    # --- PLOT ---
    #   of selected microphones
    plot_mics = [8, 18, 60, 64]                     # what microphones to plot
    arr_plot_mics = np.array(plot_mics)-1   # convert plot_mics to numpy array with correct index
    mic_legend = []                         # empty list that should hold legends for plot
    plt.figure()
    for i in range(len(arr_plot_mics)):
        plt.plot(data[:,int(arr_plot_mics[i])], '-*')
        mic_legend = np.append(mic_legend,str(arr_plot_mics[i]+1))
    plt.xlim([0, plot_samples])
    plt.ylim([-max_value_ok*1.1, max_value_ok*1.1])
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.suptitle('Selected microphones microphones')
    plt.legend(mic_legend)

    # --- PLOT ---
    #plt.figure()
    #plt.plot(initial_data[:,3])
    #plt.xlim([0, initial_samples])
    #plt.xlabel('Samples')
    #plt.ylabel('Amplitude')
    #plt.suptitle('Initial values')

    # --- PLOT ---
    #   of FFT of one signal
    mic = 1         # mic signals of FFT
    samples = len(data[:,0])
    t_stop = samples/fs
    t = np.linspace(0,t_stop,samples)
    data_FFT = np.fft.fft(data[:,mic-1])
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
    arr_mics_FFT = np.array(mics_FFT,dtype=int)-1
    FFT_mic_legend = []                         # empty list that should hold legends for plot
    plt.figure()
    for i in range(len(arr_mics_FFT)):
        data_FFT = np.fft.fft(data[:,int(arr_mics_FFT[i])])
        energy = abs(data_FFT)**2
        freq = np.fft.fftfreq(t.shape[-1])
        plt.plot(fs*freq,energy)
        FFT_mic_legend = np.append(FFT_mic_legend,str(arr_mics_FFT[i]+1))
    plt.suptitle('Energy of selected microphones signals')
    plt.xlabel('Frequency [Hz]')
    plt.legend(FFT_mic_legend)

    # Show all plots
    if show_plots:
        plt.show()

main()