# DRONE SIGNAL VISUALIZIGN
#
# Loads recroded drone signals from .m4a files into numpy arrays
#
# Plots spectogram, and energy spectrum of the recorded signals

from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class drone_signal:
    def __init__(self,filename, signal, fs, samples):
        self.signal = signal
        self.fs = fs
        self.samples = samples
        self.filename = filename
    
    def modify_signal_length(self, samp_start, samp_end):
        self.signal = self.signal[samp_start:samp_end]
        self.samples = samp_end-samp_start
        

def plot_energy(drones, norm = False):
    # --- PLOT ---
    #   of FFT of drone recordings
    #   could be several drone singals
    #
    #   drones : list where each element is a drone_signal object
    plt.figure()
    for i in range(len(drones)):
        dr = drones[i]                      # drone_signal object
        data = dr.signal                    # drone signal
        samp = dr.samples                   # number of samples
        f_s = dr.fs                         # sample rate of recording
        t_stop = samp/f_s                   # total recorded time
        t = np.linspace(0,t_stop,samp)      # time vector
        data_FFT = np.fft.fft(data)         # FFT of drone signal data
        freq = np.fft.fftfreq(t.shape[-1])  # normalized frequency vector
        if norm: 
            energy = (abs(data_FFT)/np.max(abs(data_FFT)))**2  # normalize the energy of signal
        else: 
            energy = abs(data_FFT)**2       # do not normalize the energy of signal
        plt.plot(f_s*freq, energy, alpha=0.7, label = dr.filename) 
    plt.suptitle('Spectrum drone singals', fontsize = FS_title)
    plt.xlabel('Frequency [Hz]', fontsize = FS_label)
    plt.xticks(fontsize= FS_ticks), plt.yticks(fontsize= FS_ticks)
    plt.legend(fontsize = FS_legend)
    plt.tight_layout()

def plot_spoctrogram(drone):
    # Plots spectogram of one drone recording
    #   drone: drone_signal object
    fs = drone.fs       # sampling rate of recording
    x = drone.signal    # signal of drone recording
    f, t, Sxx = signal.spectrogram(x, fs) # Obtain spectogram of the signal
    plt.figure()
    plt.pcolormesh(t, f, np.log10(2**Sxx)) #, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

def print_file_info(filename, fs, samples):
    # prints info about the drone recording file
    print(filename, ',\t fs:',fs, ',\t samples:',samples)

def import_m4a(filename, print = False):
    # imports the drone recording from the .m4a file

    # load infomration from file
    audio = AudioSegment.from_file('drone_recordings/'+filename, format='m4a')
    fs = audio.frame_rate
    audio = audio.get_array_of_samples()
    audio = np.array(audio)
    samples = audio.shape[0]
    drone = drone_signal(filename, audio, fs, samples)  # create drone_signal object
    if print:
        print_file_info(filename, fs, samples)
    return drone

# filenames
file1 = 'drone_2m.m4a'
file2 = 'drone_6m.m4a'
file3 = 'drone_16m.m4a'
file4 = 'drone_25m_side_flag.m4a'
file5 = 'drone_50m_flag.m4a'

# import data from files
audio1 = import_m4a(file1, print=True)
audio2 = import_m4a(file2, print=True)
audio3 = import_m4a(file3, print=True)
audio4 = import_m4a(file4, print=True)
audio5 = import_m4a(file5, print=True)

# modify the singla length of one audio file
audio5.modify_signal_length(44100, audio5.samples-44100)

# plot text font sizes
FS_ticks = 14
FS_label =14
FS_title = 16
FS_legend = 12

# plot energy
plot_energy([audio3, audio4], norm=True)
plot_energy([audio1, audio2], norm=True)
plot_energy([audio1, audio5], norm=True)

# plot spectrogram
plot_spoctrogram(audio1)
plot_spoctrogram(audio3)
plot_spoctrogram(audio5)

# show all plots
plt.show()