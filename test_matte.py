import numpy as np
import math
#import sounddevice
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal

font = {'size'   :  18}
plt.rc('font', **font)

def phase_shift(iptsignal, angle, dt):
    """Perform phase shift of arbitary angle

    Parameter
    =========
    iptsignal : numpy.array
        input signal
    angle : float
        angle to shift signal, in degree
    dt : float
        time step
    
    """
    # Resolve the signal's fourier spectrum
    spec = np.fft.rfft(iptsignal)
    freq = np.fft.rfftfreq(iptsignal.size, d=dt)

    # Perform phase shift in freqeuency domain
    spec *= np.exp(1.0j * np.deg2rad(angle))

    # Inverse FFT back to time domain
    phaseshift = np.fft.irfft(spec, n=len(iptsignal))
    return phaseshift


r_prime = np.zeros((3,10))

"""f = np.arange(5)[1:5][np.newaxis].T*1/4
t = np.arange(6)[1:5]
#t = np.arange(5)[1:5]
print(f*t)
sinus = np.sin(2*np.pi*f*t)
print('sin:', sinus)

FFT = np.fft.rfft(sinus,axis=1)
freq = np.fft.rfftfreq(len(t))
#print('freqs:', freq)

shift = -90
FFT *= np.exp(1j*np.deg2rad(shift))
#print('FFT shifted:', FFT)
#iFFT = np.fft.irfft(FFT, n=len(sinus))
iFFT = np.fft.irfft(FFT,axis=1)
print('Inverse shifted FFT:', iFFT)"""


if __name__ == '__main__':
    # Define Time range
    time = np.arange(0, 10000)
    signal = np.cos(time /  100.0)
    sinsignal = np.sin(time /  100.0)
    
    # Shift angle and time step
    angle, dt = -90, 1
    phsignal = phase_shift(signal, angle, dt)
    print(phsignal)

    # Comparasion between Personal shifted and theoretical result
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 14))
    axes[0].plot(time, signal, label="Cosine (raw)")
    axes[0].plot(time, sinsignal, label="Sine")
    axes[0].plot(time, phsignal, "--", label="Shift")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Comparison between Sinudoidal function and shifted signal")
    axes[0].legend()
    plt.show()
    # Comparasion between Personal shifted and Hilbert transform result
    axes[1].plot(time, signal, label="Cosine (raw)")
    axes[1].plot(time, phsignal, label="Shift")
    axes[1].plot(time, np.imag(hilbert(signal)), "--", label="Hilbert")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("Comparison between shifted signal and Hilbert transform format")
    axes[1].legend()
    plt.show()