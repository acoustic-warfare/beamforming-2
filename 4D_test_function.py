import numpy as np
#import sounddevice
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
import time


# 4D matris
def frequency_phase_shift(signal, fs, t, theta, phi, r_prime, c):
    #Nfft = 5000     # skillnad på n = defualt och N = Nfft i formen på kurvan
    #M = r_prime.shape[1]              # M = number of array elements
    FFT = np.fft.rfft(signal)         # kanske får ändra axis beroende på hur signal ser ut
    freq = np.fft.rfftfreq(len(signal))    # sample frequencies
    freq = np.reshape(freq, (len(freq),1))
    f = freq*fs     # real frequency
    #print("freq shape:", np.shape(freq))
    #FFT = np.repeat(FFT[:,np.newaxis], M, axis=1)        # 2D matrix with different frequency on each row and array element on each column
    FFT = np.reshape(FFT, (len(FFT),1,1,1))
    #print('shape FFT', np.shape(FFT))
    phase_shift = calc_phase_shift(f, theta, phi, r_prime, c)
    FFT_shifted = FFT*np.exp(-1j*phase_shift)       # apply phase shift to to every signal
    #FFT_shifted = FFT*np.exp(-1j*np.deg2rad(90))   # apply phase shift to to every signal - för test

    return FFT_shifted, f


def calc_phase_shift(f, theta, phi, r_prime, c):
    x_i = r_prime[0,:]   # array containing x-position of each array element 
    y_i = r_prime[1,:]   # array containing y-position of each array element 
    
    f = np.reshape(f, (len(f),1,1,1))
    x_i = np.reshape(x_i, (1,len(x_i),1,1))
    y_i = np.reshape(y_i, (1,len(y_i),1,1))
    sin_theta = np.reshape(np.sin(theta), (1,1,len(theta),1))
    cos_phi = np.reshape(np.cos(phi), (1,1,1,len(phi)))
    sin_phi = np.reshape(np.sin(phi), (1,1,1,len(phi)))
    k = 2*np.pi*f/c      # wave number
    
    phase_shift = k*(x_i*sin_theta*cos_phi + y_i*sin_theta*sin_phi) # each row corresponds to a frequency, each column corresponds to an array element
    #print('shape phase shift:', np.shape(phase_shift))
    return phase_shift


def main():
    start = time.time()
    c = 343
    r_prime = np.array([[3, 1, 2], [6, 4, 2], [0, 0, 0]])
    theta = np.linspace(0,np.pi,56)     # kanske konstigt när man gör om till grader
    phi = np.linspace(0,2*np.pi,56)
    N = 512 # antalet sampel som vi kommer ha?
    fs = 48828 # riktiga sampelfrekvensen
    T = N
    t = np.linspace(0,N/fs,T)
    
    f0 = 11000
    f1 = 3000
    sinus = np.sin(2*np.pi*f0*t) + np.sin(2*np.pi*f1*t)
    FFT_shifted, f = frequency_phase_shift(sinus,fs,t,theta,phi,r_prime,c)

    end = time.time()
    print('Simulation time:', round((end - start), 4), 's')

    print('f0 =', f0)
    print('f1 =', f1)

    # Frequency domain plots - 4D
    plt.figure(1)
    plt.plot(f, np.abs(FFT_shifted[:,0,25,25]),'b')         # plots för tre olika array element, alla ska ge samma resultat
    #plt.plot(f, np.abs(FFT_shifted[:,1,25,25]),'r--')
    #plt.plot(f, np.abs(FFT_shifted[:,2,25,25]),'g--')
    plt.xlabel('f [Hz]')

    # test för att kolla så förkjutningen stämmer - time
    #sinus_shifted = np.fft.irfft(FFT_shifted,axis=0)
    #plt.plot(np.rad2deg(t), sinus,'b')
    #plt.plot(np.rad2deg(t), sinus_shifted[:,0,25,25],'r--') # sinus shifted: 1st indx = time/sample, 2nd indx = array element, 3rd indx = theta, 4th indx = phi
    #plt.plot(np.rad2deg(t), sinus_shifted[:,1,25,25],'r--')

    plt.show()
    

main()

