import numpy as np
#import sounddevice
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)


def frequency_phase_shift(signal, fs, t, theta, phi, r_prime, c):
    M = r_prime.shape[1] # M = number of array elements
    FFT = np.fft.rfft(signal,axis=-1) # kanske får ändra axis beroende på hur signal ser ut
    freq = np.fft.rfftfreq(len(t))
    freq = np.reshape(freq, (len(freq),1))
    #print(np.shape(FFT))
    #print(freq)
    FFT= np.repeat(FFT[:,np.newaxis], M, axis=1)        # matris med frekvenser på raderna och array element på kolumnerna
    #FFT= np.repeat(FFT[:,:,np.newaxis], 4, axis=2)
    #FFT = np.swapaxes(FFT,0,2)
    #FFT = np.swapaxes(FFT,1,2)
     # kanske scuffed


    #print(FFT)
    print('shape FFT', np.shape(FFT))
    phase_shift = calc_phase_shift(fs, freq, theta, phi, r_prime, c)

    #FFT_shifted = FFT*np.exp(-1j*np.deg2rad(90))   # apply phase shift to each signal
    #print(np.shape(FFT_shifted))
    FFT_shifted = FFT*np.exp(-1j*phase_shift)   # apply phase shift to each signal
    #FFT_shifted = np.swapaxes(FFT,1,2)    # swap back axes 
    return FFT_shifted


def calc_phase_shift(fs, freq, theta, phi, r_prime, c):
    x_i = r_prime[0,:]   # array containing x-position of each array element 
    y_i = r_prime[1,:]   # array containing y-position of each array element 
    f = freq*fs
    k = 2*np.pi*f/c

    phase_shift = k*(x_i*np.sin(theta)*np.cos(phi) + y_i*np.sin(theta)*np.sin(phi)) # each row corresponds to a frequency, each column corresponds to an array element
    #print(phase_shift)
    print('shape phase shift:', np.shape(phase_shift))
    # kanske scuffed
    return phase_shift


def main():
    c = 343
    r_prime = np.array([[3, 1, 2], [6, 4, 2], [0, 0, 0]])
    M = r_prime.shape[1] # M = number of array elements
    theta = np.pi/3
    phi = np.pi/3
    #F = 4    # number of frequencies
    T = 512 # number of points in time vector t
    #f = np.linspace(1,5,F)*1/4
    #f = f.reshape(F,1) # transpose f
    t = np.linspace(0,2*np.pi,T)
    fs = 49000
    #sinus = np.sin(2*np.pi*f*t)
    sinus = np.sin(2*np.pi*t)
    #print('sinus:', sinus)
    #signal = np.swapaxes(signal,0,2)
    #print(signal)
    FFT_shifted = frequency_phase_shift(sinus,fs,t,theta,phi,r_prime,c)
    sinus_shifted = np.fft.irfft(FFT_shifted,axis=0)    # funkar med axis = 2 om man inte swappar tillbaka
    print(sinus_shifted[:,0])
    print(np.shape(sinus_shifted))
    #sinus_shifted = np.fft.irfft(FFT_shifted,axis=1)    
    A = np.array([[1, 7, 3], [1, 1, 1], [2, 2, 2], [9, 9, 9]])
    R = np.array([[1, 1, 7], [1, 1, 1], [0, 0, 0], [5, 6, 1]])
    G = np.array([[2, 3, 1], [2, 2, 2], [3, 0, 2], [1, 0, 3]])

    B = np.zeros((A.shape[0], A.shape[1], 3))
    #B = np.array([A, A, R])        # "scuffed" method för multiplikation och summering           
    ##print(B)
    #I = np.array([A, G, G])
    ##print(I)
    #J = I*B
    #print(J)
    #A = np.expand_dims(A, axis=0)
    #print(np.sum(J, axis = 0))      # scuffed method slut
    #print(A)
    #C = A*B
    #print(C)
    #print(I)
    #print(C[2])                     # stopp
    #sin_shifted = frequency_phase_shift(sinus,fs,t,theta,phi,r_prime,c)
    #print(np.shape(sinus))
    #print(np.shape(FFT_shifted))
    #print('sinus shifted:', np.shape(sinus_shifted))
    #t = t[:,np.newaxis] # make into a 2D vector so it can be transposed
    plt.figure()
    #3D matriser
    #plt.plot(np.transpose(np.rad2deg(t))[0,:], sinus[2,:],'b')
    ##plt.plot(np.transpose(np.rad2deg(t))[0,:], np.transpose(sinus_shifted[1,:,1]),'r--')
    #plt.plot(np.transpose(np.rad2deg(t))[0,:], sinus_shifted[2,:,1],'r--')   # andra index = array element, tredje index = frekvens, utan swapaxis

    #plt.plot(np.rad2deg(t), np.transpose(sinus)[:,2],'b')
    ##plt.plot(np.rad2deg(t), np.transpose(sinus_shifted[1,:,1]),'r--')
    #plt.plot(np.rad2deg(t), np.transpose(sinus_shifted)[:,0,2],'r--')   # andra index = array element, tredje index = frekvens, utan swapaxis
    ##plt.plot(np.rad2deg(t), np.transpose(sinus_shifted)[0,:,2],'r--')   # andra index = array element, tredje index = frekvens, utan swapaxis

    # 2D matriser
    plt.plot(np.rad2deg(t), np.transpose(sinus),'b')
    plt.plot(np.rad2deg(t), np.transpose(sinus_shifted)[0,:],'r--')
    plt.plot(np.rad2deg(t), np.transpose(sinus_shifted)[1,:],'g--')
    plt.plot(np.rad2deg(t), np.transpose(sinus_shifted)[2,:],'y--')
    #plt.plot(np.rad2deg(t), np.transpose(sinus_shifted)[:,0,2],'r--')   # andra index = array element, tredje index = frekvens, utan swapaxis
    #plt.plot(np.rad2deg(t), np.transpose(sinus_shifted)[0,:,2],'r--')   # andra index = array element, tredje index = frekvens, utan swapaxis


    plt.show()
    #for freq_ind in range(F):
    #    plt.figure()
    #    plt.plot(np.rad2deg(t), np.transpose(sinus)[:,freq_ind],'b')
    #    plt.plot(np.rad2deg(t), np.transpose(sinus_shifted)[:,freq_ind],'r--')
    #    plt.grid()
    #plt.show()


main()

