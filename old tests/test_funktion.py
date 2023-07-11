import numpy as np
#import sounddevice
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)


def frequency_phase_shift(signal, fs, t, theta, phi, r_prime, c):
    M = r_prime.shape[1]              # M = number of array elements
    FFT = np.fft.rfft(signal,axis=-1) # kanske får ändra axis beroende på hur signal ser ut
    freq = np.fft.rfftfreq(len(t))    # sample frequencies
    freq = np.reshape(freq, (len(freq),1))
    f = freq*fs     # real frequency?
    print("freq shape:", np.shape(freq))
    FFT = np.repeat(FFT[:,np.newaxis], M, axis=1)        # 2D matrix with different frequency on each row and array element on each column
    print('shape FFT', np.shape(FFT))
    phase_shift = calc_phase_shift(f, theta, phi, r_prime, c)

    FFT_shifted = FFT*np.exp(-1j*phase_shift)       # apply phase shift to to every signal
    #FFT_shifted = FFT*np.exp(-1j*np.deg2rad(90))   # apply phase shift to to every signal - för test
    
    
    # för eventuella 3D/4D matriser
    #FFT= np.repeat(FFT[:,:,np.newaxis], 4, axis=2)
    #FFT = np.swapaxes(FFT,0,2)
    #FFT = np.swapaxes(FFT,1,2)
    # kanske scuffed

    return FFT_shifted, f, FFT


def calc_phase_shift(f, theta, phi, r_prime, c):
    x_i = r_prime[0,:]   # array containing x-position of each array element 
    y_i = r_prime[1,:]   # array containing y-position of each array element 
    k = 2*np.pi*f/c      # wave number
    
    ### funkar för ett värde av theta och phi
    #phase_shift = k*(x_i*np.sin(theta)*np.cos(phi) + y_i*np.sin(theta)*np.sin(phi)) # each row corresponds to a frequency, each column corresponds to an array element
    ###

    ### ska funka för två arrayer som innehåller theta och phi
    phase_shift = k*(x_i*np.sin(theta)*np.cos(phi) + y_i*np.sin(theta)*np.sin(phi)) # each row corresponds to a frequency, each column corresponds to an array element
    ###

    print('shape phase shift:', np.shape(phase_shift))
    return phase_shift


def main():
    c = 343
    r_prime = np.array([[3, 1, 2], [6, 4, 2], [0, 0, 0]])
    #M = r_prime.shape[1] # M = number of array elements
    theta = np.pi/3
    phi = np.pi/3
    fs = 48800  # sampling frequency
    #T = 5000 # number of points in time vector t
    #T = fs   # ger rätt frekvens-peak i FFT:n
    T = 512 # antalet sampel som vi kommer ha?
    N = 512 # antalet sampel som vi kommer ha?
    t = np.linspace(0,2*np.pi,T)
    #t = np.linspace(0,N/fs,T)
    f0 = 1700
    f1 = 1750
    sinus = np.sin(2*np.pi*f0*t) + np.sin(2*np.pi*f1*t)
    FFT_shifted, f, FFT = frequency_phase_shift(sinus,fs,t,theta,phi,r_prime,c)
    
    #3D matriser
    #plt.plot(np.transpose(np.rad2deg(t))[0,:], sinus[2,:],'b')
    ##plt.plot(np.transpose(np.rad2deg(t))[0,:], np.transpose(sinus_shifted[1,:,1]),'r--')
    #plt.plot(np.transpose(np.rad2deg(t))[0,:], sinus_shifted[2,:,1],'r--')   # andra index = array element, tredje index = frekvens, utan swapaxis

    #plt.plot(np.rad2deg(t), np.transpose(sinus)[:,2],'b')
    ##plt.plot(np.rad2deg(t), np.transpose(sinus_shifted[1,:,1]),'r--')
    #plt.plot(np.rad2deg(t), np.transpose(sinus_shifted)[:,0,2],'r--')   # andra index = array element, tredje index = frekvens, utan swapaxis
    ##plt.plot(np.rad2deg(t), np.transpose(sinus_shifted)[0,:,2],'r--')   # andra index = array element, tredje index = frekvens, utan swapaxis

    # 2D matriser, tid - för att kolla så förskjutningen stämmer
    #sinus_shifted = np.fft.irfft(FFT_shifted,axis=0) 
    #plt.plot(np.rad2deg(t), np.transpose(sinus),'b')
    #plt.plot(np.rad2deg(t), np.transpose(sinus_shifted)[0,:],'r--')
    #plt.plot(np.rad2deg(t), np.transpose(sinus_shifted)[1,:],'g--')
    #plt.plot(np.rad2deg(t), np.transpose(sinus_shifted)[2,:],'y--')

    # 2D matriser, frekvens
    plt.figure(1)
    plt.plot(f/(2*np.pi), np.abs(FFT_shifted)[:,0],'b')
    plt.plot(f/(2*np.pi), np.abs(FFT_shifted)[:,1],'r--')
    plt.plot(f/(2*np.pi), np.abs(FFT_shifted)[:,2],'g--')
    plt.xlabel('f [Hz]')

    #plt.figure(2)
    #plt.plot(f, np.rad2deg(np.angle(FFT_shifted)[:,0]),'b')
    #plt.plot(f, np.rad2deg(np.angle(FFT_shifted)[:,1]),'r--')
    #plt.plot(f, np.rad2deg(np.angle(FFT_shifted)[:,2]),'g--')
    #plt.xlabel('f [Hz]')
    #plt.ylabel('angle [deg]')

    plt.figure(3)
    plt.plot(f/(2*np.pi), np.abs(FFT)[:,0],'b')
    plt.plot(f/(2*np.pi), np.abs(FFT)[:,1],'r--')
    plt.plot(f/(2*np.pi), np.abs(FFT)[:,2],'g--')
    plt.xlabel('f [Hz]')

    plt.show()


main()

