import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
import time
import calc_phase_shift

c = 343
#r_prime = np.array([[3, 1, 2], [6, 4, 2], [0, 0, 0]])
#r_prime = calc_phase_shift.r_prime
theta = np.linspace(0,np.pi,56)
phi = np.linspace(0,2*np.pi,56)
N = 512    # antalet sampel
fs = 48828 # sampelfrekvens
T = N
t = np.linspace(0,N/fs,T)
t = np.reshape(t, (len(t),1,1,1))
f = calc_phase_shift.f
f0 = 11000
f1 = 3000
f2 = 20000
k0 = 2*np.pi*f0/c
k1 = 2*np.pi*f1/c
k2 = 2*np.pi*f2/c
r = 2000
x_i = calc_phase_shift.x_i
y_i = calc_phase_shift.y_i
phi = calc_phase_shift.phi[37]      # direction of source
theta = calc_phase_shift.theta[37]  # direction of source

shift_factor = (x_i*np.sin(theta)*np.cos(phi) + y_i*np.sin(theta)*np.sin(phi))
print(np.shape(shift_factor))
signal = np.sin(2*np.pi*f0*t - k0*r + k0*shift_factor) + np.sin(2*np.pi*f1*t - k1*r + k1*shift_factor)



phase_shift = calc_phase_shift.phase_shift
#phase_shift_summed = calc_phase_shift.phase_shift_summed
print('signal shape:', np.shape(signal))

#def frequency_phase_shift(signal, phase_shift, phase_shift_summed):
#    FFT = np.fft.rfft(signal)       # Fourier transform each signal
#    FFT2 = np.reshape(FFT, (len(FFT),1,1)) # reshape into a 3D array
#    FFT = np.reshape(FFT, (len(FFT),1,1,1)) # reshape into a 4D array
#    FFT_shifted = FFT*phase_shift     # apply phase shift to to every signal
#    FFT_shifted_summed = FFT2*phase_shift_summed
#    #print('shape FFT', np.shape(FFT))
#    #FFT_shifted = FFT*np.exp(-1j*np.deg2rad(90))   # för test
#    return FFT_shifted, FFT_shifted_summed
#    #return FFT_shifted_summed

def frequency_phase_shift(signal, phase_shift):
    FFT = np.fft.rfft(signal,axis=0) # Fourier transform each signal - 3D
    #FFT = np.fft.rfft2(signal,axes=(1,0)) # Fourier transform each signal - 3D
    #FFT = np.reshape(FFT, (len(FFT[:,0]),len(FFT[0,:]),1,1)) # reshape into a 4D array from 2D
    print('shape FFT', np.shape(FFT))
    FFT_shifted = FFT*phase_shift     # apply phase shift to every signal
    #FFT_shifted = FFT*np.exp(-1j*np.deg2rad(90))   # för test
    return FFT_shifted, FFT


def main():
    start = time.time()
    #FFT_shifted, FFT_summed2 = frequency_phase_shift(signal,phase_shift,calc_phase_shift.phase_shift_summed)
    FFT_shifted, FFT = frequency_phase_shift(signal,phase_shift)
    #FFT_intensity = np.sum(np.abs(FFT)**2,axis=1)
    FFT_intensity2 = np.sum(np.abs(FFT_shifted)**2,axis=1)
    end = time.time()
    print('Simulation time:', round((end - start), 4), 's')
    print('f0 =', f0)
    print('f1 =', f1)
    #print(FFT_summed)
    #print(np.shape(FFT_intensity))
    print(np.shape(FFT_intensity2))
    #print(FFT_summed - FFT_summed2)


    # Frequency domain plots - 4D
    plt.figure(1)
    plt.plot(f[:,0,0,0], np.abs(FFT_shifted[:,1,37,37]),'b')         # plots för tre olika array element, alla ska ge samma resultat
    plt.plot(f[:,0,0,0], np.abs(FFT_shifted[:,1,25,25]),'r--')
    plt.plot(f[:,0,0,0], np.abs(FFT_shifted[:,1,10,5]),'g--')
    plt.xlabel('f [Hz]')

    #plt.figure(2)
    #plt.plot(f[:,0,0,0], FFT_intensity[:,0,0],'b')
    #plt.xlabel('f [Hz]')
 
    #plt.figure(3)
    #plt.plot(f[:,0,0,0], FFT_intensity2[:,0,0],'b')
    #plt.xlabel('f [Hz]')
    
    plt.figure(4)
    plt.plot(f[:,0,0,0], np.rad2deg(np.angle(FFT[:,0,0,0])),'b')
    plt.plot(f[:,0,0,0], np.rad2deg(np.angle(FFT[:,1,0,0])),'r--')
    plt.plot(f[:,0,0,0], np.rad2deg(np.angle(FFT[:,2,0,0])),'g--')
    plt.xlabel('f [Hz]')
    plt.ylabel('FFT')
#
    plt.figure(5)
    plt.plot(f[:,0,0,0], np.rad2deg(np.angle(FFT_shifted[:,0,37,37])),'b')
    plt.plot(f[:,0,0,0], np.rad2deg(np.angle(FFT_shifted[:,1,37,37])),'r--')
    plt.plot(f[:,0,0,0], np.rad2deg(np.angle(FFT_shifted[:,2,37,37])),'g--')
    plt.xlabel('f [Hz]')
    plt.ylabel('Shifted FFT')

    ## test för att kolla så förkjutningen stämmer i tid
    signal_shifted = np.fft.irfft(FFT_shifted,axis=0)
    plt.figure(6)
    #plt.plot(np.rad2deg(t[:,0,0,0]), signal[:,0,0,0],'b')
    plt.plot(np.rad2deg(t[:,0,0,0]), signal_shifted[:,0,37,37],'r--') # sinus shifted: 1st indx = time/sample, 2nd indx = array element, 3rd indx = theta, 4th indx = phi
    plt.plot(np.rad2deg(t[:,0,0,0]), signal_shifted[:,1,37,37],'b--')
    plt.plot(np.rad2deg(t[:,0,0,0]), signal_shifted[:,2,37,37],'g--')
    
    #plt.figure(7)
    #plt.plot(np.rad2deg(calc_phase_shift.theta[:]), np.abs(FFT_shifted[5,0,:,10]),'b')
    #plt.xlabel('theta [deg]')
#
    #plt.figure(8)
    #plt.plot(np.rad2deg(calc_phase_shift.phi[:]), np.abs(FFT_shifted[5,0,10,:]),'b')
    #plt.xlabel('phi [deg]')

    plt.show()

main()
