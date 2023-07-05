import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.colors import BoundaryNorm
#from matplotlib.ticker import MaxNLocator
np.set_printoptions(threshold=np.inf)
import time
import config
import calc_phase_shift_cartesian
import generate_signals
import calc_phase_shift_cartesian


def generate_filename():
    if config.sources == 1:
        filename ='emul_'+ 'samples='+str(config.samples) + '_'+ str(config.f_start1)+'Hz_'+'theta='+str(config.theta_deg1)+'_phi='+str(config.phi_deg1)+ \
            '_A'+str(config.active_arrays)
    elif config.sources == 2:
        filename ='emul_'+'samples='+str(config.samples) + '_'+str(config.f_start1)+str(config.f_start2)+'Hz_'+'theta='+str(config.theta_deg1)+str(config.theta_deg2)+ \
        '_phi='+str(config.phi_deg1)+str(config.phi_deg2)+'_A'+str(config.active_arrays)
    return filename


filename = 'emulated_data/' + generate_filename()
try:
    signal = np.float32(np.load(filename+'.npy',allow_pickle=True))
    print('Loading from memory: ' + filename)
except:
    generate_signals.main()
    signal = generate_signals.emulated_signals

#signal = np.reshape(signal, (len(signal[:,0]), len(signal[0,:]),1,1))
#print('signal shape:', np.shape(signal))

n_elements = config.elements # number of elements
d = config.distance
r_prime = generate_signals.calc_r_prime(d)
theta = calc_phase_shift_cartesian.theta
phi = calc_phase_shift_cartesian.phi
N = config.samples     # number of samples
fs = config.f_sampling # samplefrequency
T = N
t = np.linspace(0,N/fs,T)
t = np.reshape(t, (len(t),1,1,1))
f = calc_phase_shift_cartesian.f
print('Scanning window:\n' + 'theta:', -int(np.rad2deg(theta)[0,0,0,round(config.y_res/2)]), 'to', int(np.rad2deg(theta)[0,0,config.x_res-1,round(config.y_res/2)]), 'deg')
#print('phi:', int(np.rad2deg(phi)[0,0,0,0]), 'to', int(np.rad2deg(phi)[0,0,0,len(phi[0,0,0,:])-1]), 'deg')

x_scan = calc_phase_shift_cartesian.x_scan
y_scan = calc_phase_shift_cartesian.y_scan

theta_source1 = config.theta_deg1  # direction of source 1
phi_source1 = config.phi_deg1      # direction of source 1
theta_source2 = config.theta_deg2  # direction of source 2
phi_source2 = config.phi_deg2      # direction of source 2
theta_source1_indx = calc_phase_shift_cartesian.theta_source1_indx
phi_source1_indx = calc_phase_shift_cartesian.phi_source1_indx
theta_source2_indx = calc_phase_shift_cartesian.theta_source2_indx
phi_source2_indx = calc_phase_shift_cartesian.phi_source2_indx

print('True source angles)')
print('Source 1:\n' + 'theta = ' + str(int(theta_source1)) + ' deg' + '\n' + 'phi = ' + str(int(phi_source1)) + ' deg')
if config.sources == 2:
    print('Source 2:\n' + 'theta = ' + str(int(theta_source2)) + ' deg' + '\n' + 'phi = ' + str(int(phi_source2)) + ' deg')
    theta_source2 = np.deg2rad(theta_source2)
    phi_source2 = np.deg2rad(phi_source2)

phase_shift = calc_phase_shift_cartesian.phase_shift

def frequency_phase_shift(signal, phase_shift):
    FFT = np.fft.rfft(signal,axis=0) # Fourier transform each signal - 3D
    FFT = np.reshape(FFT, (len(FFT[:,0]),len(FFT[0,:]),1,1)) # reshape into a 4D array from 2D
    #print('shape FFT', np.shape(FFT))
    FFT_shifted = FFT*phase_shift     # apply phase shift to every signal
    #print('shape FFT shifted', np.shape(FFT_shifted))
    return FFT_shifted, FFT

def find_signal_peak(array, value, signal, source_indxes):
    signal = signal[:,source_indxes[0],source_indxes[1]]
    idx = np.argmin((np.abs(array - value)))
    max_vals = np.array([signal[idx-1], signal[idx], signal[idx+1]])
    idx = np.where(signal == np.max(max_vals))[0][0]
    #print('idx-1:', int(signal[idx-1]), ', idx:', int(signal[idx]), ', idx+1:', int(signal[idx+1]), ', idx+2:', int(signal[idx+2]))
    return idx  # frekvensen av FFTns peakar motsvarar inte alltid idx som ges först, kan vara förskjuten ett steg till höger eller vänster



def main():
    start = time.time()
    FFT_shifted, FFT = frequency_phase_shift(signal,phase_shift)
    FFT_power = np.abs(np.sum(FFT_shifted,axis=1))**2
    FFT_power = FFT_power/n_elements    # normalise with number of array elements
    FFT_power_summed_freq = np.sum(FFT_power, axis=0)
    end = time.time()
    print('Simulation time:', round((end - start), 4), 's')

    ## plot heatmap
    #fig, ax = plt.subplots()
    #ax.pcolormesh(x_scan[0,0,:,0], y_scan[0,0,0,:], np.transpose(FFT_power_summed_freq)) # heatmap summed over all frequencies
    levels = np.linspace(np.min(FFT_power_summed_freq), np.max(FFT_power_summed_freq), 50)
    plt.contourf(x_scan[0,0,:,0], y_scan[0,0,0,:], np.transpose(FFT_power_summed_freq), levels, cmap=plt.get_cmap('coolwarm'))
    #plt.gca().set_aspect(9/16)       # X/Y gives aspect ratio of Y:X 
    plt.gca().set_aspect('auto')
    plt.xlabel('x')     
    plt.ylabel('y')
    plt.colorbar()

    max_indxes = np.unravel_index(np.argmax(FFT_power), np.shape(FFT_power))
    x_max_indx = max_indxes[1]
    y_max_indx = max_indxes[2]
    print(x_max_indx)
    print(y_max_indx)

    print('Strongest source found at:')
    print('theta =', np.rad2deg(theta[0,0,x_max_indx,y_max_indx]))
    print('phi =', np.rad2deg(np.arctan2(y_scan[0,0,0,y_max_indx],x_scan[0,0,x_max_indx,0])))

    ## 3D plot
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    #theta_surface = np.outer(np.rad2deg(theta), np.ones(len(theta)))
    #phi_surface = np.outer(np.ones(len(phi)), np.rad2deg(phi))
    #ax.plot_surface(theta_surface, phi_surface, np.transpose(FFT_power[f0_indx,:,:]), cmap='viridis', edgecolor='none')
    #ax.set_xlabel('theta [deg]')  
    #ax.set_ylabel('phi [deg]') 
    #ax.set_zlabel('FFT power')

    ## Power plots
    #max_indxes = np.unravel_index(np.argmax(FFT_power), np.shape(FFT_power))
    #x_max_indx = max_indxes[1]
    #y_max_indx = max_indxes[2]
    #plt.figure()
    #plt.plot(f[:,0,0,0], FFT_power[:,x_max_indx,y_max_indx],'b')
    #plt.plot(f[:,0,0,0], FFT_power[:,37,37],'r--')
    #plt.xlabel('f [Hz]')
    #plt.xlabel('FFT Power')

    ## test för att kolla så förkjutningen stämmer i tid
    #start = time.time()
    #signal_shifted = np.fft.irfft(FFT_shifted,axis=0)
    #end = time.time()
    #print('Simulation time IFFT:', round((end - start), 4), 's')
    #plt.figure()
    ##plt.plot(np.rad2deg(t[:,0,0,0]), signal[:,0,0,0],'b')
    #plt.plot(np.rad2deg(t[:,0,0,0]), signal_shifted[:,0,x_max_indx,y_max_indx],'r--') # sinus shifted: 1st indx = time/sample, 2nd indx = array element, 3rd indx = theta, 4th indx = phi
    #plt.plot(np.rad2deg(t[:,0,0,0]), signal_shifted[:,1,x_max_indx,y_max_indx],'b--')
    #plt.plot(np.rad2deg(t[:,0,0,0]), signal_shifted[:,2,x_max_indx,y_max_indx],'g--')



    plt.show()

main()
