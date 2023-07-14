import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.colors import BoundaryNorm
#from matplotlib.ticker import MaxNLocator
np.set_printoptions(threshold=np.inf)
import time
import math
import config
import calc_phase_shift_cartesian
import generate_signals
import calc_r_prime
import active_microphones as am


def generate_filename():
    if config.sources == 1:
        filename ='emul_'+ 'samples='+str(config.samples) + '_'+ str(config.f_start1)+'Hz_'+'theta='+str(config.theta_deg1)+'_phi='+str(config.phi_deg1)+ \
            '_E'+ str(config.rows*config.columns) + '_A'+str(config.active_arrays)
    elif config.sources == 2:
        filename ='emul_'+'samples='+str(config.samples) + '_'+str(config.f_start1)+str(config.f_start2)+'Hz_'+'theta='+str(config.theta_deg1)+str(config.theta_deg2)+ \
        '_phi='+str(config.phi_deg1)+str(config.phi_deg2)+'_E'+ str(config.rows*config.columns) + '_A'+str(config.active_arrays)
    return filename

def frequency_phase_shift(signal, phase_shift, n_mics):
    FFT = np.fft.rfft(signal,axis=0) # Fourier transform each signal - 3D
    FFT = np.reshape(FFT, (int(N/2+1),n_mics,1,1)) # reshape into a 4D array from 2D
    FFT_shifted = FFT*phase_shift     # apply phase shift to every signal
    return FFT_shifted

def find_signal_peak(array, value, signal, source_indxes):
    signal = signal[:,source_indxes[0],source_indxes[1]]
    idx = np.argmin((np.abs(array - value)))
    max_vals = np.array([signal[idx-1], signal[idx], signal[idx+1]])
    idx = np.where(signal == np.max(max_vals))[0][0]
    return idx  # frekvensen av FFTns peakar motsvarar inte alltid idx som ges först, kan vara förskjuten ett steg till höger eller vänster

def validation_check(y_scan, x_scan):
    # Validation check
    xy_val_check = np.zeros((config.x_res,config.y_res)) # matrix holding values of validation map

    theta_s = np.array([config.theta_deg1, config.theta_deg2])*math.pi/180
    phi_s = np.array([config.phi_deg1, config.phi_deg2])*math.pi/180
    r_scan = config.z_scan/np.cos(theta_s)

    for source_ind in range(config.sources):
            x_source = r_scan[source_ind] * math.sin(theta_s[source_ind]) * math.cos(phi_s[source_ind]) # conv the angles to x-coord of source
            y_source = r_scan[source_ind] * math.sin(theta_s[source_ind]) * math.sin(phi_s[source_ind]) # conv the angles to y-coord of source
            x_ind = (np.abs(x_scan[0,0,:,0] - x_source)).argmin()    # find the x-index of the x_scan coordinate that is neares the true x-coord of the soruce
            y_ind = (np.abs(y_scan[0,0,0,:] - y_source)).argmin()   # find the y-index of the y_scan coordinate that is neares the true y-coord of the soruce
            xy_val_check[x_ind,y_ind] = 1   # set value to 1 at the coord of the source (all other values are 0)

    # visualize the validation map
    fig, ax = plt.subplots()
    plt.title('Actual location of sources')
    pic = ax.pcolormesh(x_scan[0,0,:,0], y_scan[0,0,0,:], xy_val_check.T )
    fig.colorbar(pic)



filename = 'emulated_data/' + generate_filename()
try:
    signal = np.float32(np.load(filename+'.npy',allow_pickle=True))
    print('Loading from memory: ' + filename)
except:
    generate_signals.main(generate_filename())
    signal = generate_signals.emulated_signals

# load index of the active microphones in the array setup
#       which microphones that are taken out are decided by 
#       the parameter mode in config.py (see active_microphones.py for more)
active_mics = am.active_microphones()
n_active_mics = len(active_mics)

# take out signals from the active microphones
signal = signal[:,am.active_microphones()]
n_elements = config.elements                # number of elements
d = config.distance                         # distance between elements
N = config.samples                          # number of samples
fs = config.f_sampling                      # samplefrequency

r_prime = calc_r_prime.calc_r_prime(d)      # r_prime holding coordinates of all microphones
theta = calc_phase_shift_cartesian.theta    # theta and phi vector
phi = calc_phase_shift_cartesian.phi

# the following variables are not currently used in this script
#   can be deleted??
#t = np.linspace(0,N/fs,N)           # time vector (1D)
#t = np.reshape(t, (len(t),1,1,1))   # reshaped time vector to 4D
#f = calc_phase_shift_cartesian.f    # frequency vector (4D)

print('Horizontal scanning window: ' + 'theta:', -int(np.rad2deg(theta)[0,0,0,round(config.y_res/2)]), \
      'to', int(np.rad2deg(theta)[0,0,config.x_res-1,round(config.y_res/2)]), 'deg')

x_scan = calc_phase_shift_cartesian.x_scan  
y_scan = calc_phase_shift_cartesian.y_scan

theta_source1 = config.theta_deg1               # direction of source 1
phi_source1 = config.phi_deg1                   # direction of source 1
theta_source2 = config.theta_deg2               # direction of source 2
phi_source2 = config.phi_deg2                   # direction of source 2

# --- PRINTS ---
print('True source angles:')
print('Source 1:\n' + 'theta = ' + str(int(theta_source1)) + ' deg' + '\n' + 'phi = ' + str(int(phi_source1)) + ' deg')
if config.sources == 2:
    print('Source 2:\n' + 'theta = ' + str(int(theta_source2)) + ' deg' + '\n' + 'phi = ' + str(int(phi_source2)) + ' deg')
    theta_source2 = np.deg2rad(theta_source2)
    phi_source2 = np.deg2rad(phi_source2)

# calculate the phase shift matrix for all scanning directions (x_scan, y_scan), for all microphones and all frequencies
phase_shift = calc_phase_shift_cartesian.phase_shift


def main():
    start = time.time()

    FFT_shifted = frequency_phase_shift(signal,phase_shift,n_active_mics)
    FFT_power = np.abs(np.sum(FFT_shifted,axis=1))**2
    FFT_power = FFT_power/n_active_mics    # normalise with number of array elements
    FFT_power_summed_freq = np.sum(FFT_power, axis=0)

    # inlagd test för bildbehandling (typ inte gjort så mycket med detta)
    P = 20
    imag = FFT_power_summed_freq**P # TEST, raise the value to some arbitrary power P to see how the result changes for the image
    imag /= np.max(imag)            # normalize w.r.t. maximum power
    
    end = time.time()
    print('Simulation time:', round((end - start), 4), 's')

    # plot heatmap
    fig, ax = plt.subplots(figsize = (16,9))
    pic = ax.pcolormesh(x_scan[0,0,:,0], y_scan[0,0,0,:], np.transpose(imag) )#,shading='gouraud') # heatmap summed over all frequencies
    fig.colorbar(pic)

    # save heatmap as .png in the folder plots/rainbow_road/
    #filename = generate_filename()+'_xres='+str(config.x_res)+'_yres='+str(config.x_res)
    #plt.savefig('plots/rainbow_road/'+filename+'.png', dpi = 500, format = 'png')

    # find strongest peak of source
    max_indxes = np.unravel_index(np.argmax(FFT_power), np.shape(FFT_power))
    x_max_indx = max_indxes[1]
    y_max_indx = max_indxes[2]

    print('Strongest source found at:')
    print('theta =', round(np.rad2deg(theta[0,0,x_max_indx,y_max_indx]),1), \
        '\t phi =', round(np.rad2deg(np.arctan2(y_scan[0,0,0,y_max_indx],x_scan[0,0,x_max_indx,0])),1)  )

    # check where the sources are actually placed
    validation_check(y_scan, x_scan)
    plt.show()

main()
