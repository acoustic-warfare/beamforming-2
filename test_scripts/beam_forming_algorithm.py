import numpy as np
np.set_printoptions(threshold=np.inf)
import time
import config
import calc_phase_shift_cartesian
import generate_signals
import config_mika
import matplotlib.pyplot as plt
import math


if config.data_type == 'emulated':
    def generate_filename():
        if config_mika.sources == 1:
            filename ='emul_'+ 'samples='+str(config_mika.samples) + '_'+ str(config_mika.f_start1)+'Hz_'+'theta='+str(config_mika.theta_deg1)+'_phi='+str(config_mika.phi_deg1)+ \
                '_A'+str(config_mika.active_arrays)
        elif config_mika.sources == 2:
            filename ='emul_'+'samples='+str(config_mika.samples) + '_'+str(config_mika.f_start1)+str(config_mika.f_start2)+'Hz_'+'theta='+str(config_mika.theta_deg1)+str(config_mika.theta_deg2)+ \
            '_phi='+str(config_mika.phi_deg1)+str(config_mika.phi_deg2)+'_A'+str(config_mika.active_arrays)
        return filename

    filename = generate_filename()
    try:
        signal = np.float32(np.load('emulated_data/' + filename+'.npy',allow_pickle=True))
        print('Loading from memory: ' + filename)
    except:
        generate_signals.main(filename)
        signal = generate_signals.emulated_signals


if config.data_type == 'recorded':
    filename = config.filename
    signal = np.load(filename)
    signal = signal.T
    signal = signal[20000:20512,0:192]      # take out 512 successive samples from signal, first 
    #print(np.shape(signal))
    filename = filename.replace(".npy","")
    #filename = filename + 'good_mics'

n_elements = config.N_MICROPHONES
n_samples = config.N_SAMPLES     
f = calc_phase_shift_cartesian.f      
theta = calc_phase_shift_cartesian.theta
print('Scanning window in horizontal direction:\n' + 'theta:', -int(np.rad2deg(theta)[0,0,0,round(config.MAX_RES_Y/2)]), 'to', int(np.rad2deg(theta)[0,0,config.MAX_RES_X-1,round(config.MAX_RES_Y/2)]), 'deg')

x_scan = calc_phase_shift_cartesian.x_scan
y_scan = calc_phase_shift_cartesian.y_scan
x_res = config.MAX_RES_X
y_res = config.MAX_RES_Y

phase_shift = calc_phase_shift_cartesian.phase_shift
phase_shift_modes = calc_phase_shift_cartesian.phase_shift_modes
n_active_mics = calc_phase_shift_cartesian.n_active_mics

mode_matrices = calc_phase_shift_cartesian.mode_matrices
mode_intervals = calc_phase_shift_cartesian.mode_intervals
active_mics_mode_list = calc_phase_shift_cartesian.active_mics_mode_list

def frequency_phase_shift(signal, phase_shift):
    signal = signal[:,calc_phase_shift_cartesian.active_mics]
    FFT = np.fft.rfft(signal,axis=0) # Fourier transform each signal
    FFT = np.reshape(FFT, (int(n_samples/2)+1,len(FFT[0,:]),1,1)) # reshape into a 4D array from 2D
    FFT_shifted = FFT*phase_shift    # apply phase shift to every signal
    return FFT_shifted

def frequency_phase_shift_modes(signal):
    FFT = np.fft.rfft(signal,axis=0) # Fourier transform each signal
    FFT = np.reshape(FFT, (int(n_samples/2)+1,len(FFT[0,:]),1,1)) # reshape into a 4D array from 2D
    FFT_shifted_list = []
    for mode in range(config.modes):
        FFT_shifted_mode = FFT[mode_intervals[mode][0]:mode_intervals[mode][-1]+1,active_mics_mode_list[mode]]*mode_matrices[mode]
        FFT_shifted_list.append(FFT_shifted_mode)
    return FFT_shifted_list

def remove_neighbors(heatmap, x_index, y_index):
    n_neighbors = 5 # numbers of neighbors to remove in x- and y-direction
    x_start = x_index-n_neighbors
    x_end = x_index+n_neighbors+1
    y_start = y_index-n_neighbors
    y_end = y_index+n_neighbors+1
    if x_start < 0:
        x_start = 0
    if y_start < 0:
        y_start = 0
    
    heatmap[x_start:x_end, y_start:y_end] = 0

def peak_detection(power):
    #power_copy = np.copy(power)        # use power_copy when handling sources at the same frequency
    heatmap = np.zeros((config.MAX_RES_X,config.MAX_RES_Y))
    for f_ind in range(0, int(n_samples/2)+1):
        if (np.max(power[f_ind,:,:]) > config.threshold_upper*np.max(power) > config.threshold_lower):
            (x_max,y_max) = np.unravel_index(power[f_ind,:,:].argmax(), np.shape(power[f_ind,:,:])) #indexes for x- and y-position of the maximum peak
            if power[f_ind,x_max,y_max] > heatmap[x_max,y_max]:
                heatmap[x_max,y_max] = power[f_ind,x_max,y_max]
                #heatmap[x_max,y_max] += 1
            #print('Found peak value:', round(power[f_ind,x_max,y_max], 8), 'at approx.', int(f[f_ind]), 'Hz')
            
            ## Handling sources at the same frequency, doesn't work very well
            #remove_neighbors(power_copy[f_ind,:,:],x_max,y_max)
            #threshold_freq = 0.7    # set between 0 and 1
            #P = 10
            #while np.max(power_copy[f_ind,:,:])**P > threshold_freq*np.max(power)**P:
            #    (x_max,y_max) = np.unravel_index(power_copy[f_ind,:,:].argmax(), np.shape(power_copy[f_ind,:,:]))
            ##    #heatmap[x_max,y_max] += 1
            #    heatmap[x_max,y_max] += power_copy[f_ind,x_max,y_max]
            ##    #print('Found peak value:', round(FFT[f_ind,x_max,y_max]), 'at approx.', int(f[f_ind]), 'Hz')
            #    remove_neighbors(power_copy[f_ind,:,:],x_max,y_max)
    return heatmap

def peak_detection_modes(power_list):
    #power_copy = np.copy(power_list)
    power_stack = power_list[0]
    for power_matrix in power_list[1:]:
        power_stack = np.vstack((power_matrix,power_stack))

    heatmap = np.zeros((config.MAX_RES_X,config.MAX_RES_Y))
    for f_ind in range(0, int(n_samples/2)+1):
        if (np.max(power_stack[f_ind,:,:]) > config.threshold_upper*np.max(power_stack) > config.threshold_lower):
            (x_max,y_max) = np.unravel_index(power_stack[f_ind,:,:].argmax(), np.shape(power_stack[f_ind,:,:])) #indexes for x- and y-position of the maximum peak
            if power_stack[f_ind,x_max,y_max] > heatmap[x_max,y_max]:
                #heatmap[x_max,y_max] = power_stack[f_ind,x_max,y_max]
                heatmap[x_max,y_max] = 1
            #print('Found peak value:', round(power_stack[f_ind,x_max,y_max], 8), 'at approx.', int(f[f_ind]), 'Hz')
    return heatmap

def calc_power(matrix, active_mics):
    matrix_power = (np.abs(np.sum(matrix,axis=1))**2)
    matrix_power = matrix_power/len(active_mics)
    return matrix_power


#### used for checking emulated data
def validation_check(y_scan, x_scan):
    # Validation check
    xy_val_check = np.zeros((config.MAX_RES_X,config.MAX_RES_Y)) # matrix holding values of validation map

    theta_s = np.array([config_mika.theta_deg1, config_mika.theta_deg2])*math.pi/180
    phi_s = np.array([config_mika.phi_deg1, config_mika.phi_deg2])*math.pi/180
    r_scan = config_mika.z_scan/np.cos(theta_s)

    for source_ind in range(config_mika.sources):
            x_source = r_scan[source_ind] * math.sin(theta_s[source_ind]) * math.cos(phi_s[source_ind]) # conv the angles to x-coord of source
            y_source = r_scan[source_ind] * math.sin(theta_s[source_ind]) * math.sin(phi_s[source_ind]) # conv the angles to y-coord of source
            x_ind = (np.abs(x_scan[0,0,:,0] - x_source)).argmin()    # find the x-index of the x_scan coordinate that is neares the true x-coord of the soruce
            y_ind = (np.abs(y_scan[0,0,0,:] - y_source)).argmin()   # find the y-index of the y_scan coordinate that is neares the true y-coord of the soruce
            xy_val_check[x_ind,y_ind] = 1   # set value to 1 at the coord of the source (all other values are 0)
            #print(x_ind)
            #print(y_ind)
    # visualize the validation map
    fig, ax = plt.subplots()
    plt.title('Actual location of source(s)')
    levels = np.linspace(0, 1, 50)
    pic = ax.contourf(x_scan[0,0,:,0], y_scan[0,0,0,:], xy_val_check.T, levels, cmap=plt.get_cmap('jet'))
    plt.axis('off')
    #plt.xlabel('x')     
    #plt.ylabel('y')
    filename_sources = generate_filename()+'source_location'
    #plt.savefig('plots/'+filename_sources+'.png', dpi = 500, format = 'png', bbox_inches='tight')
    #fig.colorbar(pic)
####

def main(signal):
    #### calculation of heatmap with full phase shift matrix
    start_standard = time.time()
    FFT_shifted = frequency_phase_shift(signal,phase_shift)
    FFT_power = (np.abs(np.sum(FFT_shifted,axis=1)))**2         # Power of FFT summed over all array element
    FFT_power /= config_mika.elements
    #FFT_power = (FFT_power/(np.max(FFT_power)+0.0001))#**2
    FFT_power /= np.max(FFT_power)
    #heatmap = np.sum(FFT_power,axis=0)
    FFT_power_summed_freq = np.sum(FFT_power,axis=0)
    FFT_power_summed_freq = (FFT_power_summed_freq/(np.max(FFT_power_summed_freq)+0.0001))**3
    end_standard = time.time()
    print('Standard algorithm time:', round((end_standard - start_standard), 4), 's')
    #time_standard = round((end_standard - start_standard), 4)
    heatmap = peak_detection(FFT_power)
    heatmap = (heatmap/(np.max(heatmap)+0.000001))#**2

    #### calculation of heatmap with full mode phase shift matrix
    start_modes_reg = time.time()
    FFT_shifted_modes = frequency_phase_shift(signal,phase_shift_modes)
    FFT_power_modes = (np.abs(np.sum(FFT_shifted_modes,axis=1))**2)         # Power of FFT summed over all array elements
    FFT_power_modes /= n_active_mics                                  # adaptive normalization
    FFT_power_modes /= np.max(FFT_power_modes)

    FFT_power_summed_freq_modes = np.sum(FFT_power_modes,axis=0)
    FFT_power_summed_freq_modes = (FFT_power_summed_freq_modes/(np.max(FFT_power_summed_freq_modes)+0.0001))**3
    end_modes_reg = time.time()
    print('Regular modes algorithm time:', round(end_modes_reg - start_modes_reg, 4), 's')
    #time_modes_reg = round(end_modes_reg - start_modes_reg, 4)

    ## peak detection using full matrices
    heatmap_modes = peak_detection(FFT_power_modes)
    heatmap_modes = (heatmap_modes/(np.max(heatmap_modes)+0.000001))#**2
    #heatmap_modes = (heatmap_modes/np.max(heatmap_modes))

    #### calculation with mode matrix separated
    start_sep= time.time()
    FFT_shifted_sep = frequency_phase_shift_modes(signal)
    power_list = list(map(calc_power, FFT_shifted_sep, active_mics_mode_list))
    # regular heatmap calculated with separated modes
    FFT_power_summed_freq_sep = sum(list(map(lambda x: np.sum(x,axis=0), power_list)))   # sum the power of each FFT modes over frequency, then sum all modes together
    FFT_power_summed_freq_sep = (FFT_power_summed_freq_sep/(np.max(FFT_power_summed_freq_sep)+0.0000001))**3
    end_sep= time.time()
    print('Separated modes algorithm time:', round(end_sep - start_sep, 4), 's')
    #time_sep = round(end_sep - start_sep, 4)

    # peak detection using separated mode matrices
    heatmap_sep = peak_detection_modes(power_list)

    #### plots
    fig, ax = plt.subplots()
    ax.pcolormesh(x_scan[0,0,:,0], y_scan[0,0,0,:], np.transpose(FFT_power_summed_freq)) # heatmap summed over all frequencies
    levels = np.linspace(np.min(FFT_power_summed_freq), np.max(FFT_power_summed_freq), 50)
    plt.pcolormesh(x_scan[0,0,:,0], y_scan[0,0,0,:], np.transpose(FFT_power_summed_freq), cmap=plt.get_cmap('jet'))
    plt.axis('off')
    #plt.gca().set_aspect(9/16)       # X/Y gives aspect ratio of Y:X 
    plt.gca().set_aspect('auto')
    plt.title('Standard heatmap')
    #plt.savefig('plots/'+filename+'_real_data.png', dpi = 500, format = 'png', bbox_inches='tight')
    #plt.xlabel('x')     
    #plt.ylabel('y')
    #plt.colorbar()
#
    fig, ax = plt.subplots()
    ax.pcolormesh(x_scan[0,0,:,0], y_scan[0,0,0,:], np.transpose(heatmap)) # heatmap summed over all frequencies
    levels = np.linspace(np.min(heatmap), np.max(heatmap), 50)
    plt.contourf(x_scan[0,0,:,0], y_scan[0,0,0,:], np.transpose(heatmap), levels, cmap=plt.get_cmap('jet'))
    plt.axis('off')
    #plt.gca().set_aspect(9/16)       # X/Y gives aspect ratio of Y:X 
    plt.gca().set_aspect('auto')
    plt.title('Heatmap using peak detection')
    #plt.savefig('plots/'+filename+'_real_data_peak.png', dpi = 500, format = 'png', bbox_inches='tight')
    #plt.xlabel('x')     
    #plt.ylabel('y')
    #plt.colorbar()

    fig, ax = plt.subplots()
    ax.pcolormesh(x_scan[0,0,:,0], y_scan[0,0,0,:], np.transpose(FFT_power_summed_freq_modes)) # heatmap summed over all frequencies
    levels = np.linspace(np.min(FFT_power_summed_freq_modes), np.max(FFT_power_summed_freq_modes), 50)
    plt.pcolormesh(x_scan[0,0,:,0], y_scan[0,0,0,:], np.transpose(FFT_power_summed_freq_modes), cmap=plt.get_cmap('jet'))
    plt.axis('off')
    #plt.gca().set_aspect(9/16)       # X/Y gives aspect ratio of Y:X 
    plt.gca().set_aspect('auto')
    plt.title('Standard heatmap using adaptive config')
    #plt.savefig('plots/'+filename+'_real_data_adaptive.png', dpi = 500, format = 'png', bbox_inches='tight')
    #plt.xlabel('x')     
    #plt.ylabel('y')
    #plt.colorbar()

    fig, ax = plt.subplots()
    ax.pcolormesh(x_scan[0,0,:,0], y_scan[0,0,0,:], np.transpose(heatmap_modes)) # heatmap summed over all frequencies
    levels = np.linspace(np.min(heatmap_modes), np.max(heatmap_modes), 50)
    plt.contourf(x_scan[0,0,:,0], y_scan[0,0,0,:], np.transpose(heatmap_modes), levels, cmap=plt.get_cmap('jet'))
    plt.axis('off')
    #plt.gca().set_aspect(9/16)       # X/Y gives aspect ratio of Y:X 
    plt.gca().set_aspect('auto')
    plt.title('Heatmap using adaptive configuration and peak detection')
    #plt.savefig('plots/'+filename+'_real_data_adaptive_peak.png', dpi = 500, format = 'png', bbox_inches='tight')
    #plt.xlabel('x')     
    #plt.ylabel('y')
    #plt.colorbar()

    fig, ax = plt.subplots()
    ax.pcolormesh(x_scan[0,0,:,0], y_scan[0,0,0,:], np.transpose(FFT_power_summed_freq_sep)) # heatmap summed over all frequencies
    levels = np.linspace(np.min(FFT_power_summed_freq_sep), np.max(FFT_power_summed_freq_sep), 50)
    plt.pcolormesh(x_scan[0,0,:,0], y_scan[0,0,0,:], np.transpose(FFT_power_summed_freq_sep), cmap=plt.get_cmap('jet'))
    plt.axis('off')
    #plt.gca().set_aspect(9/16)       # X/Y gives aspect ratio of Y:X 
    plt.gca().set_aspect('auto')
    plt.title('Regular heatmap using separated mode matrices')
    #plt.xlabel('x')     
    #plt.ylabel('y')
    #plt.colorbar()

    fig, ax = plt.subplots()
    ax.pcolormesh(x_scan[0,0,:,0], y_scan[0,0,0,:], np.transpose(heatmap_sep)) # heatmap summed over all frequencies
    levels = np.linspace(np.min(heatmap_sep), np.max(heatmap_sep), 50)
    plt.contourf(x_scan[0,0,:,0], y_scan[0,0,0,:], np.transpose(heatmap_sep), levels, cmap=plt.get_cmap('jet'))
    plt.axis('off')
    #plt.gca().set_aspect(9/16)       # X/Y gives aspect ratio of Y:X 
    plt.gca().set_aspect('auto')
    plt.title('Heatmap with peak detection using separated mode matrices')
    #plt.xlabel('x')     
    #plt.ylabel('y')
    #plt.colorbar()

    if config.data_type == 'emulated':
        validation_check(y_scan,x_scan)

    plt.show()
    return heatmap

main(signal)
