# AUDIO SIGNALS
import numpy as np
import matplotlib.pyplot as plt
from Audio_data import Audio_data
from Audio_source import Audio_source
import math
import config_mika as config
import calc_r_prime


def r_vec(theta,phi):
    r = np.array([(math.sin(theta)*math.cos(phi)), math.sin(theta)*math.sin(phi), math.cos(theta)])
    return r

def generate_array_signals(r_prime, sources, t):
    Audio_signal = np.zeros((len(t), len(r_prime[0,:])))

    for sample in range(len(t)):
        #if (sample+1 in np.arange(0,len(t),10)) or (sample == 0): # print stuff so user know how many samples that have been generated
        #    print(sample+1)                                         # print stuff so user know how many samples that have been generated
        for mic in range(len(r_prime[0,:])):
            x_i = r_prime[0,mic]
            y_i = r_prime[1,mic]
            temp_signal_sample = 0
            for source in range(len(sources)):
                if (sources[source].get_t_start() < t[sample]) and (t[sample] < sources[source].get_t_end()):
                    frequencies_ps = sources[source].get_frequency()
                    theta_source = sources[source].get_theta()
                    phi_source = sources[source].get_phi()
                    rho_soruce = sources[source].get_rho()
                    for freq_ind in range(len(frequencies_ps)):
                        k = 2*math.pi*frequencies_ps[freq_ind]/config.c
                        r_1 = np.array([x_i,y_i,0])
                        r_2 = rho_soruce * r_vec(theta_source,phi_source)
                        norm_coeff = np.linalg.norm(r_2-r_1)
                        phase_offset = -k*norm_coeff
                        element_amplitude = 1/norm_coeff
                        temp_signal_sample += element_amplitude * math.sin(2*math.pi* frequencies_ps[freq_ind] * t[sample] + phase_offset)
            Audio_signal[sample,mic] = temp_signal_sample
    return Audio_signal

#def calc_r_prime(d):
#    half = d/2
#    r_prime = np.zeros((2, config.elements))
#    element_index = 0
#    for row in range(config.rows):
#        for col in range(config.columns*config.active_arrays):
#
#            r_prime[0,element_index] = col * d - config.columns * config.active_arrays * half + half
#            r_prime[1, element_index] = row * d - config.rows * half + half
#            element_index += 1
#
#    if config.plot_setup:
#        plt.figure()
#        plt.title('Array setup')
#        plt.scatter(r_prime[0,:], r_prime[1,:].T)
#        plt.xlim([-(d*config.columns * config.active_arrays/2 + d) , d*config.columns * config.active_arrays/2 + d])
#    return r_prime
#def calc_r_prime(d):
#    half = d/2
#    r_prime = np.zeros((2, config.elements))
#    element_index = 0
#    for array in range(config.active_arrays):
#        for row in range(config.rows):
#            for col in range(config.columns):
#                r_prime[0,element_index] = col * d + half + array*config.columns*d + array*config.sep - config.columns* config.active_arrays * half
#                r_prime[1, element_index] = row * d - config.rows * half + half
#                element_index += 1
#    r_prime[0,:] -= config.active_arrays*config.sep/2
#    return r_prime

def main(filename):
    #filename = generate_filename()  # generate filename

    r_prime_all, r_prime = calc_r_prime.calc_r_prime(config.distance)

    # Create and place out sources
    source1 = Audio_source(config.f_start1, config.f_end1, config.f_res1, 
                        config.theta_deg1, config.phi_deg1, config.away_distance, 
                        config.t_start1, config.t_end1)
    source2 = Audio_source(config.f_start2, config.f_end2, config.f_res2,
                        config.theta_deg2, config.phi_deg2, config.away_distance, 
                        config.t_start2, config.t_end2)
    source_list = [source1, source2]
    sources = np.array(source_list[:config.sources])

    print("Creating data file: "+filename)
    fs = config.f_sampling                      # sampling frequency in Hz
    t_start = config.t_start                            # start time of simulation 
    t_end = config.t_end                                # end time of simulation
    #t_total = t_end - t_start                           # total simulation time
    samples = config.samples
    t = np.linspace(t_start, t_end, samples) # time vector


    # GENERATE AUDIO SIGNAL
    global emulated_signals
    emulated_signals = generate_array_signals(r_prime, sources, t)
    np.save('emulated_data/'+filename, emulated_signals)        # save to file

    print('Audio signals generated')
#main()
