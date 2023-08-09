# GENERATE AUDIO SIGNALS
# 
# Program that generates sine audio signals for each microphone in the microphone array
# by creating ideal sinus signals, and delaying them with a correct delay according to 
# the direction of arrival of the sound source.
#
# The program calculates the signals for all elements, assuming that the arrays are located in the
# xy-plane at z=0.
#
# If signals should be generated for several arrays, it is assumed that the arrays are stacked horizontally.
#
# Program can generate signals from up to 2 sources.

import numpy as np
import matplotlib.pyplot as plt
from Audio_source import Audio_source
import math
import calc_r_prime
import os
import config_test as config

def r_vec(theta,phi):
    r = np.array([(math.sin(theta)*math.cos(phi)), math.sin(theta)*math.sin(phi), math.cos(theta)])
    return r

def generate_array_signals(r_prime, sources):
    t_start = config.t_start                            # start time of simulation 
    t_end = config.t_end                                # end time of simulation
    samples = config.N_SAMPLES
    t = np.linspace(t_start, t_end, samples)    # time vector
    Audio_signal = np.zeros((len(t), len(r_prime[0,:])))

    for sample in range(len(t)):
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
                        k = 2*math.pi*frequencies_ps[freq_ind]/config.PROPAGATION_SPEED
                        r_1 = np.array([x_i,y_i,0])
                        r_2 = rho_soruce * r_vec(theta_source,phi_source)
                        norm_coeff = np.linalg.norm(r_2-r_1)
                        phase_offset = -k*norm_coeff
                        element_amplitude = 1/norm_coeff
                        temp_signal_sample += element_amplitude * math.sin(2*math.pi* frequencies_ps[freq_ind] * t[sample] + phase_offset)
            Audio_signal[sample,mic] = temp_signal_sample
    return Audio_signal

def calc_r_prime(d):
    half = d/2
    r_prime = np.zeros((2, config.N_MICROPHONES))
    element_index = 0
    for array in range(config.ACTIVE_ARRAYS):
        array *= -1
        for row in range(config.ROWS):
            for col in range(config.COLUMNS):
                r_prime[0,element_index] = - col*d - half + array*config.COLUMNS*d + array*config.ARRAY_SEPARATION + config.COLUMNS*config.ACTIVE_ARRAYS*half
                r_prime[1, element_index] = row*d + half - config.ROWS*half 
                element_index += 1
    r_prime[0,:] += (config.ACTIVE_ARRAYS-1)*config.ARRAY_SEPARATION/2
    return r_prime


def main(filename):
    # calculate r_prime for the array setup
    r_prime = calc_r_prime(config.ELEMENT_DISTANCE)

    # Create and place out sources
    source1 = Audio_source(config.f_start1, config.f_end1, config.f_res1, 
                        config.theta_deg1, config.phi_deg1, config.away_distance, 
                        config.t_start1, config.t_end1)
    source2 = Audio_source(config.f_start2, config.f_end2, config.f_res2,
                        config.theta_deg2, config.phi_deg2, config.away_distance, 
                        config.t_start2, config.t_end2)
    source_list = [source1, source2]
    sources = np.array(source_list[:config.sources])

    # GENERATE AUDIO SIGNAL
    directory = 'emulated_data'         # directory to store emulated data files
    if not os.path.exists(directory):   # create directory if non-existing
        os.makedirs(directory)
    
    print("Creating data file: "+filename)
    emulated_signals = generate_array_signals(r_prime, sources)
    np.save(directory+ '/'+filename, emulated_signals)             # save to .npy file

    print('Audio signals generated')