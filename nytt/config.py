import numpy as np
import math
# --- EMULATING DATA variables ---
f_sampling = 48828          # sampling frequency in Hz
samples = 1024               # number of samples to generate
t_start = 0                 # start time of simulation 
t_end = samples/f_sampling  # end time of simulation
c = 343

#  Emulated data
sources = 2                            # number of sources to emulate data from (supporting up to 2 sources)
# Source variables
away_distance =  10       # distance between the array and sources
# source1
f_start1 = 5000           # lowest frequency that the source emitts
f_end1 = 5000               # highest frequency that the source emitts
f_res1 = 1                 # resolution of frequency
theta_deg1 = 30            # theta angel of source placement, relative to origin of array
phi_deg1 = 10              # phi angel of source placement, relative to origin of array
t_start1 = 0                # start time of emission
t_end1 = t_end               # end time of emission

# source2
f_start2 = 300            # lowest frequency that the source emitts
f_end2 = 3000               # highest frequency that the source emitts
f_res2 = 10                  # resolution of frequency
theta_deg2 = 30            # theta angel of source placement, relative to origin of array
phi_deg2 = -180              # phi angel of source placement, relative to origin of array
t_start2 = 0              # start time of emission
t_end2 = t_end                # end time of emission


# --- ANTENNA ARRAY setup variables ---
mode = 2                                # mode of which microphones to use or not
active_arrays = 2                       # number of arrays
rows = 8                                # number of rows
columns= 8                             # number of columns
elements = rows*columns*active_arrays   # number of elements
distance = 20 * 10**(-3)                # distance between elements (m)
sep =  0 * 10**(-2)                     # separation between arrays (if more than one active)


# Beamforming resolution and scanning window
x_res = 101                         # resolution in x       just nu måste x_res = y_res!
y_res = 101                         # resolution in y
aspect_ratio = 4/3                 # Aspect ratio 16:9
alpha = 78                          # total scanning angle in theta-direction [degrees]
z_scan = 10

# Array factor
single_freq = 8*10**(3)
wavelength = c/single_freq
adaptive = 0

# --- PLOT OPTIONS ---
plot_single_f = 0
plot_setup = 1
plot_contourf = 1