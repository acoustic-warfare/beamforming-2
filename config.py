import numpy as np
import math
# --- EMULATING DATA variables ---
f_sampling = 48828/2          # sampling frequency in Hz
#f_sampling = 60000         # sampling frequency in Hz
samples = int(512)               # number of samples to generate
t_start = 0                 # start time of simulation 
t_end = samples/f_sampling  # end time of simulation
c = 343

#  Emulated data
sources = 2                            # number of sources to emulate data from (supporting up to 2 sources)

# Source variables
away_distance = 700         # distance between the array and sources
# source1
f_start1 = 7000           # lowest frequency that the source emitts
f_end1 = f_start1               # highest frequency that the source emitts
f_res1 = 20                 # resolution of frequency
theta_deg1 = -20            # theta angel of source placement, relative to origin of array
phi_deg1 = 0              # phi angel of source placement, relative to origin of array
t_start1 = 0                # start time of emission
t_end1 = t_end               # end time of emission

# source2
f_start2 = 6000            # lowest frequency that the source emitts
f_end2 = f_start2               # highest frequency that the source emitts
f_res2 = 20                  # resolution of frequency
theta_deg2 = 20            # theta angel of source placement, relative to origin of array
phi_deg2 = 45              # phi angel of source placement, relative to origin of array
t_start2 = 0              # start time of emission
t_end2 = t_end                # end time of emission


# --- ANTENNA ARRAY setup variables ---
active_arrays = 2                       # number of arrays
rows = 4                                # number of rows
columns = 4                             # number of columns
elements = rows*columns*active_arrays   # number of elements
distance = 40 * 10**(-3)                # distance between elements (m)
d =  0 * 10**(-2)                       # distance between arrays (if more than one active)

# --- OTHER variables ---
# Number of modes for adaptive weights
modes = 7

# Beamforming resolution and scanning window
x_res = 20                          # resolution in x       just nu måste x_res = y_res!
y_res = 15                          # resolution in y
aspect_ratio = 16/9                 # Aspect ratio 16:9
alpha = 78                          # total scanning angle in theta-direction [degrees]
z_scan = 10

# Array factor plots
single_freq = 8*10**(3)
wavelength = c/single_freq

# --- PLOT OPTIONS ---
plot_single_f = 0
plot_setup = 1
plot_contourf = 1