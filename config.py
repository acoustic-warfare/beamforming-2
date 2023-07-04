import numpy as np
# --- EMULATING DATA variables ---
f_sampling = 48828          # sampling frequency in Hz
#f_sampling = 60000         # sampling frequency in Hz
samples = 512               # number of samples to generate
t_start = 0                 # start time of simulation 
t_end = samples/f_sampling  # end time of simulation
c = 343

# Source variables
away_distance = 700         # distance between the array and sources
# source1
f_start1 = 11000           # lowest frequency that the source emitts
f_end1 = 11600               # highest frequency that the source emitts
f_res1 = 20                 # resolution of frequency
theta_deg1 = 20             # theta angel of source placement, relative to origin of array
phi_deg1 = 25              # phi angel of source placement, relative to origin of array
t_start1 = 0                # start time of emission
t_end1 = t_end               # end time of emission

# source2
f_start2 = 14000            # lowest frequency that the source emitts
f_end2 = 13600               # highest frequency that the source emitts
f_res2 = 20                  # resolution of frequency
theta_deg2 = 30            # theta angel of source placement, relative to origin of array
phi_deg2 = -25              # phi angel of source placement, relative to origin of array
t_start2 = 0              # start time of emission
t_end2 = t_end                # end time of emission


# --- ANTENNA ARRAY setup variables ---
r_a1 = [0, 0, 0]            # coordinate position of origin of array1
r_a2 = [0.08, 0, 0]         # coordinate position of origin of array2
r_a3 = [-0.24, 0, 0]        # coordinate position of origin of array3
r_a4 = [0.24, 0, 0]         # coordinate position of origin of array4
active_arrays = 1                  # number of arrays
rows = 7                    # number of rows
columns = 8                 # number of columns
elements = rows*columns*active_arrays     # number of elements
distance = 20 * 10**(-3)    # distance between elements (m)

# --- OTHER variables ---
# Number of modes for adaptive weights
modes = 7

# Beamforming resolution and scanning window
x_res = 24                          # resolution in x       just nu måste x_res = y_res!
y_res = 24                          # resolution in y
aspect_ratio = 16/9                 # Aspect ratio 16:9
alpha = 78                          # total scanning angle in theta-direction [degrees]
z_scan = 10
gridboxes_axis = 60                            # används för phase_shift_freq_multi_dim.py             
theta = np.linspace(0,np.pi/3,gridboxes_axis)  # används för phase_shift_freq_multi_dim.py
phi = np.linspace(-np.pi,np.pi,gridboxes_axis) # används för phase_shift_freq_multi_dim.py


# Listening values
theta_listen = 0                    # direction to listen in, theta angle
phi_listen = 45                     # direction to listen in, theta angle


# --- RECORDED OR EMULATED DATA ---
#   choose if the program should work with recorded or emulated data
audio_signals = 'emulated'              # 'emulated' or 'recorded'

#  Emulated data
sources = 2                            # number of sources to emulate data from (supporting up to 2 sources)

# Recorded data
filename = 'filename'                   # filename of recorded data
path = ''   # path to file

plot_setup = 0