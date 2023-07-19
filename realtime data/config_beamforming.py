import numpy as np
# --- EMULATING DATA variables ---
f_sampling = 48828          # sampling frequency in Hz
samples = 512               # number of samples to generate
t_start = 0                 # start time of simulation 
t_end = samples/f_sampling  # end time of simulation
c = 343

# --- ANTENNA ARRAY setup variables ---
active_arrays = 1                  # number of arrays
rows = 8                    # number of rows
columns = 8                 # number of columns
elements = rows*columns*active_arrays     # number of elements
distance = 20 * 10**(-3)    # distance between elements (m)
d = 0 * 10**(-2)     # distance between arrays (m)

# --- OTHER variables ---
# Number of modes for adaptive weights
modes = 7

# Beamforming resolution and scanning window
x_res = 19                          # resolution in x, use odd number
y_res = 19                          # resolution in y, use odd number
aspect_ratio = 4/3                 # Aspect ratio 16:9
alpha = 68                         # total scanning angle in theta-direction [degrees]
z_scan = 1                          # [m]

plot_setup = 0