# Script that calculates the time delay (in samples) for each microphone
import math
import numpy as np
import config

def calc_r_prime(d):
    half = d/2
    r_prime = np.zeros((2, config.elements))
    element_index = 0
    for array in range(config.active_arrays):
        for row in range(config.rows):
            for col in range(config.columns):
                r_prime[0,element_index] = col * d + half + array*config.columns*d + array*config.sep - config.columns* config.active_arrays * half
                r_prime[1, element_index] = row * d - config.rows * half + half
                element_index += 1
    r_prime[0,:] -= config.active_arrays*config.sep/2
    active_mics, n_active_mics = active_microphones()

    r_prime = r_prime[:,active_mics]
    return r_prime

def active_microphones():
    mode = config.mode
    rows = np.arange(0, config.rows, mode)
    columns = np.arange(0, config.columns*config.active_arrays, mode)

    mics = np.linspace(0, config.elements-1, config.elements)   # mics in one array
    arr_elem = config.rows*config.columns                       # elements in one array
    microphones = np.linspace(0, config.rows*config.columns-1,config.rows*config.columns).reshape((config.rows, config.columns))

    for a in range(config.active_arrays-1):
        a += 1
        array = mics[0+a*arr_elem : arr_elem+a*arr_elem].reshape((config.rows, config.columns))
        microphones = np.hstack((microphones, array))

    active_mics = []
    for r in rows:
        for c in columns:
            mic = microphones[r,c]
            active_mics.append(int(mic))
    return np.sort(active_mics), len(active_mics)


# CONFIG FILE NEEDS:
#   c, f_sampling, 
#   samples,
#   rows, columns   (rows and cloumns in one array)
#   active_arrays   (number of arrays to get signlas from)
#   distance        (between elements), 
#   sep             (separation between arrays)
#   mode            (vilka element som används, 1 -> alla, 2 -> varannan, 3 -> var tredje)
#   alpha           (total scanning angle (bildvinkel) in x-direction [degrees])
#   z_scan          (distance to scanning window, kan sättas till något random, tex 10)
#   x_res, y_res    (resolution in x- and y-direction of scanning window)
#   AS              (aspect ratio of camera)

c = config.c             # from config
fs = config.f_sampling          # from config
N_SAMPLES = config.samples      # from config
d = config.distance            # distance between elements, from config

alpha = config.alpha  # total scanning angle (bildvinkel) in theta-direction [degrees], from config
z_scan = config.z_scan  # distance to scanning window, from config

x_res = config.x_res  # resolution in x, from config
y_res = config.y_res  # resolution in y, from config
AS = config.aspect_ratio  # aspect ratio, from config

# Calculations for time delay starts below
r_prime = calc_r_prime(d)  # matrix holding the xy positions of each microphone
x_i = r_prime[0,:]                      # x-coord for microphone i
y_i = r_prime[1,:]                      # y-coord for microphone i

# outer limits of scanning window in xy-plane
x_scan_max = z_scan*np.tan((alpha/2)*math.pi/180)
x_scan_min = - x_scan_max
y_scan_max = x_scan_max/AS
y_scan_min = -y_scan_max

# scanning window
x_scan = np.linspace(x_scan_min, x_scan_max, x_res).reshape(x_res,1,1)
y_scan = np.linspace(y_scan_min, y_scan_max, y_res).reshape(1, y_res, 1)
r_scan = np.sqrt(x_scan**2 + y_scan**2 + z_scan**2) # distance between middle of array to the xy-scanning coordinate

# calculate time delay (in number of samples)
samp_delay = (fs/c) * (x_scan*x_i + y_scan*y_i) / r_scan   # with shape: (x_res, y_res, n_active_mics)

# adjust such that the microphone furthest away from the beam direction have 0 delay
samp_delay -= np.amin(samp_delay, axis=2).reshape(x_res, y_res, 1)