
#General constants for both c and python.
N_SAMPLES = 512             # number of samples in signal
COLUMNS = 8                 # number of columns in one array
ROWS = 8                    # number of rows in one array
ELEMENT_DISTANCE = 0.02     # distance between microphone elements (m)
ARRAY_SEPARATION = 0.0      # separation between arrays
ACTIVE_ARRAYS = 3           # number of active arrays
N_MICROPHONES = ROWS*COLUMNS*ACTIVE_ARRAYS
MAX_RES_X = 11              # max resultion in x-direction (scanning window)
MAX_RES_Y = 11              # max resultion in y-direction (scanning window)
Z = 1.0                     # z-position of scanning window
VIEW_ANGLE = 62.0           # view angle of camera image in horizontal direction
SKIP_N_MICS = 1
PROPAGATION_SPEED = 343.0   # speed of sound
ASPECT_RATIO = 16/9         # aspect ratio of camera
SAMPLE_RATE = 48828.0       # sample rate of signal
ELEMENTS = ROWS*COLUMNS*ACTIVE_ARRAYS

mode = 1
modes = 7
plot_setup = 0

# FFT USER SETTINGS
threshold_freq_lower = 0
threshold_freq_upper = 18000
threshold_upper = 0.6             # threshold value for detecting peak values (set between 0 and 1)
threshold_lower = 0.00017         # threshold value for detecting peak values (can be set to any value)
fs = SAMPLE_RATE

# --- EMULATING DATA variables ---
t_start = 0                 # start time of simulation 
t_end = N_SAMPLES/fs        # end time of simulation

sources = 1                 # number of sources to emulate data from (supporting up to 2 sources)
# Source variables
away_distance =  1          # distance between the array and sources
# source1
f_start1 = 4000             # lowest frequency that the source emitts
f_end1 = 4000               # highest frequency that the source emitts
f_res1 = 1                 # resolution of frequency
theta_deg1 = 27             # theta angel of source placement, relative to origin of array
phi_deg1 = -45              # phi angel of source placement, relative to origin of array
t_start1 = 0                # start time of emission
t_end1 = t_end              # end time of emission

# source2
f_start2 = 6000             # lowest frequency that the source emitts
f_end2 = 7000               # highest frequency that the source emitts
f_res2 = 20                 # resolution of frequency
theta_deg2 = 27             # theta angel of source placement, relative to origin of array
phi_deg2 = 135              # phi angel of source placement, relative to origin of array
t_start2 = 0                # start time of emission
t_end2 = t_end              # end time of emission


### user settings for FFT beamforming 
data_type = 'emulated'      # options: 'emulated', 'recorded'
filename = 'sine1khz.npy'   # used for recorded data
mics_used = 'all'           # options: 'all', 'only good'


###
