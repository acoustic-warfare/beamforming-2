import numpy
import ctypes
#Do not edit this config file! Add constants and expressions in config.json and build with make. 

#General constants for both c and python.
N_MICROPHONES = 192
N_SAMPLES = 512
N_TAPS = 64
COLUMNS = 8
ROWS = 8
MAX_RES = 11
MAX_RES_X = 11
MAX_RES_Y = 11
Z = 1.0
MAX_ANGLE = 62.0
VIEW_ANGLE = 62.0
UDP_PORT = 21844
ELEMENT_DISTANCE = 0.02
ARRAY_SEPARATION = 0.0
ACTIVE_ARRAYS = 3
SKIP_N_MICS = 1
PROPAGATION_SPEED = 343.0
APPLICATION_WINDOW_WIDTH = 720
APPLICATION_WINDOW_HEIGHT = 480
CAMERA_SOURCE = 2
FLIP_IMAGE = 1
APPLICATION_NAME = "BEEEEEAAAAAAM FOOOOOOORMING"
UDP_IP = "10.0.0.1"
UDP_REPLAY_IP = "127.0.0.1"
FPGA_PROTOCOL_VERSION = 2
BUFFER_LENGTH = N_SAMPLES * N_MICROPHONES
ASPECT_RATIO = 4/3
SAMPLE_RATE = 48828.0

#Python specific constants
azimuth = 0.0
elevation = 0.0
columns = 8
rows = 8
distance = 0.02
propagation_speed = 343.0
TIMEOUT = 30
FLIP_IMAGE = True
mode = 2
modes = 7
plot_setup = 0
threshold_freq_lower = 300
threshold_freq_upper = 10000
fs = int(48828)
DTYPE = ctypes.c_int32
NP_DTYPE = numpy.float32
WINDOW_SIZE = (720, 480)
