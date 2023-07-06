import numpy as np
import ctypes
import config
import matplotlib.pyplot as plt
import phase_shift_algorithm_real_data
import cv2
import time

def get_antenna_data():
    lib = ctypes.cdll.LoadLibrary("../lib/libsampler.so")

    init = lib.load
    init.restype = int

    get_data = lib.myread
    get_data.restype = None
    get_data.argtypes = [
        ctypes.POINTER(ctypes.c_float)
    ]

    init()
    
    return get_data


def get_samples():
    f = get_antenna_data()
    out = np.empty(config.BUFFER_LENGTH, dtype=np.float32)
    out_pointer = out.ctypes.data_as(
        ctypes.POINTER(ctypes.c_float))
    max_vals = np.array([])
    times = np.array([])
    i = 0
    while(True):
        start = time.time()
        f(out_pointer)
        signals = out.reshape((config.N_SAMPLES, config.N_MICROPHONES))
        signals = signals[:,0:config.ROWS*config.COLUMNS] # Use when only having one array, comment out otherwise
        #print(np.shape(signals))
        heatmap_data = phase_shift_algorithm_real_data.main(signals)
        #indxs = np.unravel_index(np.argmax(heatmap_data), np.shape(heatmap_data))
        #max_vals = np.append(max_vals, heatmap_data[indxs[0],indxs[1]])
        #print(heatmap_data, heatmap_data.dtype, heatmap_data.shape)
        heatmap_data /= 34000
        heatmap_data *= 255
        try:
            img = cv2.resize(heatmap_data, (960,540))
            img2 = img.astype('uint8')
            img2 = cv2.flip(img2, 1)
            img2 = cv2.applyColorMap(img2, cv2.COLORMAP_OCEAN)
            cv2.imshow('heatmap', img2)
            cv2.waitKey(int(30/30))
        except AttributeError:
            pass
        end = time.time()
        if i > 3:
            sim_time = round((end - start), 4)  # single loop simulation time
            #print('Individual loop time:', sim_time, 's')
            times = np.append(times, sim_time)
            print('Avg simulation time:', round(np.sum(times)/(i-3), 4), 's')
        i += 1
get_samples()
exit()