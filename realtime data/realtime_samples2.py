import numpy as np
import ctypes
import config
import matplotlib.pyplot as plt
import phase_shift_algorithm_real_data
import cv2
import time

from lib.microphone_array import receive, connect, disconnect




def get_samples():
    times = np.array([])

    data = np.empty((config.N_MICROPHONES, config.N_SAMPLES), dtype=np.float32)
    i = 0
    while(True):
        start = time.time()
        # f(out_pointer)

        receive(data)

        out = np.copy(data)

        
        # signals = out.reshape((config.N_SAMPLES, config.N_MICROPHONES))
        signals = out.T
        # print(signals[:,0])
        signals = signals[:,0:config.ROWS*config.COLUMNS] # Use when only having one array, comment out otherwise
        #print(np.shape(signals))
        heatmap_data = phase_shift_algorithm_real_data.main(signals)
        # print(heatmap_data, heatmap_data.dtype, heatmap_data.shape)

        #indxs = np.unravel_index(np.argmax(heatmap_data), np.shape(heatmap_data))
        #max_val = heatmap_data[indxs[0],indxs[1]]
        #heatmap_data /= max_val    # normalizing with maximum value

        max_val = np.max(heatmap_data)

        print(max_val)

        if max_val > 0.5:

            heatmap_data /= max_val
            heatmap_data **= 3
            heatmap_data *= 255
        try:
            img = cv2.resize(heatmap_data.T, (960,540))
            img2 = img.astype('uint8')
            img2 = cv2.flip(img2, 0)
            # img2 = cv2.flip(img2, 2)
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

connect(False)
try:
    get_samples()
except:
    disconnect()
exit()
