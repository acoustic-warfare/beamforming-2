import numpy as np
import matplotlib.pyplot as plt
import generate_signals
import config_other 
import math
import matplotlib.colors as cm
from interface import config

from lib.tests import pad_delay_wrapper, mimo_pad_wrapper, mimo_convolve_wrapper, mimo_lerp_wrapper
from lib.directions import calculate_coefficients, calculate_delays

whole_samples, adaptive_array = calculate_coefficients()
print('calculated coefficients')
samp_delay = calculate_delays()
print('calculated delays')

def generate_filename():
    if config_other.sources == 1:
        filename ='emul_'+ 'samples='+str(config_other.samples) + '_'+ str(config_other.f_start1)+'Hz_'+'theta='+str(config_other.theta_deg1)+'_phi='+str(config_other.phi_deg1)+ \
            '_E'+ str(config_other.rows*config_other.columns) + '_A'+str(config_other.active_arrays)
    elif config_other.sources == 2:
        filename ='emul_'+'samples='+str(config_other.samples) + '_'+str(config_other.f_start1)+str(config_other.f_start2)+'Hz_'+'theta='+str(config_other.theta_deg1)+str(config_other.theta_deg2)+ \
        '_phi='+str(config_other.phi_deg1)+str(config_other.phi_deg2)+'_E'+ str(config_other.rows*config_other.columns) + '_A'+str(config_other.active_arrays)
    return filename

#def generate_sig(frequency):
#    start_time = 0
#    end_time = 1
#    sample_rate = config.fs
#    time = np.arange(start_time, end_time, 1/sample_rate)
#    theta = 0
#    amplitude = 1
#    sinewave = amplitude * np.sin(2 * np.pi * frequency * time + theta)[:config.N_SAMPLES]
#    return sinewave
#
#
#def generate_signals_all(signal: np.ndarray):
#    signals = np.repeat(signal, config.N_MICROPHONES, axis=0).reshape((config.N_SAMPLES, config.N_MICROPHONES)).T
#    return np.float32(signals)

def validation_check(y_scan, x_scan):
    # Validation check
    xy_val_check = np.zeros((config.MAX_RES_X,config.MAX_RES_Y)) # matrix holding values of validation map

    theta_s = np.array([config_other.theta_deg1, config_other.theta_deg2])*math.pi/180
    phi_s = np.array([config_other.phi_deg1, config_other.phi_deg2])*math.pi/180
    r_scan = config.Z/np.cos(theta_s)

    for source_ind in range(config_other.sources):
            x_source = r_scan[source_ind] * math.sin(theta_s[source_ind]) * math.cos(phi_s[source_ind]) # conv the angles to x-coord of source
            y_source = r_scan[source_ind] * math.sin(theta_s[source_ind]) * math.sin(phi_s[source_ind]) # conv the angles to y-coord of source
            x_ind = (np.abs(x_scan[0,0,:,0] - x_source)).argmin()    # find the x-index of the x_scan coordinate that is neares the true x-coord of the soruce
            y_ind = (np.abs(y_scan[0,0,0,:] - y_source)).argmin()   # find the y-index of the y_scan coordinate that is neares the true y-coord of the soruce
            xy_val_check[x_ind,y_ind] = 1   # set value to 1 at the coord of the source (all other values are 0)
            #plt.text(x_ind, y_ind, 'source '+str(source_ind+1), fontsize = 16)

    # visualize the validation map
    fig, ax = plt.subplots()
    plt.title('Actual location of sources')
    pic = ax.pcolormesh(x_scan[0,0,:,0], y_scan[0,0,0,:], xy_val_check.T )
    fig.colorbar(pic)


#signal = generate_sig(8000)

#signals = generate_signals_all(signal)

filename = 'emulated_data/' + generate_filename()
try:
    signal = np.float32(np.load(filename+'.npy',allow_pickle=True))
    print('Loading from memory: ' + filename)
except:
    print('hej')
    generate_signals.main(generate_filename())
    signal = np.float32(np.load(filename+'.npy',allow_pickle=True))

signals = signal.T
print(signals.shape)

plt.figure()
plt.plot(signals[0,:])

save_fig = 1

### PAD
a = mimo_pad_wrapper(signals)
# plot heatmap
fig, ax = plt.subplots(figsize = (2*4,2*3))
pic = ax.pcolormesh(np.arange(config.MAX_RES_X), np.arange(config.MAX_RES_Y), a.T, cmap=plt.get_cmap('jet'))#,shading='gouraud') # heatmap summed over all frequencies
plt.axis('off')
# save heatmap as .png
if save_fig:
    path = '/home/mika/Desktop/ljudkriget/python/plots/rainbow_road/pad/'
    filename = 'pad_' + generate_filename()
    plt.savefig(path+filename+'.png', dpi = 300, format = 'png', bbox_inches='tight')

X, Y = np.meshgrid(np.arange(config.MAX_RES_X), np.arange(config.MAX_RES_Y))

norm = cm.Normalize(vmin=a.min().min(), vmax=a.max().max())
fig = plt.figure()#figsize=(8,6))
ax = fig.add_subplot(1,1,1, projection='3d')
fig.subplots_adjust(top=1.1, bottom=-.1)                           # aspect ratio
ax.view_init(azim=-45, elev=45)
ax.set_box_aspect([4, 3, 3])
surf = ax.plot_surface(X, Y, a.T, rstride=1, cstride=1, facecolors=plt.cm.jet(norm(a.T)), linewidth=0, antialiased=False, shade=False)
ax.axis('off')
if save_fig:
    path = '/home/mika/Desktop/ljudkriget/python/plots/rainbow_road/pad/'
    filename = 'pad_' + generate_filename() + '_3D'
    plt.savefig(path+filename+'.png', dpi = 300, format = 'png', bbox_inches='tight')

### INTERPOLATION
c = mimo_lerp_wrapper(signals)
fig, ax = plt.subplots(figsize = (2*4,2*3))
pic = ax.pcolormesh(np.arange(config.MAX_RES_X), np.arange(config.MAX_RES_Y), c.T, cmap=plt.get_cmap('jet'))#,shading='gouraud') # heatmap summed over all frequencies
plt.axis('off')
# save heatmap as .png
if save_fig:
    path = '/home/mika/Desktop/ljudkriget/python/plots/rainbow_road/interpolation/'
    filename = 'interp_' + generate_filename()
    plt.savefig(path+filename+'.png', dpi = 300, format = 'png', bbox_inches='tight')


norm = cm.Normalize(vmin=c.min().min(), vmax=c.max().max())
fig = plt.figure()#figsize=(8,6))
ax = fig.add_subplot(1,1,1, projection='3d')
fig.subplots_adjust(top=1.1, bottom=-.1)                           # aspect ratio
#ax.view_init(azim=300, elev=50)
ax.set_box_aspect([4, 3, 3])
ax.view_init(azim=-45, elev=45)
surf = ax.plot_surface(X, Y, c.T, rstride=1, cstride=1, facecolors=plt.cm.jet(norm(c.T)), linewidth=0, antialiased=False, shade=False)
ax.axis('off')
if save_fig:
    path = '/home/mika/Desktop/ljudkriget/python/plots/rainbow_road/interpolation/'
    filename = 'interp_' + generate_filename() + '_3D'
    plt.savefig(path+filename+'.png', dpi = 300, format = 'png', bbox_inches='tight', pad_inches=0)

# scanning window
theta_max = config.VIEW_ANGLE/2
x_scan_max = config.Z*np.tan(np.deg2rad(theta_max))
x_scan_min = -x_scan_max
y_scan_max = x_scan_max/config_other.aspect_ratio
y_scan_min = -y_scan_max

x_scan = np.linspace(x_scan_min,x_scan_max,config.MAX_RES_X)
y_scan = np.linspace(y_scan_min,y_scan_max,config.MAX_RES_Y)
x_scan = np.reshape(x_scan, (1,1,len(x_scan),1))    # reshape into 4D arrays
y_scan = np.reshape(y_scan, (1,1,1,len(y_scan)))    # reshape into 4D arrays

#validation_check(y_scan, x_scan)

plt.show()