#
# Program that plots the beam pattern for a single frequency in 3D of an array.
#
# The array is a microphone array with properties specified in config.py
#

import math
import matplotlib.pyplot as plt
import config_test as config
import numpy as np
import calc_r_prime as crp
import mpl_toolkits.mplot3d.axes3d as Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# import mpl_toolkits.mplot3d.axes3d as Axes3D
from matplotlib import cm, colors

def AF_3D(f):
    k = 2*math.pi*f/c
    P = 100
    theta = np.linspace(-90,90,P) * math.pi/180     # scan in theta direcion
    theta = theta.reshape(P,1,1)
    phi = np.linspace(0,360,P) * math.pi/180     
    phi = np.reshape(phi,(1,P,1))
    x_i = r_prime[0,:].reshape(1,1,len(r_prime[0,:]))
    y_i = r_prime[1,:].reshape(1,1,len(r_prime[1,:]))

    AF_mat = np.exp(1j*k*(np.sin(theta)*np.sin(phi) - math.sin(theta0)*np.sin(phi0))*y_i) \
            * np.exp(1j*k*(np.sin(theta)*np.cos(phi) - math.sin(theta0)*np.cos(phi0))*x_i)
    
    AF = np.square(np.absolute(np.sum(AF_mat,axis=2)/(len(r_prime[0,:]))))
    AF = 10*np.log10(AF) + 20
    AF = AF* (AF>0)/40
    theta = theta[:,:,0]
    phi = phi[:,:,0]
    x = AF*np.sin(theta)*np.cos(phi)
    y = AF*np.sin(theta)*np.sin(phi)
    z = AF * np.cos(theta)

    return x, y, z, AF


def interp_array(N1):  # add interpolated rows and columns to array
    N2 = np.empty([int(N1.shape[0]), int(2*N1.shape[1] - 1)])  # insert interpolated columns
    N2[:, 0] = N1[:, 0]  # original column
    for k in range(N1.shape[1] - 1):  # loop through columns
        N2[:, 2*k+1] = np.mean(N1[:, [k, k + 1]], axis=1)  # interpolated column
        N2[:, 2*k+2] = N1[:, k+1]  # original column
    N3 = np.empty([int(2*N2.shape[0]-1), int(N2.shape[1])])  # insert interpolated columns
    N3[0] = N2[0]  # original row
    for k in range(N2.shape[0] - 1):  # loop through rows
        N3[2*k+1] = np.mean(N2[[k, k + 1]], axis=0)  # interpolated row
        N3[2*k+2] = N2[k+1]  # original row
    return N3

c = config.PROPAGATION_SPEED
theta0 = 0 * math.pi/180    # scanning angle
phi0 = 0 * math.pi/180      # scanning angle
f = 4 * 10**3
r_prime , r_prime_ = crp.calc_r_prime(config.ELEMENT_DISTANCE)


X, Y, Z, AF = AF_3D(f)

interp_factor= 0    # interpolation factor of the beam data, for a nicer plot of the figure
for counter in range(interp_factor):  # Interpolate between points to increase number of faces in plot
    X = interp_array(X)
    Y = interp_array(Y)
    Z = interp_array(Z)

N = np.sqrt(X**2 + Y**2 + Z**2)
Rmax = np.max(N)
N = N/Rmax
# Find middle points between values for face colours
N = interp_array(N)[1::2,1::2]
mycol = cm.coolwarm(N)              # color map


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(1,1,1, projection='3d')
fig.subplots_adjust(top=1.1, bottom=-.1)              
ax.view_init(azim=300, elev=50)

y_min = min(np.min(Y), np.min(r_prime[1,:])) 
y_max = max(np.max(Y), np.max(r_prime[1,:]))
x_min =  3*y_min
x_max =  3*y_max
ax.set_box_aspect([3, 1, 3])
ax.set_xlim3d(x_min, x_max), ax.set_ylim3d(y_min, y_max)    # fig limits
cmap = plt.get_cmap('jet')

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=mycol, linewidth=0, antialiased=True, shade=False)

ax.axis('off')

### PLOT ARRAY
# plot array structure
xmin_mic = np.min(r_prime[0,:]) - config.ELEMENT_DISTANCE/2
xmax_mic = np.max(r_prime[0,:]) + config.ELEMENT_DISTANCE/2
ymin_mic = np.min(r_prime[1,:]) - config.ELEMENT_DISTANCE/2
ymax_mic = np.max(r_prime[1,:]) + config.ELEMENT_DISTANCE/2
z_pos = 0
x1 = [xmin_mic, xmin_mic, xmax_mic, xmax_mic]
y1 = [ymin_mic, ymax_mic, ymax_mic, ymin_mic]
z1 = [z_pos,z_pos,z_pos,z_pos]
vertices = [list(zip(x1,y1,z1))]
poly = Poly3DCollection(vertices, alpha=0.25, facecolor = 'darkseagreen') #'darkseagreen', 'darkolivegreen'
ax.add_collection3d(poly)

# Plot the array elements
element = 0
color_arr = ['r', 'b', 'g']#['purple', 'darkblue', 'darkgreen']
dx = config.ELEMENT_DISTANCE*0.1
dy = config.ELEMENT_DISTANCE*0.1
for array in range(config.ACTIVE_ARRAYS):
    for mic in range(config.ROWS*config.COLUMNS):
        mic += array*config.ROWS*config.COLUMNS
        x = r_prime[0,element]
        y = r_prime[1,element]
        ax.scatter(x, y, 0, color = color_arr[array], linewidths = 1, s=5)
        #ax.text(x-dx, y+dy, 0, str(element), None)
        element += 1

#filename = 'beam_A'+str(config.ACTIVE_ARRAYS)
#plt.savefig('plots/'+filename+'.png', dpi = 600, format = 'png', bbox_inches='tight')

plt.show()



