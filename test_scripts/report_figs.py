import time_delay
import config_test as config
import matplotlib.pyplot as plt
import numpy as np
import active_microphones as am
import calc_r_prime as crp
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math

plt.rcParams["font.family"] = "monospace"

#def calc_r_prime(d):
#    half = d/2      # half distance between microphones
#    r_prime = np.zeros((2, config.ELEMENT_DISTANCE))
#    element_index = 0
#
#    # give each microphone the correct coordinate
#    for array in range(config.ACTIVE_ARRAYS):
#        array *= -1
#        for row in range(config.ROWS):
#            for col in range(config.COLUMNS):
#                r_prime[0,element_index] = - col * d - half + array*config.COLUMNS*d + array*config.ARRAY_SEPARATION + config.COLUMNS* config.ACTIVE_ARRAYS * half
#                r_prime[1, element_index] = row * d - config.ROWS * half + half
#                element_index += 1
#    r_prime[0,:] += (config.ACTIVE_ARRAYS-1)*config.ARRAY_SEPARATION/2
#    active_mics = am.active_microphones()
#
#    r_prime_all = r_prime
#    r_prime = r_prime[:,active_mics]
#    return r_prime, r_prime_all, active_mics


# plot options
plot_scanning_window = 1
plot_scanning_window2 = 1

x_scan = time_delay.x_scan[:,0,0]
y_scan = time_delay.y_scan[0,:,0]
z_scan = time_delay.z_scan

x_min = time_delay.x_scan_min
x_max = time_delay.x_scan_max
y_min = time_delay.y_scan_min
y_max = time_delay.y_scan_max

alpha = config.VIEW_ANGLE
r_prime_all, r_prime = crp.calc_r_prime(config.ELEMENT_DISTANCE)
active_mics = am.active_microphones(config.mode, config.mics_used)
# Create meshgrid
X,Y = np.meshgrid(x_scan.T, y_scan)



if plot_scanning_window:
    # ----- SCANNING WINDOW AND ARRAY ------
    import beam_forming_algorithm
    imag = beam_forming_algorithm.FFT_power_summed_freq     # get image of
    fig = plt.figure()
    ax = plt.axes(projection='3d')                          # initate 3D plot
    fig.subplots_adjust(top=1.1, bottom=-.1)
    ax.set_proj_type('persp', focal_length=0.5)             # change the focal lenght to get perspective in the figure
    ax.set_box_aspect([8, 6, 6])                            # aspect ratio
    ax.set_xlim(x_min, x_max), ax.set_ylim(y_min, y_max)    # fig limits
    #ax.view_init(-25, 0, 90)                                # rotate the figure so we get the right angle of view
    ax.view_init(-25, 20, 80)                                # rotate the figure so we get the right angle of view
    plt.tight_layout()
    # Get rid of ticks
    ax.set_xticks([])                               
    ax.set_yticks([])                               
    ax.set_zticks([])

    # Get rid of grid
    ax.grid(False)

    # Get rid of the spines                         
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    cmap = plt.get_cmap('jet')
    norm = matplotlib.colors.Normalize(vmin=imag.min(), vmax=imag.max() )
    cmap = cmap(norm(imag.T))
    cmap2 = matplotlib.colors.ListedColormap(["lawngreen"])

    # Plot the scanning window with heatmap picture
    ax.plot_wireframe(X,Y, np.ones((len(x_scan),len(y_scan))).T*z_scan, color='white', linewidth = 0.75)
    ax.plot_surface(X,Y,np.ones_like(imag.T)*z_scan*1.01, cstride=1, rstride=1, facecolors=cmap, shade=False)

    # plot array structure
    xmin_mic = np.min(r_prime_all[0,:]) - config.ELEMENT_DISTANCE/2
    xmax_mic = np.max(r_prime_all[0,:]) + config.ELEMENT_DISTANCE/2
    ymin_mic = np.min(r_prime_all[1,:]) - config.ELEMENT_DISTANCE/2
    ymax_mic = np.max(r_prime_all[1,:]) + config.ELEMENT_DISTANCE/2
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
            x = r_prime_all[0,element]
            y = r_prime_all[1,element]
            if mic in active_mics:
                ax.scatter(x, y, 0, color = color_arr[array], s=5)
            else:
                ax.scatter(x, y, 0, color = 'none', edgecolor=color_arr[array], linewidths = 1, s=5)
            #ax.text(x-dx, y+dy, 0, str(element), None)
            element += 1

    #filename = 'scanning_window'
    #plt.savefig('plots/'+filename+'.png', dpi = 800, format = 'png')

# ----- END OF PLOT SCANNING WINDOW AND ARRAY ------

if plot_scanning_window2:
    fig = plt.figure()
    ax = plt.axes(projection='3d')                          # initate 3D plot
    fig.subplots_adjust(top=1.1, bottom=-.1)                
    plt.tight_layout()
    fig.subplots_adjust(top=1.1, bottom=-.1)
    ax.set_proj_type('persp', focal_length=0.5)             # change the focal lenght to get perspective in the figure
    ax.view_init(-25, 20, 80)                                # rotate the figure so we get the right angle of view
    ax.set_box_aspect([8, 6, 6])                            # aspect ratio
    ax.axis('off')

    # Plot the scanning window
    ax.plot_wireframe(X,Y, np.ones((len(x_scan),len(y_scan))).T*z_scan, color='grey', linewidth = 0.75)

    LW = 1
    FS = 15
    origo = [0,0,0]

    #plot axis
    x_axis = [x_max*0.7, 0, 0]
    y_axis = [0,y_max*0.7, 0]
    z_axis = [0,0,z_scan*0.5]
    ax.quiver(origo,origo,origo,x_axis,y_axis,z_axis,arrow_length_ratio=0.1, color = 'grey', linewidth = LW)
    ax.text(x_axis[0]*1.2,0,0, 'x', color ='grey',fontsize = FS)
    ax.text(0,y_axis[1],0, 'y', color ='grey',fontsize = FS)
    ax.text(0,0,z_axis[2], 'z', color ='grey',fontsize = FS)

    # plot lines
    point2 = [x_min,0,z_scan]
    point3 = [x_max,0,z_scan]
    ax.plot([origo[0],point2[0]],[origo[1],point2[1]],[origo[2],point2[2]], '--k', linewidth = LW)
    ax.plot([origo[0],point3[0]],[origo[1],point3[1]],[origo[2],point3[2]], '--k', linewidth = LW)

    # plot arch
    r_arc = config.Z/7
    theta_arc = np.linspace(-config.VIEW_ANGLE/2, config.VIEW_ANGLE/2, 20)/180*math.pi
    x_arc = r_arc*np.sin(theta_arc)
    y_arc =np.zeros_like(theta_arc)
    z_arc = r_arc*np.cos(theta_arc)
    ax.plot(x_arc, y_arc, z_arc, color ='r', linewidth = LW)
    ax.text(0, 0, r_arc*1.5, chr(945), color ='r', fontsize = FS)
    
    #filename = 'scanning_window2'
    #plt.savefig('plots/'+filename+'.png', dpi = 800, format = 'png', bbox_inches='tight')

plt.show()