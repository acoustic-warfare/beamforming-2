import math
import numpy as np
import matplotlib.pyplot as plt
from Matrix_array import Matrix_array

import config

def antenna_setup():
    r_a1 = config.r_a1      # coordinate position of origin of array1
    r_a2 = config.r_a2      # coordinate position of origin of array2
    r_a3 = config.r_a3      # coordinate position of origin of array3
    r_a4 = config.r_a4      # coordinate position of origin of array4fouri
    uni_distance = config.distance
    row_elements = config.rows
    column_elements = config.columns

    # array_matrix_1, array_matrix_2, array_matrix_3, array_matrix_4 below can be generated in parallell
    array_matrix_1 = Matrix_array(r_a1,uni_distance,row_elements,column_elements)
    array_matrix_2 = Matrix_array(r_a2,uni_distance,row_elements,column_elements)
    array_matrix_3 = Matrix_array(r_a3,uni_distance,row_elements,column_elements)
    array_matrix_4 = Matrix_array(r_a4,uni_distance,row_elements,column_elements)
    array_list = [array_matrix_1, array_matrix_2, array_matrix_3, array_matrix_4]

    # array_matrices contains the current active arrays that should be used 
    #  (number of arrays defined by config.matrices)
    array_matrices = np.array(array_list[:config.active_arrays], dtype=object)
    
    # array_matrices = np.array([array_matrix_1, array_matrix_2, array_matrix_3, array_matrix_4], dtype=object)

    sub_arrays = len(array_matrices)

    
    return array_matrices


def calc_r_prime(d):
    half = d/2
    r_prime = np.zeros((2, config.elements))
    element_index = 0
    for array in range(config.active_arrays):
        for row in range(config.rows):
            for col in range(config.columns):
                r_prime[0,element_index] = col * d + half + array*config.columns*d + array*config.d - config.columns* config.active_arrays * half
                r_prime[1, element_index] = row * d - config.rows * half + half
                element_index += 1
    r_prime[0,:] -= config.active_arrays*config.d/2

    if config.plot_setup:
        plt.figure()#figsize=(config.columns*config.active_arrays, config.rows))
        ax = plt.gca() #you first need to get the axis handle
        ax.set_aspect(16/9) #sets the height to width ratio to 1.5.
        element = 0
        color_arr = ['r', 'b', 'g','m']
        dx = 0
        dy = 0
        for array in range(config.active_arrays):
            plt.title('Array setup')
            for mic in range(config.rows*config.columns):
                x = r_prime[0,element]
                y = r_prime[1,element]
                plt.scatter(x, y, color=color_arr[array])
                plt.text(x-dx, y+dy, str(element+1))
                element += 1
    return r_prime

def weight_index(f):
    # calculates what mode to use, depending on the wavelength of the signal
    d = config.distance              # distance between elements
    wavelength_rel = f*d/c    # relative wavelength to distance between microphone elements
    #print('f: ' + str(f))
    if wavelength_rel>0.1581:
        mode = 1
    elif (wavelength_rel <= 0.1581) and (wavelength_rel > 0.156):
        mode = 2
    elif (wavelength_rel <= 0.156) and (wavelength_rel > 0.0986):
        mode = 3
    elif (wavelength_rel <= 0.0986) and (wavelength_rel > 0.085):
        mode = 5
    elif (wavelength_rel <= 0.085) and (wavelength_rel > 0.07):
        mode = 6
    else:
        mode = 7
    return mode

def adaptive_matrix(rows, columns):
    # Creates the weight matrix
    try:
            # array_audio_signals = np.load(filename)
            weight_matrix = np.load('adaptive_matrix'+'.npy', allow_pickle=True)
            #print("Loading from Memory: " + filename)
    except:
        weight_matrix = np.zeros((7, rows*columns))
        for mode in range(1,7+1):
            weight = np.zeros((1,rows*columns))
            row_lim = math.ceil(rows/mode)
            column_lim = math.ceil(columns/mode)
            for i in range(row_lim):
                for j in range(column_lim):
                    element_index = (mode*i*rows + mode*j) # this calculation could be wrong thanks to matlab and python index :))
                    weight[0,element_index] = 1
            weight_matrix[mode-1,:] = weight
        np.save('adaptive_matrix', weight_matrix)
    return weight_matrix


    

c = 343         # speed of sound

N = config.active_arrays * config.columns
M = config.rows

F = 501                     # number of points in frequency vector
P = 501                     # nomber of points in angle vectors
d = config.distance      # distance between elements
f = config.single_freq   # frequency if only a single frequency should be evaluated
fs = 48800             # sampling frequency

f_vec = np.linspace(100, 8* 10**3, F)
f_max = fs/2
lambda_max = c/f_max
d_min = lambda_max/2
k_vec = 2*math.pi*f_vec/c

wavelength = c/f
k = 2*math.pi*f/c


# --- Test things
#d = wavelength/2
print('distance between elements: ' + str(d*10**2) + ' cm')


#array_matrices = antenna_setup()
adaptive_weight_matrix = adaptive_matrix(config.rows, config.columns)


#r_prime = array_matrices[0].get_r_prime()
for array in range(config.active_arrays)[1:]:
    #r_prime = np.hstack((r_prime, array_matrices[array].get_r_prime()))
    adaptive_weight_matrix = np.hstack((adaptive_weight_matrix, adaptive_weight_matrix))
#xy_coord = np.dstack((x_coord,y_coord))

# NOTE, calc_r_prime GER INTE SAMMA MIC INDEX SOM VI FÅR DATA FRÅN SEN???
r_prime = calc_r_prime(config.distance)


theta = np.array([np.linspace(-90,90,P) * math.pi/180]).T
phi = np.array([1*10**(-14),90])

theta0 = 0 * math.pi/180
phi0 = 0 * math.pi/180
psi = k*d*np.sin(theta)*np.sin(phi)
psi0 = k*d*np.sin(theta0) *np.sin(phi0)
xi = k*d*np.sin(theta)*np.cos(phi)
xi0 = k*d*np.sin(theta0)*np.cos(phi0)


# --- 3D PLOT, MED ADAPTIVE ---
AF3ad_3D = np.zeros((P,len(phi),len(f_vec)))
for k_ind in range(len(f_vec)):
    AF3ad_2D = np.zeros((len(theta),len(phi)), dtype='complex128')
    f = f_vec[k_ind]
    k = 2*math.pi*f/c
    wavelength_rel = f*d/c    # relative wavelength to distance between microphone elements
    for mic in range(len(r_prime[0,:])):
        if adaptive_weight_matrix[weight_index(f)-1,mic]:
            AF3ad_2D += np.exp(1j*k*(np.sin(theta)*np.sin(phi) - math.sin(theta0)*np.sin(phi0))*r_prime[0,mic]) \
                * np.exp(1j*k*(np.sin(theta)*np.cos(phi) - math.sin(theta0)*np.cos(phi0))*r_prime[1,mic])
    AF3ad_2D = np.square(np.absolute(AF3ad_2D))
    AF3ad_3D[:,:,k_ind] = AF3ad_2D
AF3ad_3D_dBi = 10*np.log10(AF3ad_3D)


# --- 3D PLOT, MED ADAPTIVE ---
# --- TEST med normalisering med antalet element ---
# Bra om man kan få till matrismultiplikation istället för for-loopen för alla mikrofoner
AF3ad2_3D = np.zeros((P,len(phi),len(f_vec)))
for k_ind in range(len(f_vec)):
    AF3ad2_2D = np.zeros((len(theta),len(phi)), dtype='complex128')
    f = f_vec[k_ind]
    k = 2*math.pi*f/c
    for mic in range(len(r_prime[0,:])):
        if adaptive_weight_matrix[weight_index(f)-1,mic]:
            AF3ad2_2D += np.exp(1j*k*(np.sin(theta)*np.sin(phi) - math.sin(theta0)*np.sin(phi0))*r_prime[0,mic]) \
                * np.exp(1j*k*(np.sin(theta)*np.cos(phi) - math.sin(theta0)*np.cos(phi0))*r_prime[1,mic])
    norm = np.sum(adaptive_weight_matrix[weight_index(f)-1,:])
    AF3ad2_2D = np.square(np.absolute(AF3ad2_2D/norm))
    AF3ad2_3D[:,:,k_ind] = AF3ad2_2D
AF3ad2_3D_dBi = 10*np.log10(AF3ad2_3D)



# --- 3D PLOT, UTAN ADAPTIVE ---
# Bra om man kan få till matrismultiplikation istället för for-loopen för alla mikrofoner
AF3_3D = np.zeros((P,len(phi),len(f_vec)))
for k_ind in range(len(f_vec)):
    AF3_2D = np.zeros((len(theta),len(phi)), dtype='complex128')
    f = f_vec[k_ind]
    k = 2*math.pi*f/c
    for mic in range(len(r_prime[0,:])):
        AF3_2D += np.exp(1j*k*(np.sin(theta)*np.sin(phi) - math.sin(theta0)*np.sin(phi0))*r_prime[0,mic]) \
            * np.exp(1j*k*(np.sin(theta)*np.cos(phi) - math.sin(theta0)*np.cos(phi0))*r_prime[1,mic])
    #AF3_2D = np.square(np.absolute(AF3_2D/config.elements))
    AF3_2D = np.square(np.absolute(AF3_2D))
    np.max(AF3_2D)
    AF3_3D[:,:,k_ind] = AF3_2D
AF3_3D_dBi = 10*np.log10(AF3_3D)


# --- PLOTS ---
if config.plot_single_f:
    # plot |AF|²
    f1 = np.argmax(f_vec>100)
    f2 = np.argmax(f_vec>200)
    f3 = np.argmax(f_vec>500)
    f4 = np.argmax(f_vec>4000)
    f5 = np.argmax(f_vec > 20000)
    f6 = np.argmax(f_vec > 10000)
    freqs = f_vec #[f1, f2, f3, f4, f5, f6]
    plt.figure()
    for fig in range(len(freqs)):
        freq = freqs[fig]
    #plt.plot(theta*180/math.pi, AF3_dBi, 'g',linewidth=2, label='sum exp 2')
    #plt.plot(theta*180/math.pi, AF_dBi, 'r',linewidth=2, label='sin')
        #plt.plot(theta*180/math.pi, AF3_3D[:,0,freq], linewidth=2, label=str(f_vec[freq]))
        plt.plot(theta*180/math.pi, AF3_3D_dBi[:,0,fig], linewidth=2, label=str(freq))
    plt.xlabel('Theta (deg)')
    plt.ylabel('|AF|²')
    plt.title('Array factor')
    #plt.ylim(np.max(AF3ad_3D_dBi[:,0,freq])-40, np.max(AF3ad_3D_dBi[:,0,freq])+3)
    plt.grid(True)
    #plt.legend(('sin','sum exp'))
    plt.legend()


if config.active_arrays == 1:
    if config.plot_contourf:
        X, Y = np.meshgrid(theta*180/math.pi, f_vec*10**(-3))
        plt.figure()
        levels = np.linspace(np.max(AF3ad_3D_dBi[:,0,:])-20, np.max(AF3ad_3D_dBi[:,0,:]), 50)
        plt.contourf(X, Y, np.transpose(AF3ad_3D_dBi[:,0,:]), levels, cmap=plt.get_cmap('coolwarm'), extend='min' )
        plt.colorbar(label='|AF|² (dBi)', ticks = None)
        plt.ylabel('Frequency (kHz)')
        plt.xlabel('Theta (deg)')
        plt.title('phi = 0 (deg), adaptive')
        filename = 'AF_'+str(config.active_arrays) + '_array_d=' + str(config.distance*10**(2)) + '_w=' + str(config.d)
        #plt.savefig('plots/'+filename+'.png', dpi = 500, format = 'png')

if config.plot_contourf:
    X, Y = np.meshgrid(theta*180/math.pi, f_vec*10**(-3))
    plt.figure()
    levels = np.linspace(np.max(AF3ad_3D_dBi[:,1,:])-20, np.max(AF3ad_3D_dBi[:,1,:]), 50)
    plt.contourf(X, Y, np.transpose(AF3ad_3D_dBi[:,1,:]),
     levels, cmap=plt.get_cmap('coolwarm'), extend='min' )
    plt.colorbar(label='|AF|² (dBi)', ticks = None)
    plt.ylabel('Frequency (kHz)')
    plt.xlabel('Theta (deg)')
    plt.title('phi = 90 (deg), adaptive')
    filename = 'AF_'+str(config.active_arrays) + '_array_d=' + str(config.distance*10**(2)) + '_w=' + str(config.d) 
    #plt.savefig('plots/'+filename+'.png', dpi = 500, format = 'png')

if config.plot_contourf:
    X, Y = np.meshgrid(theta*180/math.pi, f_vec*10**(-3))
    plt.figure()
    levels = np.linspace(np.max(AF3ad2_3D_dBi[:,0,:])-20, np.max(AF3ad2_3D_dBi[:,0,:]), 50)
    plt.contourf(X, Y, np.transpose(AF3ad2_3D_dBi[:,0,:]), levels, cmap=plt.get_cmap('coolwarm'), extend='min' )
    plt.colorbar(label='|AF|² (dBi)', ticks = None)
    plt.ylabel('Frequency (kHz)')
    plt.xlabel('Theta (deg)')
    plt.title('phi = 0 (deg), adaptive')
    filename = 'AF_'+str(config.active_arrays) + '_array_d=' + str(config.distance*10**(2)) + '_w=' + str(config.d) 
    #plt.savefig('plots/'+filename+'.png', dpi = 500, format = 'png')

if config.plot_contourf:
    X, Y = np.meshgrid(theta*180/math.pi, f_vec*10**(-3))
    plt.figure()
    levels = np.linspace(np.max(AF3ad2_3D_dBi[:,1,:])-20, np.max(AF3ad2_3D_dBi[:,1,:]), 50)
    plt.contourf(X, Y, np.transpose(AF3ad2_3D_dBi[:,1,:]), levels, cmap=plt.get_cmap('coolwarm'), extend='min' )
    plt.colorbar(label='|AF|² (dBi)', ticks = None)
    plt.ylabel('Frequency (kHz)')
    plt.xlabel('Theta (deg)')
    plt.title('phi = 90 (deg), adaptive')
    filename = 'AF_'+str(config.active_arrays) + '_array_d=' + str(config.distance*10**(2)) + '_w=' + str(config.d) 
    #plt.savefig('plots/'+filename+'.png', dpi = 500, format = 'png')

if config.plot_contourf:
    X, Y = np.meshgrid(theta*180/math.pi, f_vec*10**(-3))
    plt.figure()
    levels = np.linspace(np.max(AF3_3D_dBi[:,0,:])-20, np.max(AF3_3D_dBi[:,0,:]), 50)
    plt.contourf(X, Y, np.transpose(AF3_3D_dBi[:,0,:]), levels, cmap=plt.get_cmap('coolwarm'), extend='min' )
    plt.colorbar(label='|AF|² (dBi)', ticks = None)
    plt.ylabel('Frequency (kHz)')
    plt.xlabel('Theta (deg)')
    plt.title('phi = 0 (deg), not adaptive')
    filename = 'AF_'+str(config.active_arrays) + '_array_d=' + str(config.distance*10**(2)) + '_not_adaptive'
    #plt.savefig('plots/'+filename+'.png', dpi = 500, format = 'png')

if config.plot_contourf:
    X, Y = np.meshgrid(theta*180/math.pi, f_vec*10**(-3))
    plt.figure()
    levels = np.linspace(np.max(AF3_3D_dBi[:,1,:])-20, np.max(AF3_3D_dBi[:,1,:]), 50)
    plt.contourf(X, Y, np.transpose(AF3_3D_dBi[:,1,:]), levels, cmap=plt.get_cmap('coolwarm'), extend='min' )
    plt.colorbar(label='|AF|² (dBi)', ticks = None)
    plt.ylabel('Frequency (kHz)')
    plt.xlabel('Theta (deg)')
    #plt.title('phi = 90 (deg), not adaptive')
    filename = 'AF_'+str(config.active_arrays) + '_array_d=' + str(config.distance*10**(2)) + '_not_adaptive'
    #plt.savefig('plots/'+filename+'.png', dpi = 500, format = 'png')


plt.show()