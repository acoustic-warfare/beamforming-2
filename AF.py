import math
import numpy as np
import matplotlib.pyplot as plt
from Matrix_array import Matrix_array

import config_AF

def antenna_setup():
    r_a1 = config_AF.r_a1      # coordinate position of origin of array1
    r_a2 = config_AF.r_a2      # coordinate position of origin of array2
    r_a3 = config_AF.r_a3      # coordinate position of origin of array3
    r_a4 = config_AF.r_a4      # coordinate position of origin of array4fouri
    uni_distance = config_AF.distance
    row_elements = config_AF.rows
    column_elements = config_AF.columns

    # array_matrix_1, array_matrix_2, array_matrix_3, array_matrix_4 below can be generated in parallell
    array_matrix_1 = Matrix_array(r_a1,uni_distance,row_elements,column_elements)
    array_matrix_2 = Matrix_array(r_a2,uni_distance,row_elements,column_elements)
    array_matrix_3 = Matrix_array(r_a3,uni_distance,row_elements,column_elements)
    array_matrix_4 = Matrix_array(r_a4,uni_distance,row_elements,column_elements)
    array_list = [array_matrix_1, array_matrix_2, array_matrix_3, array_matrix_4]

    # array_matrices contains the current active arrays that should be used 
    #  (number of arrays defined by config.matrices)
    array_matrices = np.array(array_list[:config_AF.active_arrays], dtype=object)
    
    # array_matrices = np.array([array_matrix_1, array_matrix_2, array_matrix_3, array_matrix_4], dtype=object)

    sub_arrays = len(array_matrices)


    if config_AF.plot_setup:
        for array in range(sub_arrays):
            plt.title('Array setup')
            plt.scatter(array_matrices[array].get_r_prime()[0,:], array_matrices[array].get_r_prime()[1,:])
    
    return array_matrices

def weight_index(f):
    # calculates what mode to use, depending on the wavelength of the signal
    d = config_AF.distance              # distance between elements
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

N = config_AF.active_arrays * config_AF.columns
M = config_AF.rows
n = np.array([np.linspace(0,N-1,N)])
#print('n: ' + str(n))
#print('M' + str(M))
#print('N' + str(N))

F = 101                     # number of points in frequency vector
P = 501                     # nomber of points in angle vectors
d = config_AF.distance      # distance between elements
f = config_AF.single_freq   # frequency if only a single frequency should be evaluated
fs = 16 * 10**3             # sampling frequency

f_vec = np.linspace(100, 8* 10**3, F)
print(f_vec)
f_max = fs/2
lambda_max = c/f_max
d_min = lambda_max/2
k_vec = 2*math.pi*f_vec/c

wavelength = c/f
k = 2*math.pi*f/c


# --- Test things
#d = wavelength/2
ratio = 0.02/d
#d = d*ratio
print('distance between elements: ' + str(d*10**2) + ' cm')
#print('ratio: ' + str(0.02/d))


array_matrices = antenna_setup()
adaptive_weight_matrix = adaptive_matrix(config_AF.rows, config_AF.columns)
print('Mode 1: ' + str(adaptive_weight_matrix[0,:]))
print('Mode 2: ' + str(adaptive_weight_matrix[1,:]))
print('Mode 3: ' + str(adaptive_weight_matrix[2,:]))
print('Mode 4: ' + str(adaptive_weight_matrix[3,:]))
print('Mode 5: ' + str(adaptive_weight_matrix[4,:]))
print('Mode 6: ' + str(adaptive_weight_matrix[5,:]))
print('Mode 7: ' + str(adaptive_weight_matrix[6,:]))

# --- Only needed if using third draft of AF2
x_coord = array_matrices[0].get_x_coord()
y_coord = array_matrices[0].get_y_coord()
r_prime = array_matrices[0].get_r_prime()
for array in range(config_AF.active_arrays)[1:]:
    x_coord = np.hstack((x_coord, array_matrices[array].get_x_coord()))
    y_coord = np.hstack((y_coord, array_matrices[array].get_y_coord()))
    r_prime = np.hstack((r_prime, array_matrices[array].get_r_prime()))
    adaptive_weight_matrix = np.hstack((adaptive_weight_matrix, adaptive_weight_matrix))
xy_coord = np.dstack((x_coord,y_coord))
#print(np.shape(x_coord))
#print(np.shape(y_coord))
#print(x_coord)
#print(y_coord)
#print('weight matrix: ' + str(np.shape(adaptive_weight_matrix)))
#print(adaptive_weight_matrix)


theta = np.array([np.linspace(-90,90,P) * math.pi/180]).T
phi = np.array([1*10**(-14),90])
#phi = np.linspace(0,90,2)* math.pi/180

theta0 = 0 * math.pi/180
phi0 = 0 * math.pi/180
psi = k*d*np.sin(theta)*np.sin(phi)
psi0 = k*d*np.sin(theta0) *np.sin(phi0)
xi = k*d*np.sin(theta)*np.cos(phi)
xi0 = k*d*np.sin(theta0)*np.cos(phi0)


AF_x = np.sin(N/2*(psi-psi0)) / (np.sin(1/2*(psi-psi0)))
AF_y = np.sin(M/2*(xi-xi0)) / (np.sin(1/2*(xi-xi0)))
AF = AF_y * AF_y
AF = np.square(np.absolute(AF))
AF_dBi = 10*np.log10(AF)
#print('AF_X: ' + str(np.shape(AF_x)))
#print('AF_Y: ' + str(np.shape(AF_y)))
#print(np.shape(AF))

#AF2 = np.exp(1j*np.matmul((k*d*(np.sin(theta) - math.sin(theta0))),n)) # First draft

# --- THIRD DRAFT ---
#AF2_X = np.zeros((len(theta),len(phi)), dtype='complex128')
#AF2_Y = np.zeros((len(theta),len(phi)), dtype='complex128')
#print('AF2_X: ' + str(np.shape(AF2_X)))
#print('AF2_Y: ' + str(np.shape(AF2_Y)))
#for x_ind in range(np.shape(x_coord)[1]):
#    AF2_X += np.exp(1j*k*(np.sin(theta)*np.sin(phi) - math.sin(theta0)*np.sin(phi0))*x_coord[0,x_ind]) # third draft, can handle several arrays defined in config_AF.py file
#    #AF2_X += np.exp(1j*(np.sin(theta) - math.sin(theta0))*(k*xy_coord[0,x_coord,0])) # second draft, can handle several arrays defined in config_AF.py file
#print(np.shape(y_coord)[0])
#for y_ind in range(np.shape(y_coord)[0]):
#    AF2_Y += np.exp(1j*k*(np.sin(theta)*np.cos(phi) - math.sin(theta0)*np.cos(phi0))*y_coord[y_ind,0])
#AF2 = AF2_X*AF2_Y
#print(np.shape(AF2))
#AF2 = np.square(np.absolute(AF2))
#AF2_dBi = 10*np.log10(AF2)

# --- FOURTH DRAFT ---
AF3 = np.zeros((len(theta),len(phi)), dtype='complex128')
for mic in range(len(r_prime[0,:])):
    AF3 += np.exp(1j*k*(np.sin(theta)*np.sin(phi) - math.sin(theta0)*np.sin(phi0))*r_prime[0,mic]) \
        * np.exp(1j*k*(np.sin(theta)*np.cos(phi) - math.sin(theta0)*np.cos(phi0))*r_prime[1,mic])
AF3 = np.square(np.absolute(AF3))
AF3_dBi = 10*np.log10(AF3)


# Calculate |AF|² for the arrays defined in config_AF.py, for a range of frequencies
# only a linear array with 1xN elements
AF_3D = np.zeros((P,F))
for k_ind in range(F):
    f = f_vec[k_ind]
    k = 2*math.pi*f/c
    # AF_2D = np.exp(1j*np.matmul((k*d*(np.sin(theta) - math.sin(theta0))),n)) # first draft
    AF_2D = np.exp(1j*k*(np.sin(theta) - math.sin(theta0))*(xy_coord[0,:,0])) # second draft, can handle several arrays defined in config_AF.py file
    #print(np.shape(AF_2D))
    #print(np.shape(adaptive_weight_matrix))
    AF_2D = np.sum(AF_2D, axis=1) #, keepdims=True)
    AF_2D = np.square(np.absolute(AF_2D))
    AF_3D[:,k_ind] = AF_2D
    AF_3D_dBi = 10*np.log10(AF_3D)

if config_AF.active_arrays == 1:
    AFad_3D = np.zeros((P,F))
    for k_ind in range(F):
        f = f_vec[k_ind]
        k = 2*math.pi*f/c
        # AF_2D = np.exp(1j*np.matmul((k*d*(np.sin(theta) - math.sin(theta0))),n)) # first draft
        AFad_2D = np.exp(1j*k*(np.sin(theta) - math.sin(theta0))*(xy_coord[0,:,0])) # second draft, can handle several arrays defined in config_AF.py file
        #print(np.shape(AF_2D))
        #print(np.shape(adaptive_weight_matrix))
        AFad_2D *= adaptive_weight_matrix[weight_index(f)-1,0:config_AF.columns]
        AFad_2D = np.sum(AFad_2D, axis=1) #, keepdims=True)
        AFad_2D = np.square(np.absolute(AFad_2D))
        AFad_3D[:,k_ind] = AFad_2D
    AFad_3D_dBi = 10*np.log10(AFad_3D)

# --- 3D PLOT, MED ADAPTIVE ---
# Bra om man kan få till matrismultiplikation istället för for-loopen för alla mikrofoner
AF3ad_3D = np.zeros((P,len(phi),F))
for k_ind in range(F):
    AF3ad_2D = np.zeros((len(theta),len(phi)), dtype='complex128')
    f = f_vec[k_ind]
    k = 2*math.pi*f/c
    print('f: ' + str(f) + '. mics: ' + str(np.sum(adaptive_weight_matrix[weight_index(f)-1,:])))
    wavelength_rel = f*d/c    # relative wavelength to distance between microphone elements
    print('wavelength_rel: ' + str(wavelength_rel) + ', mode:' + str(weight_index(f)))
    #print(adaptive_weight_matrix[weight_index(f)-1,:])
    for mic in range(len(r_prime[0,:])):
        if adaptive_weight_matrix[weight_index(f)-1,mic]:
            AF3ad_2D += np.exp(1j*k*(np.sin(theta)*np.sin(phi) - math.sin(theta0)*np.sin(phi0))*r_prime[0,mic]) \
                * np.exp(1j*k*(np.sin(theta)*np.cos(phi) - math.sin(theta0)*np.cos(phi0))*r_prime[1,mic])
    AF3ad_2D = np.square(np.absolute(AF3ad_2D))
    AF3ad_3D[:,:,k_ind] = AF3ad_2D
AF3ad_3D_dBi = 10*np.log10(AF3ad_3D)
print(np.shape(AF3ad_3D_dBi))

# --- 3D PLOT, MED ADAPTIVE ---
# --- TEST med normalisering med antalet element ---
# Bra om man kan få till matrismultiplikation istället för for-loopen för alla mikrofoner
AF3ad2_3D = np.zeros((P,len(phi),F))
for k_ind in range(F):
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
AF3_3D = np.zeros((P,len(phi),F))
for k_ind in range(F):
    AF3_2D = np.zeros((len(theta),len(phi)), dtype='complex128')
    f = f_vec[k_ind]
    k = 2*math.pi*f/c
    for mic in range(len(r_prime[0,:])):
        AF3_2D += np.exp(1j*k*(np.sin(theta)*np.sin(phi) - math.sin(theta0)*np.sin(phi0))*r_prime[0,mic]) \
            * np.exp(1j*k*(np.sin(theta)*np.cos(phi) - math.sin(theta0)*np.cos(phi0))*r_prime[1,mic])
    AF3_2D = np.square(np.absolute(AF3_2D))
    np.max(AF3_2D)
    AF3_3D[:,:,k_ind] = AF3_2D
AF3_3D_dBi = 10*np.log10(AF3_3D)

## GAIN
AF_gain = AF_3D_dBi.max(axis=0)
AF3ad_gain = AF3ad_3D_dBi.max(axis=0)
AF3_gain = AF3_3D_dBi.max(axis=0)

plt.figure()
plt.plot(f_vec, AF_gain)

'''
# plot |AF|²
fig = plt.figure()
# Creating color map
ax = plt.axes(projection ='3d')
surf = ax.plot_surface(theta*180/math.pi, f_vec, AF_3D_dBi, cmap = plt.get_cmap('coolwarm'),
                       edgecolor ='none',linewidth=0, antialiased=False)
fig.colorbar(surf, ax = ax,
             shrink = 0.5, aspect = 5)
plt.title('Array factor')
'''

# --- PLOTS ---
if config_AF.plot_single_f:
    # plot |AF|²
    freq1 = np.argmax(f_vec>2*10**3)
    freq2 = np.argmax(f_vec>4*10**3)
    freq3 = np.argmax(f_vec>6*10**3)
    plt.figure()
    #plt.plot(theta*180/math.pi, AF3_dBi, 'g',linewidth=2, label='sum exp 2')
    #plt.plot(theta*180/math.pi, AF_dBi, 'r',linewidth=2, label='sin')
    plt.plot(theta*180/math.pi, AF3_3D[:,0,freq1], linewidth=2, label=str(f_vec[freq1]*10*(-3)))
    plt.plot(theta*180/math.pi, AF3_3D[:,0,freq2], linewidth=2, label=str(f_vec[freq2]*10*(-3)))
    plt.plot(theta*180/math.pi, AF3_3D[:,0,freq3], linewidth=2, label=str(f_vec[freq3]*10*(-3)))
    plt.xlabel('Theta (deg)')
    plt.ylabel('|AF|²')
    plt.title('Array factor')
    #plt.ylim(np.max(AF3ad_3D_dBi[:,0,freq])-40, np.max(AF3ad_3D_dBi[:,0,freq])+3)
    plt.grid(True)
    #plt.legend(('sin','sum exp'))
    plt.legend()


if config_AF.plot_contourf:
    X, Y = np.meshgrid(theta*180/math.pi, f_vec*10**(-3))
    plt.figure()
    levels = np.linspace(np.max(AF_3D_dBi)-20, np.max(AF_3D_dBi), 50)
    plt.contourf(X, Y, np.transpose(AF_3D_dBi), levels, cmap=plt.get_cmap('coolwarm'), extend='min' )
    plt.colorbar(label='|AF|² (dBi)', ticks = None)
    plt.ylabel('Frequency (kHz)')
    plt.xlabel('Theta (deg)')
    plt.title('phi = 0 (deg), 1x8 array')
    filename = 'AF_'+str(config_AF.active_arrays) + '_array_d=' + str(config_AF.distance*10**(2)) + '_w=' + str(config_AF.d) 
    #plt.savefig('plots/'+filename+'.png', dpi = 500, format = 'png')

if config_AF.active_arrays == 1:
    if config_AF.plot_contourf:
        X, Y = np.meshgrid(theta*180/math.pi, f_vec*10**(-3))
        plt.figure()
        levels = np.linspace(np.max(AF3ad_3D_dBi[:,0,:])-20, np.max(AF3ad_3D_dBi[:,0,:]), 50)
        plt.contourf(X, Y, np.transpose(AF3ad_3D_dBi[:,0,:]), levels, cmap=plt.get_cmap('coolwarm'), extend='min' )
        plt.colorbar(label='|AF|² (dBi)', ticks = None)
        plt.ylabel('Frequency (kHz)')
        plt.xlabel('Theta (deg)')
        plt.title('phi = 0 (deg), adaptive')
        filename = 'AF_'+str(config_AF.active_arrays) + '_array_d=' + str(config_AF.distance*10**(2)) + '_w=' + str(config_AF.d) 
        #plt.savefig('plots/'+filename+'.png', dpi = 500, format = 'png')

if config_AF.plot_contourf:
    X, Y = np.meshgrid(theta*180/math.pi, f_vec*10**(-3))
    plt.figure()
    levels = np.linspace(np.max(AF3ad_3D_dBi[:,1,:])-20, np.max(AF3ad_3D_dBi[:,1,:]), 50)
    plt.contourf(X, Y, np.transpose(AF3ad_3D_dBi[:,1,:]), levels, cmap=plt.get_cmap('coolwarm'), extend='min' )
    plt.colorbar(label='|AF|² (dBi)', ticks = None)
    plt.ylabel('Frequency (kHz)')
    plt.xlabel('Theta (deg)')
    plt.title('phi = 90 (deg), adaptive')
    filename = 'AF_'+str(config_AF.active_arrays) + '_array_d=' + str(config_AF.distance*10**(2)) + '_w=' + str(config_AF.d) 
    #plt.savefig('plots/'+filename+'.png', dpi = 500, format = 'png')

if config_AF.plot_contourf:
    X, Y = np.meshgrid(theta*180/math.pi, f_vec*10**(-3))
    plt.figure()
    levels = np.linspace(np.max(AF3ad2_3D_dBi[:,0,:])-20, np.max(AF3ad2_3D_dBi[:,0,:]), 50)
    plt.contourf(X, Y, np.transpose(AF3ad2_3D_dBi[:,0,:]), levels, cmap=plt.get_cmap('coolwarm'), extend='min' )
    plt.colorbar(label='|AF|² (dBi)', ticks = None)
    plt.ylabel('Frequency (kHz)')
    plt.xlabel('Theta (deg)')
    plt.title('phi = 0 (deg), adaptive')
    filename = 'AF_'+str(config_AF.active_arrays) + '_array_d=' + str(config_AF.distance*10**(2)) + '_w=' + str(config_AF.d) 
    #plt.savefig('plots/'+filename+'.png', dpi = 500, format = 'png')

if config_AF.plot_contourf:
    X, Y = np.meshgrid(theta*180/math.pi, f_vec*10**(-3))
    plt.figure()
    levels = np.linspace(np.max(AF3ad2_3D_dBi[:,1,:])-20, np.max(AF3ad2_3D_dBi[:,1,:]), 50)
    plt.contourf(X, Y, np.transpose(AF3ad2_3D_dBi[:,1,:]), levels, cmap=plt.get_cmap('coolwarm'), extend='min' )
    plt.colorbar(label='|AF|² (dBi)', ticks = None)
    plt.ylabel('Frequency (kHz)')
    plt.xlabel('Theta (deg)')
    plt.title('phi = 90 (deg), adaptive')
    filename = 'AF_'+str(config_AF.active_arrays) + '_array_d=' + str(config_AF.distance*10**(2)) + '_w=' + str(config_AF.d) 
    #plt.savefig('plots/'+filename+'.png', dpi = 500, format = 'png')

if config_AF.plot_contourf:
    X, Y = np.meshgrid(theta*180/math.pi, f_vec*10**(-3))
    plt.figure()
    levels = np.linspace(np.max(AF3_3D_dBi[:,0,:])-20, np.max(AF3_3D_dBi[:,0,:]), 50)
    plt.contourf(X, Y, np.transpose(AF3_3D_dBi[:,0,:]), levels, cmap=plt.get_cmap('coolwarm'), extend='min' )
    plt.colorbar(label='|AF|² (dBi)', ticks = None)
    plt.ylabel('Frequency (kHz)')
    plt.xlabel('Theta (deg)')
    plt.title('phi = 0 (deg), not adaptive')
    filename = 'AF_'+str(config_AF.active_arrays) + '_array_d=' + str(config_AF.distance*10**(2)) + '_w=' + str(config_AF.d) 
    #plt.savefig('plots/'+filename+'.png', dpi = 500, format = 'png')

if config_AF.plot_contourf:
    X, Y = np.meshgrid(theta*180/math.pi, f_vec*10**(-3))
    plt.figure()
    levels = np.linspace(np.max(AF3_3D_dBi[:,1,:])-20, np.max(AF3_3D_dBi[:,1,:]), 50)
    plt.contourf(X, Y, np.transpose(AF3_3D_dBi[:,1,:]), levels, cmap=plt.get_cmap('coolwarm'), extend='min' )
    plt.colorbar(label='|AF|² (dBi)', ticks = None)
    plt.ylabel('Frequency (kHz)')
    plt.xlabel('Theta (deg)')
    plt.title('phi = 90 (deg), not adaptive')
    filename = 'AF_'+str(config_AF.active_arrays) + '_array_d=' + str(config_AF.distance*10**(2)) + '_w=' + str(config_AF.d) 
    #plt.savefig('plots/'+filename+'.png', dpi = 500, format = 'png')
plt.show()