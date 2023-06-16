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

    if wavelength_rel>0.1581:
        mode = 1
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
F = 500
P = 500
d = config_AF.distance
f = config_AF.single_freq
fs = 16 * 10**3 # sampling frequency

f_vec = np.linspace(100, 8* 10**3, F)
f_max = fs/2
lambda_max = c/f_max
d_min = lambda_max/2
k_vec = 2*math.pi*f_vec/c


wavelength = c/f



array_matrices = antenna_setup()
adaptive_weight_matrix = adaptive_matrix(config_AF.rows, config_AF.columns)
#print(np.shape(adaptive_weight_matrix))

x_coord = array_matrices[0].get_x_coord()
y_coord = array_matrices[0].get_y_coord()
for array in range(config_AF.active_arrays)[1:]:
    x_coord = np.hstack((x_coord, array_matrices[array].get_x_coord()))
    y_coord = np.hstack((y_coord, array_matrices[array].get_y_coord()))
xy_coord = np.dstack((x_coord,y_coord))

print(np.shape(xy_coord))

wavelength = c/f
k = 2*math.pi*f/c
n = np.array([np.linspace(0,N-1,N)])


theta = np.array([np.linspace(-90,90,P) * math.pi/180]).T
theta0 = 0 * math.pi/180
psi = k*d*np.sin(theta)
psi0 = k*d*math.sin(theta0) 

# phi = np.linspace(-90,90,P)
# phi0 = 0
# xi = k*d*np.sin(theta)*np.cos(phi)
# xi0 = k*d*math.sin(theta0)*math.cos(phi0)


AF = np.sin(N/2*(psi-psi0)) / (np.sin(1/2*(psi-psi0)))
AF = np.square(np.absolute(AF))
AF_dBi = 10*np.log10(AF)

#AF2 = np.exp(1j*np.matmul((k*d*(np.sin(theta) - math.sin(theta0))),n)) # First draft
AF2 = np.exp(1j*(np.sin(theta) - math.sin(theta0))*(k*xy_coord[0,:,0])) # second draft, can handle several arrays defined in config_AF.py file
print(weight_index(f))
print(adaptive_weight_matrix[weight_index(f)-1,0:8])

#AF2 *= adaptive_weight_matrix[weight_index(f)-1,0:8]
AF2_terms = AF2
AF2 = np.sum(AF2, axis=1)#, keepdims=True)
AF2 = np.square(np.absolute(AF2))
AF2_dBi = 10*np.log10(AF2)
print(np.shape(AF2_terms))
print(np.shape(AF2))

# Calculate |AF|² for the arrays defined in config_AF.py, for a range of frequencies
AF_3D = np.zeros((P,F))
for k_ind in range(F):
    k = k_vec[k_ind]
    f = k*c/2*math.pi
    # AF_2D = np.exp(1j*np.matmul((k*d*(np.sin(theta) - math.sin(theta0))),n)) # first draft
    AF_2D = np.exp(1j*k*(np.sin(theta) - math.sin(theta0))*(xy_coord[0,:,0])) # second draft, can handle several arrays defined in config_AF.py file
    #print(np.shape(AF_2D))
    #print(np.shape(adaptive_weight_matrix))
    AF_2D *= adaptive_weight_matrix[weight_index(f)-1,0:config_AF.columns]
    AF_2D = np.sum(AF_2D, axis=1) #, keepdims=True)
    AF_2D = np.square(np.absolute(AF_2D))
    AF_3D[:,k_ind] = AF_2D
AF_3D_dBi = 10*np.log10(AF_3D)



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
    plt.figure()
    plt.plot(theta*180/math.pi, AF_dBi, 'r',linewidth=2)
    plt.plot(theta*180/math.pi, AF2_dBi, 'b--', linewidth=2)
    plt.xlabel('Theta (deg)')
    plt.ylabel('|AF|²')
    plt.title('Array factor')
    #plt.ylim(np.max(AF_dBi)-20, np.max(AF_dBi)+3)
    plt.grid(True)
    plt.legend(('sin','sum exp'))


if config_AF.plot_contourf:
    X, Y = np.meshgrid(theta*180/math.pi, f_vec*10**(-3))
    plt.figure()
    levels = np.linspace(np.max(AF_3D_dBi)-20, np.max(AF_3D_dBi), 50)
    plt.contourf(X, Y, np.transpose(AF_3D_dBi), levels, cmap=plt.get_cmap('coolwarm'), extend='min' )
    plt.colorbar(label='|AF|² (dBi)', ticks = None)
    plt.ylabel('Frequency (kHz)')
    plt.xlabel('Theta (deg)')
    filename = 'AF_'+str(config_AF.active_arrays) + '_array_d=' + str(config_AF.distance*10**(2)) + '_w=' + str(config_AF.d) 
    #plt.savefig('plots/'+filename+'.png', dpi = 500, format = 'png')

plt.show()