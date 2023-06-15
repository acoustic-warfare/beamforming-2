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

    if plot_setup:
        for array in range(sub_arrays):
            plt.title('Array setup')
            plt.scatter(array_matrices[array].get_r_prime()[0,:], array_matrices[array].get_r_prime()[1,:])
    
    #array_coordinates = np.array([array_matrix_1.get_r_prime(), array_matrix_2.get_r_prime(), \
    #                              array_matrix_3.get_r_prime(), array_matrix_4.get_r_prime()])[:config_AF.active_arrays]
    return array_matrices


plot_single_f = 0
plot_setup = 0
plot_cont = 1
c = 343         # speed of sound

F = 100
N = 8
P = 500
d = 20 * 10**-3 
f = 100 #8 * 10**3
fs = 16 * 10**3 # sampling frequency

f_vec = np.linspace(100, 8 * 10**3, F)
f_max = fs/2
lambda_max = c/f_max
d_min = lambda_max/2
k_vec = 2*math.pi*f_vec/c


array_matrices = antenna_setup()
array_coord = np.zeros((3, config_AF.elements))

for array in range(config_AF.active_arrays):
    array_coord = np.vstack((array_coord, array_matrices[array].get_r_prime()))

print(np.shape(array_coord))

wavelength = c/f
k = 2*math.pi*f/c

theta = np.array([np.linspace(-90,90,P) * math.pi/180]).T
theta0 = 0 * math.pi/180

n = np.array([np.linspace(0,N-1,N)])
# phi = np.linspace(-90,90,P)
# phi0 = 0

psi = k*d*np.sin(theta) #*np.sin(phi)
psi0 = k*d*math.sin(theta0) #*math.sin(phi0)

# xi = k*d*np.sin(theta)*np.cos(phi)
# xi0 = k*d*math.sin(theta0)*math.cos(phi0)


AF = np.sin(N/2*(psi-psi0)) / (np.sin(1/2*(psi-psi0)))
AF = np.square(np.absolute(AF))
AF_dBi = 10*np.log10(AF)

AF2 = np.exp(1j*np.matmul((k*d*(np.sin(theta) - math.sin(theta0))),n))
AF2_terms = AF2
AF2 = np.sum(AF2, axis=1) #, keepdims=True)
AF2 = np.square(np.absolute(AF2))
AF2_dBi = 10*np.log10(AF2)

if plot_single_f:
    # plot |AF|²
    plt.figure()
    plt.plot(theta*180/math.pi, AF_dBi, 'r',linewidth=2)
    plt.plot(theta*180/math.pi, AF2_dBi, 'b--', linewidth=2)
    plt.xlabel('Theta (deg)')
    plt.ylabel('|AF|²')
    plt.title('Array factor')
    #plt.ylim(-40, np.max(AF_dBi)+3)
    plt.grid(True)
    plt.legend(('sin','sum exp'))



#print(np.shape(k_vec))
#A = np.tile(np.array([1,2,3,4]).transpose(), (3, 1))
#B = np.swapaxes(np.repeat(A[:, :, np.newaxis], 2, axis=2),1,2)
#print(A)
#print(B)
#print(np.shape(B))
#print(B[:,:,0])

#A = np.tile(np.array(k_vec).transpose(), (P, 1))
#K = np.swapaxes(np.repeat(A[:, :, np.newaxis], N, axis=2),1,2)
#print(A)
#print(K)
#print(np.shape(K))
#print(K[:,:,0])
#K_sum = np.sum(K, axis=1) #, keepdims=True)


AF_3D = np.zeros((P,F))
for k_ind in range(F):
    k = k_vec[k_ind]
    AF_2D = np.exp(1j*np.matmul((k*d*(np.sin(theta) - math.sin(theta0))),n))
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

if plot_cont:
    X, Y = np.meshgrid(theta, f_vec)
    plt.figure()
    #plt.imshow(AF_3D_dBi)
    levels = np.linspace(-10, np.max(AF_dBi), 50)
    plt.contourf(X, Y, np.transpose(AF_3D_dBi), levels, cmap=plt.get_cmap('coolwarm'), extend='min' )
    plt.colorbar()


plt.show()