# --- ANTENNA ARRAY setup variables ---
rows = 8                    # number of rows
columns = 8                 # number of columns
elements = rows*columns     # number of elements
c = 343
single_freq = 8*10**(3)
wavelength = c/single_freq
#distance = wavelength/2   
distance = 20 * 10**(-3) # distance between elements (m)
d =  0 * 10**(-3) # distance between arrays (if more than one active)
r_a1 = [0, 0, 0]            # coordinate position of origin of array1
r_a2 = [-0.08 - d/2, 0, 0]         # coordinate position of origin of array2
r_a3 = [0.08 + d/2, 0, 0]        # coordinate position of origin of array3
r_a4 = [0.24 + 3*d/2, 0, 0]         # coordinate position of origin of array4

active_arrays = 3

# --- PLOT OPTIONS ---
plot_single_f = 0
plot_setup = 0
plot_contourf = 1