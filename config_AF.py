# --- ANTENNA ARRAY setup variables ---
rows = 8                    # number of rows
columns = 8                 # number of columns
elements = rows*columns     # number of elements
distance = 20 * 10**(-3)    # distance between elements (m)

d = 0 # distance between arrays (if more than one active)
r_a1 = [-0.24 - 3*d/2, 0, 0]            # coordinate position of origin of array1
r_a2 = [-0.08 - d/2, 0, 0]         # coordinate position of origin of array2
r_a3 = [0.08 + d/2, 0, 0]        # coordinate position of origin of array3
r_a4 = [0.24 + 3*d/2, 0, 0]         # coordinate position of origin of array4

active_arrays = 1