def calc_r_prime(d):
    half = d/2
    r_prime = np.zeros((2, config_GS.elements))
    element_index = 0
    for row in range(config_GS.rows):
        for col in range(config_GS.columns*config_GS.arrays):
            print(str(row) + ' ' + str(col))
            r_prime[0,element_index] = col * d - config_GS.columns * config_GS.arrays * half + half
            r_prime[1, element_index] = row * d - config_GS.rows * half + half
            element_index += 1

    plt.figure()
    plt.title('Array setup')
    plt.scatter(r_prime[0,:], r_prime[1,:].T)
    plt.xlim([-(d*config_GS.columns * config_GS.arrays/2 + d) , d*config_GS.columns * config_GS.arrays/2 + d])

    return r_prime
