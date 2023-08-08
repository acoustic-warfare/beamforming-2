from math import pi
import numpy as np

class Audio_source:
    #
    # Audio source with directional position data
    # Audio source is a sum of sine waves with frequencies ranging from
    # f_start to f_end, with uniform distribution
    #
    # Rho is the distance between the origin and the audio source
    #
    # t_start and t_end is the span of time which the audio source emittss ound

    def __init__(self, f_start, f_end, f_res, theta_deg, phi_deg, rho, t_start, t_end):
        

        self.__theta = theta_deg*pi/180
        self.__phi = phi_deg*pi/180
        self.__frequency = np.linspace(f_start, f_end, f_res)
        self.__t_start = t_start
        self.__t_end = t_end
        self.__rho = rho

    def get_theta(self):
        return self.__theta

    def get_phi(self):
        return self.__phi

    def get_frequency(self):
        return self.__frequency

    def get_t_start(self):
        return self.__t_start

    def get_t_end(self):
        return self.__t_end

    def get_rho(self):
        return self.__rho