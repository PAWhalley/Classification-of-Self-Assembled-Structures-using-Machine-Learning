import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def sphere_generator(noise_parameter, nclouds, npoints):

    data = []
    for cloudn in range(nclouds):
        x_eps = np.random.normal(size = npoints)*noise_parameter
        y_eps = np.random.normal(size = npoints)*noise_parameter
        z_eps = np.random.normal(size = npoints)*noise_parameter
        # create dataset
        Phi = 2*np.pi*np.random.rand(npoints) - np.pi
        Theta = np.pi*np.random.rand(npoints) - 0.5*np.pi
        X = (1+x_eps)*np.cos(Theta)*np.cos(Phi)
        Y = (1+y_eps)*np.cos(Theta)*np.sin(Phi)
        Z = (1+z_eps)*np.sin(Theta)
        cloud = np.stack((X, Y, Z), axis=1)
        data.append(cloud)    
        
    return np.array(data, dtype=np.float32)

def ellipsoid_generator(noise_parameter, xstretch, ystretch, zstretch, nclouds, npoints):
    
    data = []
    for cloudn in range(nclouds):
        a_x = xstretch  #x-direction stretch
        b_y = ystretch  #y-direction stretch
        c_z  = zstretch #z-direction stretch
        x_eps = np.random.normal(size = npoints)*noise_parameter
        y_eps = np.random.normal(size = npoints)*noise_parameter
        z_eps = np.random.normal(size = npoints)*noise_parameter
        # create dataset
        Phi = 2*np.pi*np.random.rand(npoints) - np.pi
        Theta = np.pi*np.random.rand(npoints) - 0.5*np.pi
        X = (a_x+x_eps)*np.cos(Theta)*np.cos(Phi)
        Y = (b_y+y_eps)*np.cos(Theta)*np.sin(Phi)
        Z = (c_z+z_eps)*np.sin(Theta)
        
        cloud = np.stack((X, Y, Z), axis=1)
        data.append(cloud) 
        
    return np.array(data, dtype=np.float32)
        