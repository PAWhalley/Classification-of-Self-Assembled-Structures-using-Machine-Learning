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

def return_rod(noise_parameter, r, z_stretch, nclouds, npoints):
    
    data = []
    for cloudn in range(nclouds):
        z_eps = np.random.normal(size = npoints)*noise_parameter
        x_eps = np.random.normal(size = npoints)*noise_parameter
        # create dataset 
        Theta = 2*np.pi*np.random.rand(npoints) - np.pi
        #X = r*(1+x_eps)*np.cos(Theta)
        X = r*(1+x_eps)*np.cos(Theta)
        Y = np.random.uniform(low= -z_stretch/2,high = z_stretch/2,size = npoints)
        Z = r*(1+z_eps)*np.sin(Theta)

        data_rod = np.stack((X,Y,Z), axis = 1)
        data.append(data_rod) 
        #print(np.shape(np.array(data, dtype=np.float32)))
        
    return np.array(data, dtype=np.float32)

def return_square(noise_parameter,nclouds, npoints):
    m = npoints//6
    data = []
    for cloudn in range(nclouds):
        x_eps = np.random.normal(size = m)*noise_parameter
        y_eps = np.random.normal(size = m)*noise_parameter
        z_eps = np.random.normal(size = m)*noise_parameter
        # create dataset
        L  = np.random.uniform(1,10)

        X1 = np.random.uniform(low = -L/2, high= L/2,size = m)
        Y1 = np.random.uniform(low = -L/2, high= L/2,size = m)
        Z1 = np.zeros(m)+(L/2+z_eps)

        X2 = np.random.uniform(low = -L/2, high= L/2,size = m)
        Y2 = np.random.uniform(low = -L/2, high= L/2,size = m)
        Z2 = np.zeros(m)+(-L/2+z_eps)

        X3 = np.random.uniform(low = -L/2, high= L/2,size = m)
        Y3 = np.zeros(m)+(L/2+y_eps)
        Z3 = np.random.uniform(low = -L/2, high= L/2,size = m)

        X4 = np.random.uniform(low = -L/2, high= L/2,size = m)
        Y4 = np.zeros(m)+(-L/2+y_eps)
        Z4 = np.random.uniform(low = -L/2, high= L/2,size = m)

        X5 = np.zeros(m)+(L/2+x_eps)
        Y5 = np.random.uniform(low = -L/2, high= L/2,size = m)
        Z5 = np.random.uniform(low = -L/2, high= L/2,size = m)

        X6 = np.zeros(m)+(-L/2+x_eps)
        Y6 = np.random.uniform(low = -L/2, high= L/2,size = m)
        Z6 = np.random.uniform(low = -L/2, high= L/2,size = m)

    #    data_ellipsoid = np.array([[X1,Y1,Z1],[X2,Y2,Z2],[X3,Y3,Z3],[X4,Y4,Z4],[X5,Y5,Z5],[X6,Y6,Z6]])
        d1 = np.array([X1,Y1,Z1])
        d2 = np.array([X2,Y2,Z2])
        d3 = np.array([X3,Y3,Z3])
        d4 = np.array([X4,Y4,Z4])
        d5 = np.array([X5,Y5,Z5])
        d6 = np.array([X6,Y6,Z6])

        data_ellipsoid = np.hstack((d1,d2,d3,d4,d5,d6))
        data_ellipsoid = np.array(data_ellipsoid)
        data.append(np.transpose(data_ellipsoid))
    return np.array(data, dtype=np.float32)
        
