# Rotate_point_cloud2 from https://github.com/charlesq34/pointnet/blob/master/provider.py

import numpy as np
import torch

def rotate_point_cloud2(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = [np.arccos(1-2*np.random.uniform())]
        angles[1:] = np.random.uniform(size=(2)) * 2 * np.pi
        cosval = np.cos(angles)
        sinval = np.sin(angles)

        x_rot_mat = np.array([[1, 0, 0],
                              [0, cosval[0], -sinval[0]],
                              [0, sinval[0], cosval[0]]])

        y_rot_mat = np.array([[cosval[1], 0, sinval[1]],
                              [0, 1, 0],
                              [-sinval[1], 0, cosval[1]]])

        z_rot_mat = np.array([[cosval[2], -sinval[2], 0],
                              [sinval[2], cosval[2], 0],
                              [0, 0, 1]])

        # Overall rotation calculated from x,y,z -->
        # order matters bc matmult not commutative 
        overall_rot = np.dot(z_rot_mat,np.dot(y_rot_mat,x_rot_mat))
        # Transposes bc overall_rot operates on col. vec [[x,y,z]]
        rotated_pc = np.dot(overall_rot,batch_data[k,:,:3].T).T
        rotated_data[k] = np.concatenate((rotated_pc, batch_data[k,:,3:]), axis=1)

    return torch.tensor(rotated_data)

def LOP(spheres, s, N):
    LOP = np.array([])
    for sphere in spheres:
        ns = np.array([])
        count = 0
        for x in torch.tensor(uniform_spherical_distribution(100)):
            for y in sphere:
                if np.linalg.norm(x-y) < s:
                    count += 1
            ns = np.append(ns, count)
            count = 0
        ns2 = ns ** 2
        LOP = np.append(LOP, np.average(ns2) - np.average(ns) ** 2)
    return LOP


# Function to distribute N points on the surface of a sphere 
# (source: http://www.softimageblog.com/archives/115)
def uniform_spherical_distribution(N): 
    pts = []   
    inc = np.pi * (3 - np.sqrt(5)) 
    off = 2 / float(N) 
    for k in range(0, int(N)): 
        y = k * off - 1 + (off / 2) 
        r = np.sqrt(1 - y*y) 
        phi = k * inc 
        pts.append([np.cos(phi)*r, y, np.sin(phi)*r])   
    return pts


def rotate_point_cloud(batch_data):
	'''
	Rotates the pointcloud uniformally - PW.
	This is the one we should use
	'''
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        overall_rot = random_rotation_matrix()
        # Transposes bc overall_rot operates on col. vec [[x,y,z]]
        rotated_pc = np.dot(overall_rot,batch_data[k,:,:3].T).T
        rotated_data[k] = np.concatenate((rotated_pc, batch_data[k,:,3:]), axis=1)

    return torch.tensor(rotated_data)

def random_rotation_matrix():
	'''
	Creates rotation matrix - PW
	'''
    theta = np.arccos(2*np.random.uniform(low = 0,high = 1)-1)
    phi = np.random.uniform(low = 0,high = 2*np.pi)
    u = np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
    theta = np.random.uniform(low = 0,high = 2*np.pi)
    A = np.zeros((3,3))
    A[0][0] = np.cos(theta) + (u[0]**2)*(1-np.cos(theta))
    A[0][1] = u[0]*u[1]*(1-np.cos(theta)) - u[2]*np.sin(theta)
    A[0][2] = u[0]*u[2]*(1-np.cos(theta)) + u[1]*np.sin(theta)
    A[1][0] = u[1]*u[0]*(1-np.cos(theta)) + u[2]*np.sin(theta)
    A[1][1] = np.cos(theta) + (u[1]**2)*(1-np.cos(theta))
    A[1][2] = u[1]*u[2]*(1-np.cos(theta)) - u[0]*np.sin(theta)
    A[2][0] = u[2]*u[0]*(1-np.cos(theta)) - u[1]*np.sin(theta)
    A[2][1] = u[2]*u[1]*(1-np.cos(theta)) + u[0]*np.sin(theta)
    A[2][2] = np.cos(theta) + (u[2]**2)*(1-np.cos(theta))
    return A
