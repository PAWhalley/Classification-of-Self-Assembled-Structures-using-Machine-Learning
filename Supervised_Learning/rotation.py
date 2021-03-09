# From from https://github.com/charlesq34/pointnet/blob/master/provider.py

import numpy as np
import torch

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #angles = np.random.uniform(size=(3)) * 2 * np.pi
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


def LOP(spheres, s):
    '''
    Takes an array of spheres and returns something similar to the hyperuniformity of each sphere.
    Can be thought of as the variance of the number of points in a disk of radius s on the 
    suface of the sphere.
    '''
    LOP = np.array([])
    for sphere in spheres:
        #print(sphere)
        ns = np.array([])
        count = 0
        for x in sphere:
            for y in sphere:
                if np.linalg.norm(x-y) < s:
                    count += 1
            ns = np.append(ns, count)
            count = 0
        ns2 = ns ** 2
        LOP = np.append(LOP, np.average(ns2) - np.average(ns) ** 2)
    return LOP
