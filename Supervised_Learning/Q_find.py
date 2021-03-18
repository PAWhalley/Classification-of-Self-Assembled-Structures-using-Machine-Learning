import numpy as np
import torch
from scipy.spatial.distance import cdist

# Function to calculate Q
R=0.1

# Function to calculate if points x and y are within R of each other
def charfun(x,y, R):
    if min(abs(x-y), 1 - abs(x-y)) <= R:
        return 1
    return 0

# Function that takes in the some number of point clouds in the form
#  of trainset, R and the number of points in each cloud
# and returns an array with the all the values of Q in a list.
def Q_finder(trainset, R, npoints):
    labels = []
    for x in trainset.view(-1,npoints).detach().numpy():
        labels.append(1/(len(x)**2) * sum(charfun(k, j, R) for k in x for j in x))
    return labels


def Q_findmat(trainset, R, npoints):
    trainset = trainset.view(-1,npoints, 1)
    labels = []
    counter = 0
    n_squared = npoints * npoints
    for x in trainset:
        counter += 1
        dist_mat = cdist(x,x, lambda x,y:min(abs(x-y), 1 - abs(x-y)))
        dist_mat[dist_mat <= R,], dist_mat[dist_mat> R] = 0, 1
        labels.append((1/n_squared) * (n_squared - (dist_mat.sum(axis = 0)).sum(axis = 0)))
    return labels
    

