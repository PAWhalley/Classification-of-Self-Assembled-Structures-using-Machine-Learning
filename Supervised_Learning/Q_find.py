import numpy as np
import torch

# Function to calculate Q
R=0.1

# Function to calculate if points x and y are withing R of each other
def charfun(x,y, R):
    if min(abs(x-y), 1 - abs(x-y)) <= R:
        return 1
    return 0

# Function that takes in the some number of point clouds in the form
#  of trainset, R and the number of points in each cloud
# and returns an array with the all the valus of Q in a list.
def Q_finder(trainset, R, npoints):
    labels = []
    for x in trainset.view(-1,npoints).detach().numpy():
        labels.append(1/(len(x)**2) * sum(charfun(k, j, R) for k in x for j in x))
    return labels
