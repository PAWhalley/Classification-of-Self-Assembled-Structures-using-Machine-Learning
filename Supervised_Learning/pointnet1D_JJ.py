# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 09:40:41 2021

@author: Johanna

pointnet from https://github.com/myx666/pointnet-in-pytorch/blob/master/pointnet.pytorch/dataset.py
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
#from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt

import sys 
sys.path.insert(0, './Documents/Uni/ML')
from spheres_JJ import sphere_generator

class PointNetCls(nn.Module):
    def __init__(self, k=16):
        super(PointNetCls, self).__init__()
        self.conv1 = torch.nn.Conv1d(npoints, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(k)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        x = F.relu(self.bn6(self.dropout(x)))
        return F.log_softmax(x, dim=1)
    


###########################################################    
# Training
# Data generation 
nclouds = 100
npoints = 100

# Generate uniform
uniform = torch.rand(nclouds, npoints, 1)

# Generate normals with mean in U[0, 1] and covaraince in U[0, 0.1]
means = torch.rand(nclouds)
covs = torch.rand(nclouds)*0.1
normal = torch.randn(nclouds, npoints, 1)
for i in range(nclouds):
    normal[i, :, 0] = normal[i, :, 0]*covs[i] + means[i]
    # to those smaller than 0 I add 1
    (normal[i, :, 0])[normal[i, :, 0]<0] = 1 + (normal[i, :, 0])[normal[i, :, 0]<0]
    # to those bigger than 1 I substract 1
    (normal[i, :, 0])[normal[i, :, 0]>1] = -1 + (normal[i, :, 0])[normal[i, :, 0]>1]
 
trainset = torch.cat((normal, uniform))

# Labels
zeros = torch.zeros(nclouds, dtype=torch.long)
ones = torch.ones(nclouds, dtype=torch.long)
labels = torch.cat((zeros, ones))

dataset = TensorDataset(trainset, labels)
# Creating the batches
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                          shuffle=True, num_workers=2, drop_last=True)

#########################################################
# Testing clouds

ncloudstest = 50
npointstest = 100

# Generate uniform
uniformtest = torch.rand(ncloudstest, npointstest, 1)

# Generate normals with mean in U[0, 1] and covaraince in U[0, 0.1]
meanstest = torch.rand(ncloudstest)
covstest = torch.rand(ncloudstest)*0.1
normaltest = torch.randn(ncloudstest, npointstest, 1)
for i in range(ncloudstest):
    normaltest[i, :, 0] = normaltest[i, :, 0]*covstest[i] + meanstest[i]
    # to those smaller than 0 I add 1
    (normaltest[i, :, 0])[normaltest[i, :, 0]<0] = 1 + (normaltest[i, :, 0])[normaltest[i, :, 0]<0]
    # to those bigger than 1 I substract 1
    (normaltest[i, :, 0])[normaltest[i, :, 0]>1] = -1 + (normaltest[i, :, 0])[normaltest[i, :, 0]>1]

testset = torch.cat((normaltest, uniformtest))

# Labels
testzeros = torch.zeros(ncloudstest, dtype=torch.long)
testones = torch.ones(ncloudstest, dtype=torch.long)
testlabels = torch.cat((testzeros, testones))
#print(labels.view(-1,2).size())


testset = TensorDataset(testset, testlabels)
# Creating the batches
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=True, num_workers=2, drop_last=True)

############################################################

net = PointNetCls(k=2)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

nepochs = 10

train_acc = np.zeros(nepochs)
test_acc = np.zeros(nepochs)
for epoch in range(nepochs):  # loop over the dataset multiple times
    epoch_train_acc = 0.0
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        
        _, train_predicted = torch.max(outputs.data, 1)
        epoch_train_acc += (train_predicted == labels).sum().item()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
            
    epoch_test_acc = 0.0
    for data in testloader:
        inputs, labels = data
        outputs = net(inputs)
        _, test_predicted = torch.max(outputs.data, 1)
        epoch_test_acc += (test_predicted == labels).sum().item()

            
    
    train_acc[epoch] = epoch_train_acc/nclouds/2
    test_acc[epoch] = epoch_test_acc/ncloudstest/2
    

print('Finished Training')

x = np.arange(nepochs)
plt.plot(x, train_acc, label='Training accuracy')
plt.plot(x, test_acc, label='Test accuracy')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
#plt.minorticks_on()
plt.xticks(x)
plt.grid(True, which='both')
plt.show()


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test point clouds: %d %%' % (
    100 * correct / total))
