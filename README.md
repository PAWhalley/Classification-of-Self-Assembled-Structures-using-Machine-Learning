# Classification of Self-Assembled Structures using Machine Learning
In molecular dynamic simulations linking the microscopic shape of a cluster to the macroscopic properties of a material is of great importance. Extracting lower dimensional representations of our data, also called order parameters can enable us to make predictions about the state of our system (e.g. phase our system is in). Learning a low dimensional representation of our data can also assist the classification of that data. Specifically, the extracted lower dimensional representation can be used in combination with the original data for classification. Calculating order parameters can be extremely computationally expensive and establishing new ones can be tremendously difficult. 
In this repository we have method that explore how machine learning, both supervised and unsupervised, can be used to faster compute known order parameters, avoiding the direct (and expensive) formulaic approach and also to devise new order parameters (extract features). 
On a selection of toy examples, which are simplifications of actual molecular data we 
use a supervised learning tool (PointNet) to predict order parameters arising in opinion dynamics and artificially generated point clouds of spherical and ellipsoidal shape. We also use unsupervised learning techniques (autoencoders and restricted Boltzmann machines) to extract new order parameters.


PointNet for supervised learning was adapted from https://github.com/myx666/pointnet-in-pytorch/blob/master/pointnet.pytorch/dataset.py
