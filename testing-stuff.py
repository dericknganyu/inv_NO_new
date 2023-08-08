import numpy as np 
import torch, time
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch

import time
import argparse
from utilities.readData import readtoArray
from utilities.colorMap import parula

from utilities.utils import *
from utilities.models import *
from utilities.add_noise import *

torch.manual_seed(0)
np.random.seed(0)

res = 64 
res = res + 1
dX = 200
ntrain = 1000

PB = ['poisson', 'darcyPWC', 'darcyLN']
NORMA = ['Range', 'UnitGaussian', 'Gaussian', ]  

for pb in PB:
    for norma in NORMA:

        if pb == 'darcyPWC':
            fileName = "datasets/new_aUP_Square_TrainData=1_TestData=1_Resolution=513X513_Domain=[0,1]X[0,1].hdf5" 
            normPATH = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/normalisers/darcyPWC/UnitGaussianNormalizer/'
            pcaPATH  = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/pca/darcyPWC/UnitGaussianNormalizer/'
        if pb == 'darcyLN': 
            fileName = "datasets/aUL_Square_TrainData=1_TestData=1_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
            normPATH = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/normalisers/darcyLN/UnitGaussianNormalizer/'
            pcaPATH  = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/pca/darcyLN/UnitGaussianNormalizer/'
        if pb == 'poisson': 
            fileName = "datasets/fUG_Square_TrainData=1_TestData=1_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
            normPATH = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/normalisers/poisson/UnitGaussianNormalizer/'
            pcaPATH  = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/pca/poisson/UnitGaussianNormalizer/'

        # old_res = res
        # res = dX

        if norma == 'UnitGaussian':
            x_normalizer = torch.load(pcaPATH+"param_normalizer-%s_res-%s_d-%s-ntrain.pt"%(res-1, dX, ntrain))
            pcaX = torch.load(pcaPATH+"param_pca-%s_res-%s_d-%s-ntrain.pt"%(res-1, dX, ntrain))

        if norma == 'Gaussian':
            x_normalizer = torch.load(pcaPATH+"param_gaussian-normalizer-%s_res-%s_d-%s-ntrain.pt"%(res-1, dX, ntrain))
            pcaX = torch.load(pcaPATH+"param_pca-%s_res-%s_d-%s-ntrain.pt"%(res-1, dX, ntrain))

        if norma == 'Range':
            x_normalizer = torch.load(pcaPATH+"param_range-normalizer-%s_res-%s_d-%s-ntrain.pt"%(res-1, dX, ntrain))
            pcaX = torch.load(pcaPATH+"param_pca-%s_res-%s_d-%s-ntrain.pt"%(res-1, dX, ntrain))


        NAME = 'pca-check/' + pb + '-' + norma
        X_train, Y_train, X_test, Y_test = readtoArray(fileName, 1, 1, Nx = 512, Ny = 512)

        print ("Converting dataset to numpy array and subsamping.")
        tt = time.time()
        X_train = SubSample(np.array(X_train), res, res)
        X_test  = SubSample(np.array(X_test ), res, res)
        print ("    Conversion completed after %.2f minutes"%((time.time()-tt)/60))


        X_train0 = X_train.reshape(1, -1)
        X_test0  = X_test.reshape(1, -1)

        X_train1 = pcaX.transform(X_train0)
        X_test1  = pcaX.transform(X_test0)
        # if norma != 'Range':
        x_normalizer.cpu()

        X_train2 = x_normalizer.encode(torch.from_numpy(X_train1).float())
        X_test2  = x_normalizer.encode(torch.from_numpy(X_test1).float())

        # xx0 = np.arange(X_train0.shape[1])
        # xx1 = np.arange(X_train1.shape[1])
        fig = plt.figure(figsize=((5+2)*2, (5+0.5)*4))

        colourMap = parula() #plt.cm.jet #plt.cm.coolwarm

        plt.subplot(4, 2, 1)
        plt.xlabel('x')#, fontsize=16, labelpad=15)
        plt.ylabel('y')#, fontsize=16, labelpad=15)
        plt.title(r"Parameter, $\lambda(s)$")
        plt.imshow(X_train[0], cmap=colourMap, extent=[0, 1, 0, 1], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
        plt.colorbar()#format=OOMFormatter(-5))

        plt.subplot(4, 2, 2)
        plt.xlabel('x')#, fontsize=16, labelpad=15)
        plt.ylabel('y')#, fontsize=16, labelpad=15)
        plt.title(r"Parameter, $\lambda(s)$")
        plt.imshow(X_test[0], cmap=colourMap, extent=[0, 1, 0, 1], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
        plt.colorbar()#format=OOMFormatter(-5))


        plt.subplot(4, 1, 2)
        plt.plot(X_train0[0], label = '1')
        plt.plot(X_test0[0], label = '2')
        plt.title('reshaped')
        plt.legend(loc = 'upper right')

        plt.subplot(4, 1, 3)
        plt.plot(X_train1[0], label = '1')
        plt.plot(X_test1[0], label = '2')
        plt.title('PCA')
        plt.legend(loc = 'upper right')

        plt.subplot(4, 1, 4)
        plt.plot(X_train2[0], label = '1')
        plt.plot(X_test2[0], label = '2')
        plt.title('Normalised')
        plt.legend(loc = 'upper right')

        plt.savefig(NAME+'_figure_0.png')