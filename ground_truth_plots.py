import numpy as np 
import torch, time
import matplotlib.pyplot as plt
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


parser = argparse.ArgumentParser(description='For inputing resolution')

#Adding required parser argument
parser.add_argument('--res', default=64, type=int, help='Specify redolution')
parser.add_argument('--pb', default='darcyPWC', type=str, help='Specify Problem: poison, darcyPWC, darcyLN')


args = parser.parse_args()
res = args.res + 1
pb = args.pb



if pb == 'darcyPWC':
    fileName = "datasets/new_aUP_Square_TrainData=1_TestData=1_Resolution=513X513_Domain=[0,1]X[0,1].hdf5" 
if pb == 'darcyLN': 
    fileName = "datasets/aUL_Square_TrainData=1_TestData=1_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
if pb == 'poisson': 
    fileName = "datasets/fUG_Square_TrainData=1_TestData=1_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"


X_train, Y_train, _, _ = readtoArray(fileName, 1, 1, Nx = 512, Ny = 512)

print ("Converting dataset to numpy array and subsamping.")
tt = time.time()
X_train = SubSample(np.array(X_train), res, res)
Y_train = SubSample(np.array(Y_train), res, res)
print ("    Conversion completed after %.2f minutes"%((time.time()-tt)/60))


fig = plt.figure(figsize=((6, 5))) 

plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("FDM Truth")
colourMap = parula()
plt.imshow(X_train[0], cmap=colourMap, extent=[0, 1, 0, 1], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))


fig.tight_layout() 
plt.savefig('figures/%s-parameter-GroundTruth.png'%(pb),dpi=300)

fig = plt.figure(figsize=((6, 5))) 

plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("FDM Truth")
colourMap = parula()
plt.imshow(Y_train[0], cmap=colourMap, extent=[0, 1, 0, 1], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))


fig.tight_layout() 
plt.savefig('figures/%s-solution-GroundTruth.png'%(pb),dpi=300)