from utilities.readData import readtoArray
import torch
import numpy as np
import matplotlib.pyplot as plt
from utilities.colorMap import parula
import time
import argparse


def add_noise2d(data, delta, d=2, p=2):
    #d, the dimension of the data
    #p, the order of the norm

    if isinstance(data, tuple):
        X_train, Y_train, X_test, Y_test  = data
    if isinstance(data, str):
        X_train, Y_train, X_test, Y_test  = readtoArray(data)
    if delta == 0:
        return X_train, Y_train, X_test, Y_test 
    ntrain, res, _ = np.shape(Y_train)
    ntest , _  , _ = np.shape(Y_test)

    np.random.seed(0)
    noise = np.random.normal(size=(ntrain + ntest, res, res))
    
    h = 1/(res -1) #step size
    norm_train = (h**(d/p))*np.linalg.norm(Y_train, axis=(1,2), ord=p)
    norm_train  = norm_train.reshape(-1, 1, 1)

    norm_test  = (h**(d/p))*np.linalg.norm(Y_test , axis=(1,2), ord=p) 
    norm_test  = norm_test.reshape(-1, 1, 1)

    Y_train = Y_train + delta * norm_train * noise[0:ntrain, :, :]
    Y_test  = Y_test  + delta * norm_test  * noise[ntrain::, :, :]

    return X_train, Y_train, X_test, Y_test 

def add_noise1d(data, delta, d=2, p=2):
    #d, the dimension of the data
    #p, the order of the norm

    if isinstance(data, tuple):
        X_train, Y_train, X_test, Y_test  = data
    if isinstance(data, str):
        X_train, Y_train, X_test, Y_test  = readtoArray(data)
    if delta == 0:
        return X_train, Y_train, X_test, Y_test 
    ntrain, res = np.shape(Y_train)
    ntest , _   = np.shape(Y_test)

    np.random.seed(0)
    noise = np.random.normal(size=(ntrain + ntest, res))
    
    h = 1/(res -1) #step size
    norm_train = (h**(d/p))*np.linalg.norm(Y_train, axis=1, ord=p)
    norm_train  = norm_train.reshape(-1, 1)

    norm_test  = (h**(d/p))*np.linalg.norm(Y_test , axis=1, ord=p) 
    norm_test  = norm_test.reshape(-1, 1)

    Y_train = Y_train + delta * norm_train * noise[0:ntrain, :]
    Y_test  = Y_test  + delta * norm_test  * noise[ntrain::, :]

    return X_train, Y_train, X_test, Y_test 


if __name__ == '__main__':
    xmin, xmax = 0, 1 #-1, 1
    ymin, ymax = 0, 1 #-1, 1
    #Nx, Ny = 512, 512 #16, 16
    #N_train, N_test = 1, 1
    
    parser = argparse.ArgumentParser(description='parse mode')
    parser.add_argument('--delta' , default =0.1, type = float, help='noise level')
    args = parser.parse_args()

    prefix = ""
    data = prefix+"new_aUP_Square_TrainData=1_TestData=1_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
    delta = args.delta
    
    X_train, Y_train, X_test, Y_test = readtoArray(data)
    _, Y_train_noise, _, _      = add_noise2d(data, delta)
    
    fig = plt.figure(figsize=((5+2)*2, 5))
    
    colourMap = parula() #plt.cm.jet
                
    fig.suptitle("Plot of a randomly generated $f$ and its FDM generated $u$ satisfying $- \Delta u = f, \partial \Omega = 0$ on $\Omega = [%s,%s]x[%s,%s]$"%(xmin, xmax, ymin, ymax))
    
    plt.subplot(1, 2, 1)
    plt.xlabel('x')#, fontsize=16, labelpad=15)
    plt.ylabel('y')#, fontsize=16, labelpad=15)
    plt.title("u")
    plt.imshow(Y_train[0], cmap=colourMap, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.xlabel('x')#, fontsize=16, labelpad=15)
    plt.ylabel('y')#, fontsize=16, labelpad=15)
    plt.title("$u + \delta \cdot ||u||_2 \cdot \mathcal{N}(0,1)$")
    plt.imshow(Y_train_noise[0], cmap=colourMap, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
    plt.colorbar()
    
    t = time.localtime()
    plt.savefig("figures/noisy_data_%s-delta_"%(delta)+time.strftime('%Y%m%d-%H:%M:%S', t)+'.png')
    plt.show()
    

