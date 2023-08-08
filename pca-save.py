import numpy as np
import torch

from utilities.utils import *

import time
from utilities.readData import readtoArray

from sklearn.decomposition import PCA

torch.manual_seed(0)
np.random.seed(0)


dataPATHpoisson  = "../../../../../../localdata/Derick/stuart_data/Darcy_421/fUG_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
dataPathDarcyPWC = "../../../../../../localdata/Derick/stuart_data/Darcy_421/new_aUP_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
dataPathDarcyLN  = "../../../../../../localdata/Derick/stuart_data/Darcy_421/aUL_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
listdataPATH     = [dataPATHpoisson, dataPathDarcyPWC, dataPathDarcyLN]

pcaPATHpoisson  = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/pca/poisson/UnitGaussianNormalizer/'
pcaPATHDarcyPWC = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/pca/darcyPWC/UnitGaussianNormalizer/'
pcaPATHDarcyLN  = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/pca/darcyLN/UnitGaussianNormalizer/'
listPCAPATH     = [pcaPATHpoisson, pcaPATHDarcyPWC, pcaPATHDarcyLN]

ntrain = 1000
ntest = 5000

for dataPATH, pcaPATH in zip(listdataPATH , listPCAPATH):

    X_train, Y_train, _, _ = readtoArray(dataPATH, 1024, 5000, Nx = 512, Ny = 512)

    print ("Converting dataset to numpy array.")
    tt = time.time()
    X_train0 = np.array(X_train)
    Y_train0 = np.array(Y_train)
    print ("    Conversion completed after %.4f seconds"%(time.time()-tt))

    for res in [64]: #[32, 128, 256, 512]: # 
        res = res + 1
        print ("Subsampling dataset to the required resolution.", res)
        tt = time.time()
        X_train1 = SubSample(X_train0, res, res)
        Y_train1 = SubSample(Y_train0, res, res)
        print ("    Subsampling completed after %.4f seconds"%(time.time()-tt))

        print ("Taking out the required train/test size.")
        tt = time.time()
        x_train1 = X_train1[:ntrain, :, :].reshape(ntrain, -1)
        y_train1 = Y_train1[:ntrain, :, :].reshape(ntrain, -1)
        print ("    Taking completed after %.4f seconds"%(time.time()-tt))

        for dX in [150]:#[50, 70, 100, 150, 250, 30]: # 

            print ("Obtaining the PCA functions")   
            tt = time.time()
             
            pcaX = PCA(n_components = dX, random_state = 0).fit(x_train1)
            pcaY = PCA(n_components = dX, random_state = 0).fit(y_train1) 
            torch.save(pcaX, pcaPATH+"param_pca-%s_res-%s_d-%s-ntrain.pt"%(res-1, dX, ntrain))
            torch.save(pcaY, pcaPATH+"solut_pca-%s_res-%s_d-%s-ntrain.pt"%(res-1, dX, ntrain)) 

            print ("    Obtaining the PCA functions done after %s seconds"%(time.time()-tt))


            x_train = pcaX.transform(x_train1)
            y_train = pcaY.transform(y_train1)


            # x_normalizer = UnitGaussianNormalizer(torch.from_numpy(x_train).float().to(device))
            # y_normalizer = UnitGaussianNormalizer(torch.from_numpy(y_train).float().to(device))
            # torch.save(x_normalizer, pcaPATH+"param_normalizer-%s_res-%s_d-%s-ntrain.pt"%(res-1, dX, ntrain))
            # torch.save(y_normalizer, pcaPATH+"solut_normalizer-%s_res-%s_d-%s-ntrain.pt"%(res-1, dX, ntrain))

                    #NEW
            # x_normalizer = GaussianNormalizer(torch.from_numpy(x_train).float().to(device))
            # y_normalizer = GaussianNormalizer(torch.from_numpy(y_train).float().to(device))
            # torch.save(x_normalizer, pcaPATH+"param_gaussian-normalizer-%s_res-%s_d-%s-ntrain.pt"%(res-1, dX, ntrain))
            # torch.save(y_normalizer, pcaPATH+"solut_gaussian-normalizer-%s_res-%s_d-%s-ntrain.pt"%(res-1, dX, ntrain))

            x_normalizer = RangeNormalizer(torch.from_numpy(x_train).float().to(device))
            y_normalizer = RangeNormalizer(torch.from_numpy(y_train).float().to(device))
            torch.save(x_normalizer, pcaPATH+"param_range-normalizer-%s_res-%s_d-%s-ntrain.pt"%(res-1, dX, ntrain))
            torch.save(y_normalizer, pcaPATH+"solut_range-normalizer-%s_res-%s_d-%s-ntrain.pt"%(res-1, dX, ntrain))