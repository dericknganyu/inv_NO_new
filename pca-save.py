import numpy as np
import torch

from utilities.utils import *

import time
from utilities.readData import readtoArray

from sklearn.decomposition import PCA

torch.manual_seed(0)
np.random.seed(0)


dataPATHpoisson   = "../../../../../../localdata/Derick/stuart_data/Darcy_421/fUG_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
dataPathDarcyPWC  = "../../../../../../localdata/Derick/stuart_data/Darcy_421/new_aUP_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
dataPathDarcyLN   = "../../../../../../localdata/Derick/stuart_data/Darcy_421/aUL_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
dataPathNavierS   = "../../../../../../localdata/Derick/stuart_data/Darcy_421/NavierStokes_TrainData=1000_TestData=5000_Resolution=64X64_Domain=[0,1]X[0,1].hdf5"
dataPathStructM   = "../../../../../../localdata/Derick/stuart_data/Darcy_421/StructuralMechanics_TrainData=1000_TestData=5000_Resolution=41X41_Domain=[0,1]X[0,1].hdf5"
dataPathHelmholtz = "../../../../../../localdata/Derick/stuart_data/Darcy_421/Helmholtz_TrainData=1000_TestData=5000_Resolution=101X101_Domain=[0,1]X[0,1].hdf5"
dataPathAdvection = "../../../../../../localdata/Derick/stuart_data/Darcy_421/Advection_TrainData=1000_TestData=5000_Resolution=200_Domain=[0,1].hdf5"
listdataPATH      = [dataPathAdvection]#dataPathStructM, dataPathHelmholtz, dataPathNavierS]#, dataPathAdvection]#[dataPathDarcyPWC, dataPATHpoisson, dataPathDarcyLN]

pcaPATHpoisson    = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/pca/poisson/UnitGaussianNormalizer/'
pcaPATHDarcyPWC   = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/pca/darcyPWC/UnitGaussianNormalizer/'
pcaPATHDarcyLN    = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/pca/darcyLN/UnitGaussianNormalizer/'
pcaPathNavierS   = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/pca/navierStokes/UnitGaussianNormalizer/'
pcaPathStructM   = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/pca/structuralMechanics/UnitGaussianNormalizer/'
pcaPathHelmholtz = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/pca/helmholtz/UnitGaussianNormalizer/'
pcaPathAdvection = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/pca/advection/UnitGaussianNormalizer/'
listPCAPATH       = [pcaPathAdvection]#pcaPathStructM, pcaPathHelmholtz, pcaPathNavierS]#, pcaPathAdvection]#[pcaPathDarcyPWC, pcaPathpoisson, pcaPathDarcyLN]

ntrain = 1000
ntest = 5000

for dataPATH, pcaPATH in zip(listdataPATH , listPCAPATH):
    if not os.path.exists(dataPATH):
        os.makedirs(dataPATH)
    if not os.path.exists(pcaPATH):
        os.makedirs(pcaPATH)

    if dataPATH == dataPathNavierS:
        res1 = 64
        reslist = [res1-1]
    elif dataPATH == dataPathStructM: 
        res1 = 41
        reslist = [res1-1]
    elif dataPATH == dataPathHelmholtz: 
        res1 = 101
        reslist = [res1-1]
    elif dataPATH == dataPathAdvection: 
        res1 = 200
        reslist = [res1-1]
    else:
        res1 = 512
        reslist = [32, 64, 128, 256, 512]

    X_train, Y_train, X_test, Y_test = readtoArray(dataPATH, 1000, 5000, Nx = res1, Ny = res1)#, 1000, 5000, Nx = 512, Ny = 512)

    print ("Converting dataset to numpy array.")
    tt = time.time()
    X_train0 = np.array(X_train)
    Y_train0 = np.array(Y_train)
    print ("    Conversion completed after %.4f seconds"%(time.time()-tt))

    for res in reslist: #[32, 128, 256, 512]: # 
        res = res + 1
        print ("Subsampling dataset to the required resolution.", res)
        tt = time.time()
        
        if dataPATH == dataPathAdvection: 
            X_train1 = SubSample1D(X_train0, res)
            Y_train1 = SubSample1D(Y_train0, res)
            print ("    Subsampling completed after %.4f seconds"%(time.time()-tt))

            print ("Taking out the required train/test size.")
            tt = time.time()
            x_train1 = X_train1[:ntrain, :].reshape(ntrain, -1)
            y_train1 = Y_train1[:ntrain, :].reshape(ntrain, -1)
            print ("    Taking completed after %.4f seconds"%(time.time()-tt))
        else:
            X_train1 = SubSample(X_train0, res, res)
            Y_train1 = SubSample(Y_train0, res, res)
            print ("    Subsampling completed after %.4f seconds"%(time.time()-tt))

            print ("Taking out the required train/test size.")
            tt = time.time()
            x_train1 = X_train1[:ntrain, :, :].reshape(ntrain, -1)
            y_train1 = Y_train1[:ntrain, :, :].reshape(ntrain, -1)
            print ("    Taking completed after %.4f seconds"%(time.time()-tt))

        for dX in [30, 50, 70, 100, 150, 200]: # 

            print ("Obtaining the PCA functions")   
            tt = time.time()
             
            pcaX = PCA(n_components = dX, random_state = 0).fit(x_train1)
            pcaY = PCA(n_components = dX, random_state = 0).fit(y_train1) 
            torch.save(pcaX, pcaPATH+"param_pca-%s_res-%s_d-%s-ntrain.pt"%(res-1, dX, ntrain))
            torch.save(pcaY, pcaPATH+"solut_pca-%s_res-%s_d-%s-ntrain.pt"%(res-1, dX, ntrain)) 

            print ("    Obtaining the PCA functions done after %s seconds"%(time.time()-tt))


            x_train = pcaX.transform(x_train1)
            y_train = pcaY.transform(y_train1)


            x_normalizer = UnitGaussianNormalizer(torch.from_numpy(x_train).float().to(device))
            y_normalizer = UnitGaussianNormalizer(torch.from_numpy(y_train).float().to(device))
            torch.save(x_normalizer, pcaPATH+"param_normalizer-%s_res-%s_d-%s-ntrain.pt"%(res-1, dX, ntrain))
            torch.save(y_normalizer, pcaPATH+"solut_normalizer-%s_res-%s_d-%s-ntrain.pt"%(res-1, dX, ntrain))

                    #NEW
            # x_normalizer = GaussianNormalizer(torch.from_numpy(x_train).float().to(device))
            # y_normalizer = GaussianNormalizer(torch.from_numpy(y_train).float().to(device))
            # torch.save(x_normalizer, pcaPATH+"param_gaussian-normalizer-%s_res-%s_d-%s-ntrain.pt"%(res-1, dX, ntrain))
            # torch.save(y_normalizer, pcaPATH+"solut_gaussian-normalizer-%s_res-%s_d-%s-ntrain.pt"%(res-1, dX, ntrain))

            x_normalizer = RangeNormalizer(torch.from_numpy(x_train).float().to(device))
            y_normalizer = RangeNormalizer(torch.from_numpy(y_train).float().to(device))
            torch.save(x_normalizer, pcaPATH+"param_range-normalizer-%s_res-%s_d-%s-ntrain.pt"%(res-1, dX, ntrain))
            torch.save(y_normalizer, pcaPATH+"solut_range-normalizer-%s_res-%s_d-%s-ntrain.pt"%(res-1, dX, ntrain))