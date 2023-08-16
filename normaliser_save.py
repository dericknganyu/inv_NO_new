import numpy as np
import torch

from utilities.utils import *

import time
from utilities.readData import readtoArray

torch.manual_seed(0)
np.random.seed(0)


dataPATHpoisson   = "../../../../../../localdata/Derick/stuart_data/Darcy_421/fUG_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
dataPathDarcyPWC  = "../../../../../../localdata/Derick/stuart_data/Darcy_421/new_aUP_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
dataPathDarcyLN   = "../../../../../../localdata/Derick/stuart_data/Darcy_421/aUL_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
dataPathNavierS   = "../../../../../../localdata/Derick/stuart_data/Darcy_421/NavierStokes_TrainData=1000_TestData=5000_Resolution=64X64_Domain=[0,1]X[0,1].hdf5"
dataPathStructM   = "../../../../../../localdata/Derick/stuart_data/Darcy_421/StructuralMechanics_TrainData=1000_TestData=5000_Resolution=41X41_Domain=[0,1]X[0,1].hdf5"
dataPathHelmholtz = "../../../../../../localdata/Derick/stuart_data/Darcy_421/Helmholtz_TrainData=1000_TestData=5000_Resolution=101X101_Domain=[0,1]X[0,1].hdf5"
dataPathAdvection = "../../../../../../localdata/Derick/stuart_data/Darcy_421/Advection_TrainData=1000_TestData=5000_Resolution=200_Domain=[0,1].hdf5"
listdataPATH      = [dataPathStructM, dataPathHelmholtz, dataPathNavierS]#, dataPathAdvection]#[dataPathDarcyPWC, dataPATHpoisson, dataPathDarcyLN]

normPATHpoisson   = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/normalisers/poisson/UnitGaussianNormalizer/'
normPATHDarcyPWC  = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/normalisers/darcyPWC/UnitGaussianNormalizer/'
normPATHDarcyLN   = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/normalisers/darcyLN/UnitGaussianNormalizer/'
normPathNavierS   = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/normalisers/navierStokes/UnitGaussianNormalizer/'
normPathStructM   = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/normalisers/structuralMechanics/UnitGaussianNormalizer/'
normPathHelmholtz = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/normalisers/helmholtz/UnitGaussianNormalizer/'
normPathAdvection = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/normalisers/advection/UnitGaussianNormalizer/'
listNORMPATH      = [normPathStructM, normPathHelmholtz, normPathNavierS]#, normPathAdvection]#[normPATHDarcyPWC, normPATHpoisson, normPATHDarcyLN]

ntrain = 1000
ntest = 5000

for dataPATH, normPATH in zip(listdataPATH , listNORMPATH):
    if dataPATH == dataPathNavierS:
        res1 = 64
        reslist = [res1-1]
    elif dataPATH == dataPathStructM: 
        res1 = 41
        reslist = [res1-1]
    elif dataPATH == dataPathHelmholtz: 
        res1 = 101
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

    for res in reslist:
        res = res + 1
        print ("Subsampling dataset to the required resolution.", res)
        tt = time.time()
        X_train = SubSample(X_train0, res, res)
        Y_train = SubSample(Y_train0, res, res)
        print ("    Subsampling completed after %.4f seconds"%(time.time()-tt))

        new_res = closest_power(res)
        X_train_cs = CubicSpline3D(X_train, new_res, new_res)
        Y_train_cs = CubicSpline3D(Y_train, new_res, new_res)

        print ("Taking out the required train/test size.")
        tt = time.time()
        x_train = torch.from_numpy(X_train[:ntrain, :, :]).float()
        y_train = torch.from_numpy(Y_train[:ntrain, :, :]).float()

        x_train_cs = torch.from_numpy(X_train_cs[:ntrain, :, :]).float()
        y_train_cs = torch.from_numpy(Y_train_cs[:ntrain, :, :]).float()
        print ("    Taking completed after %.4f seconds"%(time.time()-tt))
        print("...")


        x_normalizer = UnitGaussianNormalizer(x_train)
        y_normalizer = UnitGaussianNormalizer(y_train)

        x_normalizer_cs = UnitGaussianNormalizer(x_train_cs)
        y_normalizer_cs = UnitGaussianNormalizer(y_train_cs)

        torch.save(x_normalizer, normPATH+"param_normalizer-%s-res-%s-ntrain.pt"%(res-1, ntrain))
        torch.save(y_normalizer, normPATH+"solut_normalizer-%s-res-%s-ntrain.pt"%(res-1, ntrain))

        torch.save(x_normalizer_cs, normPATH+"param_normalizer-cs%s-res-%s-ntrain.pt"%(new_res-1, ntrain))
        torch.save(y_normalizer_cs, normPATH+"solut_normalizer-cs%s-res-%s-ntrain.pt"%(new_res-1, ntrain))
        
        #torch.save(x_normalizer, "normalisers/inv-%s-y_normalizer.pt"%(res-1))
        #torch.save(y_normalizer, "normalisers/inv-%s-x_normalizer.pt"%(res-1))