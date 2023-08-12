from pkgutil import ModuleInfo
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
 
params               = dict()
params["xmin"]       = 0
params["xmax"]       = 1
params["ymin"]       = 0
params["ymax"]       = 1


parser = argparse.ArgumentParser(description='For inputing resolution')

#Adding required parser argument
parser.add_argument('--res', default=512, type=int, help='Specify redolution')
parser.add_argument('--ep', default=10000, type = int, help='epochs')#
parser.add_argument('--lr', default=0.1, type = float, help='learning rate')#
parser.add_argument('--wd', default=1e-5, type = float, help='weight decay')
parser.add_argument('--no', default='pino', type=str, help='Specify Neural operator: pino, fno, ufno, pcalin, pcann')
parser.add_argument('--pb', default='darcyPWC', type=str, help='Specify Problem: poison, darcyPWC, darcyLN')
parser.add_argument('--gm', default=0.5, type = float, help='gamma')#
parser.add_argument('--ss', default=2000, type = int, help='step size')#
parser.add_argument('--ps', default=2000, type = int, help='plot step')#
parser.add_argument('--init', default='random', type=str, help='Specify initialisation: random, choice, ...')
parser.add_argument('--nr', default=0.0, type = float, help='noise ratio')#
parser.add_argument('--ntest', default=100, type = int, help='number of testing samples')#
parser.add_argument('--bs', default=10, type = int, help='batch_size')#




#parsing
args = parser.parse_args()
epochs = args.ep
res = args.res + 1
learning_rate = args.lr
no = args.no
pb = args.pb
wd = args.wd
gamma = args.gm
step_size = args.ss
plot_step = epochs/5 #args.ps
write_step = epochs/200
init = args.init
noise_ratio = args.nr
ntest= args.ntest
batch_size = args.bs

if no == 'fno':
    modes = 12
    width = 32
    model = FNO2d(modes, modes, width).cuda()
    if pb == 'darcyPWC':
        MODELS = {16:'',
                32:'',
                64:'/home/derick/Documents/FNO/darcy_flow_mu_P/main/files/inv/last_model_inv_065~res_0.166186~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220722-230511.pt',
                128:'',
                256:'',
                512:''}
    if pb == 'poisson':
        MODELS = {16:'',
                32:'',
                64:'/home/derick/Documents/FNO/poisson-problem/zmain-paper-data/files/last_model_inv_065~res_0.029854~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_1000~epochs_20220629-131403.pt',
                128:'',
                256:'',
                512:''}
    if pb == 'darcyLN':
        MODELS = {16:'',
                32:'',
                64:'/home/derick/Documents/FNO/darcy_flow_mu_L/main/files/last_model_inv_065~res_0.029069~RelL2TestError_1000~ntrain_100~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220225-224304.pt',
                128:'',
                256:'',
                512:''}
    if pb == 'navierStokes':
        res = 64
        MODELS = {63:'/home/derick/Documents/FNO/navier_stokes/main/files/inv/last_model_inv_064~res_0.004186~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20230712-192004.pt'}
    if pb == 'helmholtz':
        res = 101
        MODELS = {100:'/home/derick/Documents/FNO/helmholtz/main/files/inv/last_model_inv_101~res_0.00467~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20230712-213904.pt'}
    if pb == 'structuralMechanics':
        res = 41
        MODELS = {40:'/home/derick/Documents/FNO/structural_mechanics/main/files/inv/last_model_inv_021~res_0.21811~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20230714-123804.pt'}
    saved_model = MODELS[res-1]


if no == 'pino':
    modes = 12
    width = 32
    model = FNO2d(modes, modes, width).cuda()
    if pb == 'darcyPWC':
        MODELS = {16:'',
                32:'',
                64:'/home/derick/Documents/PINO/darcy_flow_mu_P/files/inv/last_model_res=65_0.170276-relErr-darcyPWC-pino-0.0-noise.pt',
                128:'',
                256:'',
                512:''}
    if pb == 'poisson':
        MODELS = {16:'',
                32:'',
                64:'/home/derick/Documents/PINO/poisson/files/inv/last_model_NLC_inv_065~res_0.026256~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220907-142514.pt',
                128:'',
                256:'',
                512:''}
    if pb == 'darcyLN':
        MODELS = {16:'',
                32:'',
                64:'',
                128:'',
                256:'',
                512:''}
    if pb == 'navierStokes':
        res = 64
        MODELS = {63:''}
    if pb == 'helmholtz':
        res = 101
        MODELS = {100:''}
    if pb == 'structuralMechanics':
        res = 41
        MODELS = {40:''}
    saved_model = MODELS[res-1]

if no == 'ufno':
    modes = 12
    width = 32
    if pb == 'darcyPWC':
        MODELS = {16:'',
                32:'',
                64:'/home/derick/Documents/U-FNO/darcy_flow_mu_P/files/inv/last_model_inv_065~res_0.092811~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220723-050037.pt',
                128:'',
                256:'',
                512:''}
    if pb == 'poisson':
        MODELS = {16:'',
                32:'',
                64:'/home/derick/Documents/U-FNO/poisson/files/inv/last_model_inv_065~res_0.029157~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220805-031402.pt',
                128:'',
                256:'',
                512:''}
    if pb == 'darcyLN':
        MODELS = {16:'',
                32:'',
                64:'/home/derick/Documents/U-FNO/darcy_flow_mu_L/files/inv/last_model_inv_065~res_0.019607~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220720-201906.pt',
                128:'',
                256:'',
                512:''}
    if pb == 'navierStokes':
        res = 64
        MODELS = {63:'/home/derick/Documents/U-FNO/navier_stokes/files/inv/last_model_inv_064~res_0.01047~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20230720-005051.pt'}
    if pb == 'helmholtz':
        res = 101
        MODELS = {100:'/home/derick/Documents/U-FNO/helmholtz/files/inv/last_model_inv_101~res_0.001322~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20230720-010156.pt'}
    if pb == 'structuralMechanics':
        res = 41
        MODELS = {40:'/home/derick/Documents/U-FNO/structural_mechanics/files/inv/last_model_inv_041~res_0.165815~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20230719-172639.pt'}
    saved_model = MODELS[res-1]
    padding = round_to_multiple(res, 2**3, direction='up') - res 
    model = UFNO2d_modif(modes, modes, width, padding).cuda()
    saved_model = MODELS[res-1]

if no == 'mwt':
    ich = 3
    initializer = get_initializer('xavier_normal')
    model = MWT2d(ich, 
                alpha = 12,
                c = 4,
                k = 4, 
                base = 'legendre', # 'chebyshev'
                nCZ = 4,
                L = 0,
                initializer = initializer,
                ).to(device)
    if pb == 'darcyPWC':
        MODELS = {16:'',
                32:'',
                64:'/home/derick/Documents/MWT/darcy_flow_mu_P/files/inv/last_model_cs_inv_065~res_0.104698~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220811-063212.pt',
                128:'',
                256:'',
                512:''}
    if pb == 'poisson':
        MODELS = {16:'',
                32:'',
                64:'/home/derick/Documents/MWT/poisson/files/inv/last_model_cs_inv_065~res_0.035693~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220811-065018.pt',
                128:'',
                256:'',
                512:''}
    if pb == 'darcyLN':
        MODELS = {16:'',
                32:'',
                64:'/home/derick/Documents/MWT/darcy_flow_mu_L/files/inv/last_model_064~res_0.019985~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220806-095817.pt',
                128:'',
                256:'',
                512:''}
    if pb == 'navierStokes':
        res = 64
        MODELS = {63:'/home/derick/Documents/MWT/navier_stokes/files/inv/last_model_cs_inv_064~res_0.004158~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20230717-194058.pt'}
    if pb == 'helmholtz':
        res = 101
        MODELS = {100:'/home/derick/Documents/MWT/helmholtz/files/inv/last_model_cs_inv_101~res_0.00467~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20230717-223246.pt'}
    if pb == 'structuralMechanics':
        res = 41
        MODELS = {40:'/home/derick/Documents/MWT/structural_mechanics/files/inv/last_model_cs_inv_041~res_0.163985~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20230717-194115.pt'}
    saved_model = MODELS[res-1]

if no == "pcalin":
    if pb == 'darcyPWC':
        dX = 100
        dY = 100
        params = {}
        params["layers"] = [dX , dY]
        model = pcalin(params).cuda()
        MODELS = {16:'',
                32:'',
                64:'/home/derick/Documents/PCANN/darcy_flow_mu_P/pcalin-inv/models/last_model_065~res_0.219101~RelL2TestError_100~rd_1000~ntrain_5000~ntest_200~BatchSize_0.001~LR_0.15~Reg_0.01~gamma_2500~Step_8000~epochs_20221017-222658-191947.pt',
                128:'',
                256:'',
                512:''}
    if pb == 'poisson':
        dX = 250
        dY = 250
        params = {}
        params["layers"] = [dX , dY]
        model = pcalin(params).cuda()
        MODELS = {16:'',
                32:'',
                64:'/home/derick/Documents/PCANN/poisson_paper_data/pcalin-inv/models/last_model_065~res_0.030348~RelL2TestError_250~rd_1000~ntrain_5000~ntest_1000~BatchSize_0.001~LR_0.1~Reg_0.01~gamma_2500~Step_8000~epochs_20220820-005036-097452.pt',
                128:'',
                256:'',
                512:''}
    if pb == 'darcyLN':
        dX = 250
        dY = 250
        params = {}
        params["layers"] = [dX , dY]
        model = pcalin(params).cuda()
        MODELS = {16:'',
                32:'',
                64:'',
                128:'',
                256:'',
                512:''}
    if pb == 'navierStokes':
        res = 64
        MODELS = {63:''}
    if pb == 'helmholtz':
        res = 101
        MODELS = {100:''}
    if pb == 'structuralMechanics':
        res = 41
        MODELS = {40:''}
    saved_model = MODELS[res-1]

if no == "pcann":
    if pb == 'darcyPWC':
        dX = 30
        dY = 30
        p_drop = 0.01
        model = pcann_snn(in_features=dX, out_features=dY, p_drop=p_drop, use_selu=True).cuda()
        MODELS = {16:'',
                32:'',
                64:'/home/derick/Documents/PCANN/darcy_flow_mu_P/pcann-inv/models/last_model_UGN_inv_sgd_065~res_0.098339~RelL2TestError_30~rd_0.01~pdrop_1000~ntrain_5000~ntest_100~BatchSize_0.0001000000~LR_0.1000~Reg_0.0100~gamma_2500~Step_10000~epochs_20221017-234203-430020.pt',
                128:'',
                256:'',
                512:''}
    if pb == 'poisson':
        dX = 200
        dY = 200
        p_drop = 0.001
        model = pcann_snn(in_features=dX, out_features=dY, p_drop=p_drop, use_selu=True).cuda()
        MODELS = {16:'',
                32:'',
                64:'/home/derick/Documents/PCANN/poisson_paper_data/pcann-inv/models/last_model_inv_sgd_065~res_0.098880~RelL2TestError_200~rd_0.001~pdrop_1000~ntrain_5000~ntest_500~BatchSize_0.0000100000~LR_0.1000~Reg_0.1000~gamma_2500~Step_10000~epochs_20220912-182046-914270.pt',
                128:'',
                256:'',
                512:''}
    if pb == 'darcyLN':
        dX = 100
        dY = 100
        p_drop = 0
        model = pcann_snn(in_features=dX, out_features=dY, p_drop=p_drop, use_selu=True).cuda()
        MODELS = {16:'',
                32:'',
                64:'',
                128:'',
                256:'',
                512:''}
    if pb == 'navierStokes':
        res = 64
        MODELS = {63:''}
    if pb == 'helmholtz':
        res = 101
        MODELS = {100:''}
    if pb == 'structuralMechanics':
        res = 41
        MODELS = {40:''}
    saved_model = MODELS[res-1]


grids = []
grids.append(np.linspace(0, 1, res))
grids.append(np.linspace(0, 1, res))
grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
grid = grid.reshape(1,res,res,2)
grid = torch.tensor(grid, dtype=torch.float)


model.load_state_dict(torch.load(saved_model))    
model.eval()

myloss = LpLoss(size_average=False)

grid = grid.cuda()
mollifier = torch.sin(np.pi*grid[...,0]) * torch.sin(np.pi*grid[...,1]) * 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  

#---------------------------------------------------------------------------------------------
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)


if pb == 'darcyPWC':
    fileName_ex = "datasets/new_aUP_Square_TrainData=1_TestData=1_Resolution=513X513_Domain=[0,1]X[0,1].hdf5" 
    fileName    = "/localdata/Derick/stuart_data/Darcy_421/new_aUP_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
    normPATH    = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/normalisers/darcyPWC/UnitGaussianNormalizer/'
    pcaPATH     = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/pca/darcyPWC/UnitGaussianNormalizer/'
if pb == 'darcyLN': 
    fileName_ex = "datasets/aUL_Square_TrainData=1_TestData=1_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
    fileName    = "/localdata/Derick/stuart_data/Darcy_421/aUL_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
    normPATH    = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/normalisers/darcyLN/UnitGaussianNormalizer/'
    pcaPATH     = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/pca/darcyLN/UnitGaussianNormalizer/'
if pb == 'poisson': 
    fileName_ex = "datasets/fUG_Square_TrainData=1_TestData=1_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
    fileName    = "/localdata/Derick/stuart_data/Darcy_421/fUG_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
    normPATH    = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/normalisers/poisson/UnitGaussianNormalizer/'
    pcaPATH     = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/pca/poisson/UnitGaussianNormalizer/'
if pb == 'navierStokes': 
    fileName_ex = "datasets/NavierStokes_TrainData=1_TestData=1_Resolution=64X64_Domain=[0,1]X[0,1].hdf5"
    fileName    = "/localdata/Derick/stuart_data/Darcy_421/NavierStokes_TrainData=1000_TestData=5000_Resolution=64X64_Domain=[0,1]X[0,1].hdf5"
    normPATH    = '/localdata/Derick/stuart_data/Darcy_421/operators/normalisers/navierStokes/UnitGaussianNormalizer/'
    pcaPATH     = ''
if pb == 'helmholtz': 
    fileName_ex = "datasets/Helmholtz_TrainData=1_TestData=1_Resolution=101X101_Domain=[0,1]X[0,1].hdf5"
    fileName    = "/localdata/Derick/stuart_data/Darcy_421/Helmholtz_TrainData=1000_TestData=5000_Resolution=101X101_Domain=[0,1]X[0,1].hdf5"
    normPATH    = '/localdata/Derick/stuart_data/Darcy_421/operators/normalisers/helmholtz/UnitGaussianNormalizer/'
    pcaPATH     = ''
if pb == 'structuralMechanics': 
    fileName_ex = "datasets/StructuralMechanics_TrainData=1_TestData=1_Resolution=41X41_Domain=[0,1]X[0,1].hdf5"
    fileName    = "/localdata/Derick/stuart_data/Darcy_421/StructuralMechanics_TrainData=1000_TestData=5000_Resolution=41X41_Domain=[0,1]X[0,1].hdf5"
    normPATH    = '/localdata/Derick/stuart_data/Darcy_421/operators/normalisers/structuralMechanics/UnitGaussianNormalizer/'
    pcaPATH     = ''



_, _, X_test, Y_test = readtoArray(fileName, 1, 1, Nx = 512, Ny = 512)
print ("Converting dataset to numpy array and subsamping.")
tt = time.time()
X_test = SubSample(np.array(X_test[ :ntest, :, :]), res, res)
Y_test = SubSample(np.array(Y_test[ :ntest, :, :]), res, res)

print ("    Conversion completed after %.2f minutes"%((time.time()-tt)/60))

print ("Adding noise.")
tt = time.time()
useless = np.zeros((1, res, res))#Y_TRAIN[ :2, :, :]
_, _, _, Y_test_noisy = add_noise((useless, useless, useless, Y_test), noise_ratio)

print ("    Adding noise completed after %.2f minutes"%((time.time()-tt)/60))

ntrain = 1000
x_normalizer = torch.load(normPATH+"param_normalizer-%s-res-%s-ntrain.pt"%(res-1, ntrain)) # UnitGaussianNormalizer(x)
y_normalizer = torch.load(normPATH+"solut_normalizer-%s-res-%s-ntrain.pt"%(res-1, ntrain)) # UnitGaussianNormalizer(y)

x = torch.from_numpy(X_test).float()#.cuda()
y = torch.from_numpy(Y_test).float()#.cuda()
y_noisy = torch.from_numpy(Y_test_noisy).float()#.cuda()

if no == 'mwt':
    old_res = res
    res = closest_power(res)
    X_test0 = CubicSpline3D(X_test, res, res)
    Y_test0 = CubicSpline3D(Y_test, res, res)
    Y_test0_noisy = CubicSpline3D(Y_test_noisy, res, res)

    grids_mwt = []
    grids_mwt.append(np.linspace(0, 1, res))
    grids_mwt.append(np.linspace(0, 1, res))
    grid_mwt = np.vstack([xx.ravel() for xx in np.meshgrid(*grids_mwt)]).T
    grid_mwt = grid_mwt.reshape(1,res,res,2)
    grid_mwt = torch.tensor(grid_mwt, dtype=torch.float).cuda()

    x_normalizer = torch.load(normPATH+"param_normalizer-cs%s-res-%s-ntrain.pt"%(res-1, ntrain)) # UnitGaussianNormalizer(x)
    y_normalizer = torch.load(normPATH+"solut_normalizer-cs%s-res-%s-ntrain.pt"%(res-1, ntrain)) # UnitGaussianNormalizer(y)

    # x = torch.from_numpy(X_test0).float()#.cuda()
    y = torch.from_numpy(Y_test0).float()#.cuda()
    y_noisy = torch.from_numpy(Y_test0_noisy).float()#.cuda()

elif no == 'pcalin' or no == 'pcann':
    if pb == 'darcyPWC':
        batch_size = 200  if no == 'pcalin' else 100
    if pb == 'poisson':
        batch_size = 1000 if no == 'pcalin' else 500

    old_res = res
    res = dX
    X_test = X_test.reshape(ntest, -1)
    Y_test = Y_test.reshape(ntest, -1)
    Y_test_noisy = Y_test_noisy.reshape(ntest, -1)

    pcaX = torch.load(pcaPATH+"param_pca-%s_res-%s_d-%s-ntrain.pt"%(old_res-1, dX, ntrain)) #if no =='pcalin' else torch.load(pcaPATH+"full_param_pca-%s_res-%s_d-%s-ntrain.pt"%(old_res-1, dX, ntrain))
    pcaY = torch.load(pcaPATH+"solut_pca-%s_res-%s_d-%s-ntrain.pt"%(old_res-1, dX, ntrain)) #if no =='pcalin' else torch.load(pcaPATH+"full_solut_pca-%s_res-%s_d-%s-ntrain.pt"%(old_res-1, dX, ntrain)) 

    X_test0 = pcaX.transform(X_test)
    Y_test0 = pcaY.transform(Y_test)
    Y_test0_noisy = pcaY.transform(Y_test_noisy)

    old_x_normalizer = x_normalizer
    old_y_normalizer = y_normalizer

    x_normalizer = torch.load(pcaPATH+"param_normalizer-%s_res-%s_d-%s-ntrain.pt"%(old_res-1, dX, ntrain)) #if no =='pcalin' else torch.load(pcaPATH+"full_param_normalizer-%s_res-%s_d-%s-ntrain.pt"%(old_res-1, dX, ntrain)) 
    y_normalizer = torch.load(pcaPATH+"solut_normalizer-%s_res-%s_d-%s-ntrain.pt"%(old_res-1, dX, ntrain)) #if no =='pcalin' else torch.load(pcaPATH+"full_solut_normalizer-%s_res-%s_d-%s-ntrain.pt"%(old_res-1, dX, ntrain))  

    x = torch.from_numpy(X_test0).float().cuda()
    y = torch.from_numpy(Y_test0).float().cuda()
    y_noisy = torch.from_numpy(Y_test0_noisy).float().cuda()


y = y_normalizer.encode(y)
y_noisy = y_normalizer.encode(y_noisy)
x_normalizer.cuda()
y_normalizer.cuda()


test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y, y_noisy), batch_size=batch_size, shuffle=False)

ACCURACY = 0
test_l2 = 0
test_l2_noisy = 0
total = len(test_loader)
i = 0
for x, y, y_noisy in test_loader:
    i += 1
    print("Running %s of %s"%(i, total))
    x, y, y_noisy = x.cuda(), y.cuda(), y_noisy.cuda()
    
    if no == 'mwt':
        out = model(torch.cat([y.reshape(batch_size, res, res, 1), grid_mwt.repeat(batch_size,1,1,1)], dim=3)).reshape(batch_size, res, res)
        out_noisy = model(torch.cat([y_noisy.reshape(batch_size, res, res, 1), grid_mwt.repeat(batch_size,1,1,1)], dim=3)).reshape(batch_size, res, res)
    elif no == 'pcalin' or no == 'pcann':
        out = model(y)
        out_noisy = model(y_noisy)
    else: 
        out = model(y.reshape(batch_size, res, res, 1)).reshape(batch_size, res, res) 
        out_noisy = model(y_noisy.reshape(batch_size, res, res, 1)).reshape(batch_size, res, res)

    out = x_normalizer.decode(out)
    out_noisy = x_normalizer.decode(out_noisy)
    
    if no == 'pcalin' or no == 'pcann':
        x = pcaX.inverse_transform(x.detach().cpu().numpy().reshape(batch_size, -1)).reshape(batch_size, old_res, old_res)
        x = torch.from_numpy(x).float().to(device)

        out = pcaX.inverse_transform(out.detach().cpu().numpy().reshape(batch_size, -1)).reshape(batch_size, old_res, old_res)
        out = torch.from_numpy(out).float().to(device)

        out_noisy = pcaX.inverse_transform(out_noisy.detach().cpu().numpy().reshape(batch_size, -1)).reshape(batch_size, old_res, old_res)
        out_noisy = torch.from_numpy(out_noisy).float().to(device)
    
    if no == 'mwt':
        #print("Returning to orignial resolution to get test error based on it")
        out = CubicSpline3D(out.detach().cpu().numpy(), old_res, old_res)
        out = torch.from_numpy(out).float().to(device)

        out_noisy = CubicSpline3D(out_noisy.detach().cpu().numpy(), old_res, old_res)
        out_noisy = torch.from_numpy(out_noisy).float().to(device)

    test_l2 += myloss(out.view(batch_size,-1), x.view(batch_size,-1)).item()
    test_l2_noisy += myloss(out_noisy.view(batch_size,-1), x.view(batch_size,-1)).item()

test_l2 /= ntest
test_l2_noisy /= ntest
   
noise_ratio = 100*noise_ratio
if noise_ratio >= 1:
    noise_ratio = int(noise_ratio) 

directory = 'figures-backward'
if not os.path.exists(directory):
    os.makedirs(directory)
file = open(directory+'/Log.txt',"a")
file.write("Problem: %s NeuralOperator: %s NoiseRatio: %s NoiselessError: %.4f Noisy Error: %.4f\n"%(pb, no, noise_ratio, test_l2, test_l2_noisy))
ModelInfos = '%s-%s-%s-noise'%(pb, no, noise_ratio)

directory = 'figures-backward' + '/' + pb + '/' + no
if not os.path.exists(directory):
    os.makedirs(directory)
U_train, F_train, _, _ = readtoArray(fileName_ex, 1, 1, 512, 512)

print("Starting the Verification with Sampled Example")
tt = time.time()

if no == 'fno' or no == 'ufno' or no == 'pino':  
    U_FDM = SubSample(np.array(U_train), res, res)[0]
    F_train = SubSample(np.array(F_train), res, res)
    useless = np.zeros((1, res, res))#Y_TRAIN[ :2, :, :]
    _, _, _, F_train_noisy = add_noise((useless, useless, useless, F_train), noise_ratio/100)

    print("      Doing FNO on Example...")
    tt = time.time()
    ff = torch.from_numpy(F_train).float().cuda()
    ff_noisy = torch.from_numpy(F_train_noisy).float().cuda()

    ff = y_normalizer.encode(ff)
    ff_noisy = y_normalizer.encode(ff_noisy)

    ff = ff.reshape(1,res,res,1)
    ff_noisy = ff_noisy.reshape(1,res,res,1)

    U_NO = x_normalizer.decode(model(ff).reshape(1, res, res)).detach().cpu().numpy()
    U_NO_noisy = x_normalizer.decode(model(ff_noisy).reshape(1, res, res)).detach().cpu().numpy()

    U_NO = U_NO[0] 
    U_NO_noisy = U_NO_noisy[0]
    print("            FNO completed after %.4f secondes"%(time.time()-tt))

if no == 'mwt':  
    U_FDM = SubSample(np.array(U_train), old_res, old_res)[0]
    F_train = SubSample(np.array(F_train), old_res, old_res)
    useless = np.zeros((1, old_res, old_res))#Y_TRAIN[ :2, :, :]
    _, _, _, F_train_noisy = add_noise((useless, useless, useless, F_train), noise_ratio/100)

    print("      Doing MWT on Example...")
    tt = time.time()
    F_train_cs = CubicSpline3D(F_train, res, res)
    F_train_cs_noisy = CubicSpline3D(F_train_noisy, res, res)

    ff = torch.from_numpy(F_train_cs).float().cuda()
    ff_noisy = torch.from_numpy(F_train_cs_noisy).float().cuda()

    ff = y_normalizer.encode(ff)
    ff_noisy = y_normalizer.encode(ff_noisy)

    ff = torch.cat([ff.reshape(1,res,res,1), grid_mwt.repeat(1,1,1,1)], dim=3).cuda() #ff.reshape(1,res,res,1).cuda()#
    ff_noisy = torch.cat([ff_noisy.reshape(1,res,res,1), grid_mwt.repeat(1,1,1,1)], dim=3).cuda() #ff.reshape(1,res,res,1).cuda()#

    U_NO = model(ff)
    U_NO_noisy = model(ff_noisy)

    U_NO = U_NO.reshape(1, res, res)
    U_NO_noisy = U_NO_noisy.reshape(1, res, res)

    U_NO = x_normalizer.decode(U_NO)
    U_NO_noisy = x_normalizer.decode(U_NO_noisy)

    U_NO = U_NO.detach().cpu().numpy()
    U_NO_noisy = U_NO_noisy.detach().cpu().numpy()

    U_NO = CubicSpline3D(U_NO, old_res, old_res)
    U_NO_noisy = CubicSpline3D(U_NO_noisy, old_res, old_res)

    U_NO = U_NO[0] 
    U_NO_noisy = U_NO_noisy[0] 

    print("            MWT completed after %s"%(time.time()-tt))

if no == 'pcalin' or no == 'pcann':
    F_train = SubSample(F_train, old_res, old_res)
    U_train = SubSample(U_train, old_res, old_res)
    U_FDM = np.array(U_train[0])

    useless = np.zeros((1, old_res, old_res))#Y_TRAIN[ :2, :, :]
    _, _, _, F_train_noisy = add_noise((useless, useless, useless, F_train), noise_ratio/100)

    ff = np.array(F_train[0])
    ff_noisy = np.array(F_train_noisy[0])    

    print("      Doing PCANN on Example...")
    tt = time.time()

    inPCANN = pcaY.transform(ff.reshape(1, -1))
    inPCANN_noisy = pcaY.transform(ff_noisy.reshape(1, -1))

    inPCANN = torch.from_numpy(inPCANN).float().to(device)
    inPCANN_noisy = torch.from_numpy(inPCANN_noisy).float().to(device)

    inPCANN = model(y_normalizer.encode(inPCANN))
    inPCANN_noisy = model(y_normalizer.encode(inPCANN_noisy))

    inPCANN = x_normalizer.decode(inPCANN) #changed!!
    inPCANN_noisy = x_normalizer.decode(inPCANN_noisy) #changed!!

    inPCANN = inPCANN.detach().cpu().numpy()
    inPCANN_noisy = inPCANN_noisy.detach().cpu().numpy()

    U_NO = pcaX.inverse_transform(inPCANN).reshape(old_res, old_res)
    U_NO_noisy = pcaX.inverse_transform(inPCANN_noisy).reshape(old_res, old_res)
    print("            PCANN completed after %s"%(time.time()-tt))

myLoss = LpLoss(size_average=False)
print()
print("Ploting comparism of FDM and FNO Simulation results\n\n")
fig = plt.figure(figsize=((5+2)*4, (5+0.5)*2))

# if pb[0:5] == 'darcy':
#     fig.suptitle(r"Plot of $-\nabla \cdot (a(s) \nabla u(s)) = f(s), \partial \Omega = 0$ with $u|_{\partial \Omega}  = 0.$")
# if pb == 'poisson':
#     fig.suptitle("Plot of $- \Delta u = f(x, y)$ on $\Omega = ]0,1[ x ]0,1[$ with $u|_{\partial \Omega}  = 0.$")

colourMap = plt.cm.magma # parula() #plt.cm.jet #plt.cm.coolwarm

plt.subplot(2, 4, 1)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("Input")
plt.imshow(F_train[0], cmap=colourMap, extent=[0, 1, 0,1], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#(format=OOMFormatter(-5))

plt.subplot(2, 4, 2)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("Truth")
plt.imshow(U_FDM, cmap=colourMap, extent=[0, 1, 0,1], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#(format=OOMFormatter(-5))

plt.subplot(2, 4, 3)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title('inv'+no.upper())
plt.imshow(U_NO, cmap=colourMap, extent=[0, 1, 0,1], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#(format=OOMFormatter(-5))

plt.subplot(2, 4, 4)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("FDM-"+'inv'+no.upper()+", RelL2Err = "+str(round(myLoss.rel_single(U_NO, U_FDM).item(), 3)))
plt.imshow(np.abs(U_FDM - U_NO), cmap=colourMap, extent=[0, 1, 0,1], origin='lower', aspect = 'auto', norm=matplotlib.colors.LogNorm())#, vmin=0, vmax=1, )
plt.colorbar()#(format=OOMFormatter(-5))

plt.subplot(2, 4, 5)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("Noisy Input")
plt.imshow(F_train_noisy[0], cmap=colourMap, extent=[0, 1, 0, 1], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#(format=OOMFormatter(-5))

plt.subplot(2, 4, 6)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("Truth")
plt.imshow(U_FDM, cmap=colourMap, extent=[0, 1, 0, 1], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#(format=OOMFormatter(-5))

plt.subplot(2, 4, 7)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title('Noisy inv'+no.upper())
plt.imshow(U_NO_noisy, cmap=colourMap, extent=[0, 1, 0, 1], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#(format=OOMFormatter(-5))

plt.subplot(2, 4, 8)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("FDM-"+'inv'+no.upper()+", RelL2Err = "+str(round(myLoss.rel_single(U_NO_noisy, U_FDM).item(), 3)))
plt.imshow(np.abs(U_FDM - U_NO_noisy), cmap=colourMap, extent=[0, 1, 0, 1], origin='lower', aspect = 'auto', norm=matplotlib.colors.LogNorm())#, vmin=0, vmax=1, )
plt.colorbar()#(format=OOMFormatter(-5))

fig.tight_layout()   
plt.savefig(directory+'/compare-'+ModelInfos+'.png',dpi=500)

#plt.show()

fig = plt.figure(figsize=((5+1)*2, 5))


plt.subplot(1, 2, 1)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title('inv'+no.upper())
plt.imshow(U_NO_noisy, cmap=colourMap, extent=[0, 1, 0, 1], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

plt.subplot(1, 2, 2)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("Truth-inv"+no.upper()+", RelL2Err = "+str(round(myLoss.rel_single(U_NO_noisy, U_FDM).item(), 3)))
plt.imshow(np.abs(U_FDM - U_NO_noisy), cmap=colourMap, extent=[0, 1, 0, 1], origin='lower', aspect = 'auto', norm=matplotlib.colors.LogNorm())#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

fig.tight_layout()
plt.savefig(directory+'/'+ModelInfos+'.png',dpi=500)

#plt.show()
print ('Relative Error- Without Noise: %.6f | With Noise: %.6f'%(test_l2, test_l2_noisy))
