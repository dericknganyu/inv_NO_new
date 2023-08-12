import numpy as np 
import torch, time
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import pandas as pd
import math as m

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
parser.add_argument('--res', default=64, type=int, help='Specify redolution')
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

if no == 'fno':
    modes = 12
    width = 32
    model = FNO2d(modes, modes, width).cuda()
    if pb == 'darcyPWC':
        MODELS = {16:'/home/derick/Documents/FNO/darcy_flow_mu_P/main/files/last_model_017~res_0.033087~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220722-214218.pt',
                32:'/home/derick/Documents/FNO/darcy_flow_mu_P/main/files/last_model_033~res_0.01497~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220722-214037.pt',
                64:'/home/derick/Documents/FNO/darcy_flow_mu_P/main/files/last_model_065~res_0.011312~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220722-223521.pt',
                128:'/home/derick/Documents/FNO/darcy_flow_mu_P/main/files/last_model_129~res_0.010614~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220723-003817.pt',
                256:'/home/derick/Documents/FNO/darcy_flow_mu_P/main/files/last_model_257~res_0.010755~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220723-081532.pt',
                512:'/home/derick/Documents/FNO/darcy_flow_mu_P/main/files/last_model_513~res_0.010999~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220723-175027.pt'}
    if pb == 'poisson':
        MODELS = {16:'/home/derick/Documents/FNO/poisson-problem/zmain-paper-data/files/last_model_017~res_0.015515~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220716-014135.pt',
                32:'/home/derick/Documents/FNO/poisson-problem/zmain-paper-data/files/last_model_033~res_0.007198~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220628-104233.pt',
                64:'/home/derick/Documents/FNO/poisson-problem/zmain-paper-data/files/last_model_065~res_0.006563~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220628-134410.pt',
                128:'/home/derick/Documents/FNO/poisson-problem/zmain-paper-data/files/last_model_129~res_0.006073~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220628-162546.pt',
                256:'/home/derick/Documents/FNO/poisson-problem/zmain-paper-data/files/last_model_257~res_0.006714~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220629-024847.pt',
                512:'/home/derick/Documents/FNO/poisson-problem/zmain-paper-data/files/last_model_513~res_0.006618~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_600~epochs_20220708-030537.pt'}
    if pb == 'darcyLN':
        MODELS = {16:'/home/derick/Documents/FNO/darcy_flow_mu_L/main/files/last_model_017~res_0.004132~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220226-195929.pt',
                32:'/home/derick/Documents/FNO/darcy_flow_mu_L/main/files/last_model_033~res_0.003551~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220225-201127.pt',
                64:'/home/derick/Documents/FNO/darcy_flow_mu_L/main/files/last_model_065~res_0.003488~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220225-230803.pt',
                128:'/home/derick/Documents/FNO/darcy_flow_mu_L/main/files/last_model_129~res_0.003628~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220226-003411.pt',
                256:'/home/derick/Documents/FNO/darcy_flow_mu_L/main/files/last_model_257~res_0.003846~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220226-152815.pt',
                512:'/home/derick/Documents/FNO/darcy_flow_mu_L/main/files/last_model_inv_513~res_0.030209~RelL2TestError_1000~ntrain_100~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220226-173354.pt'}
    if pb == 'navierStokes':
        res = 64
        MODELS = {63:'/home/derick/Documents/FNO/navier_stokes/main/files/last_model_064~res_0.003295~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20230712-192040.pt'}
    if pb == 'helmholtz':
        res = 101
        MODELS = {100:'/home/derick/Documents/FNO/helmholtz/main/files/last_model_101~res_0.037424~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20230712-214049.pt'}
    if pb == 'structuralMechanics':
        res = 41
        MODELS = {40:'/home/derick/Documents/FNO/structural_mechanics/main/files/last_model_041~res_0.069594~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20230714-122957.pt'}
    saved_model = MODELS[res-1]



if no == 'pino':
    modes = 12
    width = 32
    model = FNO2d(modes, modes, width).cuda()
    if pb == 'darcyPWC':
        MODELS = {16:'',
                32:'',
                64:'/home/derick/Documents/PINO/darcy_flow_mu_P/files/last_model_NL_065~res_0.007829~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220907-200923.pt',
                128:'/home/derick/Documents/PINO/darcy_flow_mu_P/files/last_model_NL_129~res_0.006633~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220907-223431.pt',
                256:'/home/derick/Documents/PINO/darcy_flow_mu_P/files/last_model_NL_257~res_0.007138~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220908-034046.pt',
                512:'/home/derick/Documents/PINO/darcy_flow_mu_P/files/last_model_NL_513~res_0.008388~RelL2TestError_1000~ntrain_5000~ntest_5~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220908-141820.pt'}
    if pb == 'poisson':
        MODELS = {16:'',
                32:'',
                64:'/home/derick/Documents/PINO/poisson/files/last_model_NLC_065~res_0.00311~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220907-014319.pt',
                128:'/home/derick/Documents/PINO/poisson/files/last_model_NLC_129~res_0.002976~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220907-032831.pt',
                256:'/home/derick/Documents/PINO/poisson/files/last_model_NLC_257~res_0.003393~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220907-095708.pt',
                512:'/home/derick/Documents/PINO/poisson/files/last_model_NLC_513~res_0.003103~RelL2TestError_1000~ntrain_5000~ntest_5~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220908-065711.pt'}
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
        MODELS = {16:'/home/derick/Documents/U-FNO/darcy_flow_mu_P/files/last_model_017~res_0.028659~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220723-003010.pt',
                32:'/home/derick/Documents/U-FNO/darcy_flow_mu_P/files/last_model_033~res_0.014182~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220723-002449.pt',
                64:'/home/derick/Documents/U-FNO/darcy_flow_mu_P/files/last_model_065~res_0.007763~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220723-030318.pt',
                128:'/home/derick/Documents/U-FNO/darcy_flow_mu_P/files/last_model_129~res_0.007091~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220723-070406.pt',
                256:'/home/derick/Documents/U-FNO/darcy_flow_mu_P/files/last_model_257~res_0.007431~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220723-151817.pt',
                512:'/home/derick/Documents/U-FNO/darcy_flow_mu_P/files/last_model_513~res_0.009456~RelL2TestError_1000~ntrain_5000~ntest_5~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220725-023609.pt'}
    if pb == 'poisson':
        MODELS = {16:'',
                32:'/home/derick/Documents/U-FNO/poisson/files/last_model_033~res_0.007337~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220804-234423.pt',
                64:'/home/derick/Documents/U-FNO/poisson/files/last_model_065~res_0.006325~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220805-012029.pt',
                128:'/home/derick/Documents/U-FNO/poisson/files/last_model_129~res_0.005584~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220805-043744.pt',
                256:'/home/derick/Documents/U-FNO/poisson/files/last_model_257~res_0.006233~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220805-105243.pt',
                512:'/home/derick/Documents/U-FNO/poisson/files/last_model_513~res_0.005354~RelL2TestError_1000~ntrain_5000~ntest_5~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220806-160812.pt'}
    if pb == 'darcyLN':
        MODELS = {16:'/home/derick/Documents/U-FNO/darcy_flow_mu_L/files/last_model_017~res_0.004046~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220719-093634.pt',
                32:'/home/derick/Documents/U-FNO/darcy_flow_mu_L/files/last_model_033~res_0.00284~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220719-021551.pt',
                64:'/home/derick/Documents/U-FNO/darcy_flow_mu_L/files/last_model_065~res_0.002723~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220719-042400.pt',
                128:'/home/derick/Documents/U-FNO/darcy_flow_mu_L/files/last_model_129~res_0.002962~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220719-062523.pt',
                256:'/home/derick/Documents/U-FNO/darcy_flow_mu_L/files/last_model_257~res_0.003337~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220719-215923.pt',
                512:'/home/derick/Documents/U-FNO/darcy_flow_mu_L/files/last_model_513~res_0.005513~RelL2TestError_1000~ntrain_5000~ntest_5~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220720-165622.pt'}
    if pb == 'navierStokes':
        res = 64
        MODELS = {63:'/home/derick/Documents/U-FNO/navier_stokes/files/last_model_064~res_0.006126~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20230719-234425.pt'}
    if pb == 'helmholtz':
        res = 101
        MODELS = {100:'/home/derick/Documents/U-FNO/helmholtz/files/last_model_101~res_0.038356~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20230720-010021.pt'}
    if pb == 'structuralMechanics':
        res = 41
        MODELS = {40:'/home/derick/Documents/U-FNO/structural_mechanics/files/last_model_041~res_0.057511~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20230719-172731.pt'}
    
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
        MODELS = {16:'/home/derick/Documents/MWT/darcy_flow_mu_P/files/last_model_cs_017~res_0.029181~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220811-030410.pt',
                32:'/home/derick/Documents/MWT/darcy_flow_mu_P/files/last_model_cs_033~res_0.014518~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220811-013959.pt',
                64:'/home/derick/Documents/MWT/darcy_flow_mu_P/files/last_model_cs_065~res_0.008028~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220810-214932.pt',
                128:'/home/derick/Documents/MWT/darcy_flow_mu_P/files/last_model_cs_129~res_0.006054~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220811-214522.pt',
                256:'/home/derick/Documents/MWT/darcy_flow_mu_P/files/last_model_cs_257~res_0.005905~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220812-233259.pt',
                512:''}
    if pb == 'poisson':
        MODELS = {16:'/home/derick/Documents/MWT/poisson/files/last_model_cs_017~res_0.017984~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220811-031640.pt',
                32:'/home/derick/Documents/MWT/poisson/files/last_model_cs_033~res_0.006904~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220811-063636.pt',
                64:'/home/derick/Documents/MWT/poisson/files/last_model_cs_065~res_0.00467~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220811-064918.pt',
                128:'/home/derick/Documents/MWT/poisson/files/last_model_cs_129~res_0.004694~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220812-001750.pt',
                256:'/home/derick/Documents/MWT/poisson/files/last_model_cs_257~res_0.004834~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20220812-154509.pt',
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
        MODELS = {63:'/home/derick/Documents/MWT/navier_stokes/files/last_model_cs_064~res_0.002496~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20230717-183727.pt'}
    if pb == 'helmholtz':
        res = 101
        MODELS = {100:'/home/derick/Documents/MWT/helmholtz/files/last_model_cs_101~res_0.038721~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20230717-223156.pt'}
    if pb == 'structuralMechanics':
        res = 41
        MODELS = {40:'/home/derick/Documents/MWT/structural_mechanics/files/last_model_cs_041~res_0.056396~RelL2TestError_1000~ntrain_5000~ntest_10~BatchSize_0.001~LR_0.0001~Reg_0.5~gamma_100~Step_500~epochs_20230717-193923.pt'}
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
                64:'/home/derick/Documents/PCANN/darcy_flow_mu_P/pcalin/models/last_model_adam_065~res_0.065629~RelL2TestError_100~rd_1000~ntrain_5000~ntest_1000~BatchSize_0.001~LR_0.1~Reg_0.01~gamma_10000~Step_8000~epochs_20221016-234026-373734.pt',
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
                64:'/home/derick/Documents/PCANN/poisson_paper_data/pcalin/models/last_model_065~res_0.001639~RelL2TestError_250~rd_1000~ntrain_5000~ntest_1000~BatchSize_0.001~LR_0.015~Reg_0.01~gamma_2500~Step_8000~epochs_20220819-222531-313487.pt',#'/localdata/Derick/stuart_data/Darcy_421/poisson/-fwd-linear/models/last_model_065~res_0.001639~RelL2TestError_250~rd_1000~ntrain_5000~ntest_1000~BatchSize_0.001~LR_0.015~Reg_0.1~gamma_2500~Step_8000~epochs_20220819-153234-079612.pt',
                128:'',
                256:'',
                512:''}
    if pb == 'darcyLN':
        dX = 200
        dY = 200
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
        dX = 150
        dY = 150
        p_drop = 0
        model = pcann_snn(in_features=dX, out_features=dY, p_drop=p_drop, use_selu=True).cuda()
        MODELS = {16:'',
                32:'',
                64:'/home/derick/Documents/PCANN/darcy_flow_mu_P/pcann/models/last_model_sgd_065~res_0.025560~RelL2TestError_150~rd_0~pdrop_1000~ntrain_5000~ntest_500~BatchSize_0.000001~LR_0.5000~Reg_0.5000~gamma_7500~Step_20000~epochs_20221016-222513-635492.pt',
                128:'',
                256:'',
                512:''}
    if pb == 'poisson':
        dX = 150
        dY = 150
        p_drop = 0
        model = pcann_snn(in_features=dX, out_features=dY, p_drop=p_drop, use_selu=True).cuda()
        MODELS = {16:'',
                32:'',
                64:'/home/derick/Documents/PCANN/poisson_paper_data/pcann/models/last_model_sgd_065~res_0.007835~RelL2TestError_150~rd_0~pdrop_1000~ntrain_5000~ntest_500~BatchSize_0.000000500~LR_0.1000~Reg_0.5000~gamma_2000~Step_20000~epochs_20221016-120749-681164.pt',#'/localdata/Derick/stuart_data/Darcy_421/poisson/--fwd2/RN/epochs 20000 step 2000 gamma 0.5/models/last_model_sgd_065~res_0.007831~RelL2TestError_150~rd_0~pdrop_1000~ntrain_5000~ntest_500~BatchSize_0.000000500~LR_0.1000~Reg_0.5000~gamma_2000~Step_20000~epochs_20221011-183051-950006.pt',
                128:'',
                256:'',
                512:''}
    if pb == 'darcyLN':
        dX = 200
        dY = 200
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
    #fileName = "datasets/new_aUP_Square_TrainData=1_TestData=1_Resolution=513X513_Domain=[0,1]X[0,1].hdf5" 
    fileName = "/localdata/Derick/stuart_data/Darcy_421/new_aUP_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
    normPATH = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/normalisers/darcyPWC/UnitGaussianNormalizer/'
    pcaPATH  = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/pca/darcyPWC/UnitGaussianNormalizer/'
if pb == 'darcyLN': 
    #fileName = "datasets/aUL_Square_TrainData=1_TestData=1_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
    fileName = "/localdata/Derick/stuart_data/Darcy_421/aUL_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
    normPATH = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/normalisers/darcyLN/UnitGaussianNormalizer/'
    pcaPATH  = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/pca/darcyLN/UnitGaussianNormalizer/'
if pb == 'poisson': 
    #fileName = "datasets/fUG_Square_TrainData=1_TestData=1_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
    fileName = "/localdata/Derick/stuart_data/Darcy_421/fUG_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
    normPATH = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/normalisers/poisson/UnitGaussianNormalizer/'
    pcaPATH  = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/pca/poisson/UnitGaussianNormalizer/'
if pb == 'navierStokes': 
    #fileName = "datasets/NavierStokes_TrainData=1_TestData=1_Resolution=64X64_Domain=[0,1]X[0,1].hdf5"
    fileName = "/localdata/Derick/stuart_data/Darcy_421/NavierStokes_TrainData=1000_TestData=5000_Resolution=64X64_Domain=[0,1]X[0,1].hdf5"
    normPATH = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/normalisers/navierStokes/UnitGaussianNormalizer/'
    pcaPATH  = ''
if pb == 'helmholtz': 
    #fileName = "datasets/Helmholtz_TrainData=1_TestData=1_Resolution=101X101_Domain=[0,1]X[0,1].hdf5"
    fileName = "/localdata/Derick/stuart_data/Darcy_421/Helmholtz_TrainData=1000_TestData=5000_Resolution=101X101_Domain=[0,1]X[0,1].hdf5"
    normPATH = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/normalisers/helmholtz/UnitGaussianNormalizer/'
    pcaPATH  = ''
if pb == 'structuralMechanics': 
    #fileName = "datasets/StructuralMechanics_TrainData=1_TestData=1_Resolution=41X41_Domain=[0,1]X[0,1].hdf5"
    fileName= "/localdata/Derick/stuart_data/Darcy_421/StructuralMechanics_TrainData=1000_TestData=5000_Resolution=41X41_Domain=[0,1]X[0,1].hdf5"
    normPATH = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/normalisers/structuralMechanics/UnitGaussianNormalizer/'
    pcaPATH  = ''

_, _, X_TRAIN, Y_TRAIN = readtoArray(fileName, 1, 1, Nx = 512, Ny = 512)


#ntest = 100

print ("Converting dataset to numpy array and subsamping.")
tt = time.time()
X_TRAIN = SubSample(np.array(X_TRAIN[ :ntest, :, :]), res, res)
Y_TRAIN = SubSample(np.array(Y_TRAIN[ :ntest, :, :]), res, res)

print ("    Conversion completed after %.2f minutes"%((time.time()-tt)/60))

print ("Adding noise.")
tt = time.time()
useless = np.zeros((1, res, res))#Y_TRAIN[ :2, :, :]
_, _, _, Y_TRAIN_noisy = add_noise((useless, useless, useless, Y_TRAIN), noise_ratio)

print ("    Adding noise completed after %.2f minutes"%((time.time()-tt)/60))

ntrain = 1000
x_normalizer = torch.load(normPATH+"param_normalizer-%s-res-%s-ntrain.pt"%(res-1, ntrain)) # UnitGaussianNormalizer(x_train)
y_normalizer = torch.load(normPATH+"solut_normalizer-%s-res-%s-ntrain.pt"%(res-1, ntrain)) # UnitGaussianNormalizer(y_train)

X = torch.from_numpy(X_TRAIN).float().cuda()
Y = torch.from_numpy(Y_TRAIN).float().cuda()
Y_noisy = torch.from_numpy(Y_TRAIN_noisy).float().cuda()

ACCURACY = 0
TESTX_l2 = 0
TESTY_l2 = 0
TESTY_l2_noisy = 0
TESTY_l2_learned = 0
TESTY_l2_learned_noisy = 0


TIMESTAMP = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')
directory = 'figures/%s/noiseRatio=%s/%s'%(pb, noise_ratio, no)
if not os.path.exists(directory):
    os.makedirs(directory)
file = open(directory+'/Results_%s_samples_'%(ntest)+TIMESTAMP+'.txt',"a")
file.write("Accuracy Input_Error Output_Error Output_Error_Noiseless Masked_Output_Error Masked_Output_Error_Noiseless\n")


for samp in range(ntest):
    print("Working on sample %s of %s"%(samp+1, ntest))
    res = args.res #+ 1 #Mainly for mwt since res was changed for it
    X_train = X_TRAIN[samp].reshape(1, res, res)
    Y_train = Y_TRAIN[samp].reshape(1, res, res)
    Y_train_noisy = Y_TRAIN_noisy[samp].reshape(1, res, res)
    x = X[samp].reshape(1, res, res)
    y = Y[samp].reshape(1, res, res)
    y_noisy = Y_noisy[samp].reshape(1, res, res)

    if no == 'mwt':
        old_res = res
        res = closest_power(res)
        X_train0 = CubicSpline3D(X_train, res, res)
        Y_train0 = CubicSpline3D(Y_train, res, res)
        Y_train0_noisy = CubicSpline3D(Y_train_noisy, res, res)

        grids_mwt = []
        grids_mwt.append(np.linspace(0, 1, res))
        grids_mwt.append(np.linspace(0, 1, res))
        grid_mwt = np.vstack([xx.ravel() for xx in np.meshgrid(*grids_mwt)]).T
        grid_mwt = grid_mwt.reshape(1,res,res,2)
        grid_mwt = torch.tensor(grid_mwt, dtype=torch.float).cuda()

        x_normalizer = torch.load(normPATH+"param_normalizer-cs%s-res-%s-ntrain.pt"%(res-1, ntrain)) # UnitGaussianNormalizer(x_train)
        y_normalizer = torch.load(normPATH+"solut_normalizer-cs%s-res-%s-ntrain.pt"%(res-1, ntrain)) # UnitGaussianNormalizer(y_train)

        x = torch.from_numpy(X_train0).float().cuda()
        y = torch.from_numpy(Y_train0).float().cuda()
        y_noisy = torch.from_numpy(Y_train0_noisy).float().cuda()

    if no == 'pcalin' or no == 'pcann':
        old_res = res
        res = dX
        X_train = X_train.reshape(1, -1)
        Y_train = Y_train.reshape(1, -1)
        Y_train_noisy = Y_train_noisy.reshape(1, -1)

        pcaX = torch.load(pcaPATH+"param_pca-%s_res-%s_d-%s-ntrain.pt"%(old_res-1, dX, ntrain))
        pcaY = torch.load(pcaPATH+"solut_pca-%s_res-%s_d-%s-ntrain.pt"%(old_res-1, dX, ntrain))

        X_train0 = pcaX.transform(X_train)
        Y_train0 = pcaY.transform(Y_train)
        Y_train0_noisy = pcaY.transform(Y_train_noisy)

        old_x_normalizer = x_normalizer

        if no == 'pcalin':
            x_normalizer = torch.load(pcaPATH+"param_normalizer-%s_res-%s_d-%s-ntrain.pt"%(old_res-1, dX, ntrain)) # UnitGaussianNormalizer(x_train)
            y_normalizer = torch.load(pcaPATH+"solut_normalizer-%s_res-%s_d-%s-ntrain.pt"%(old_res-1, dX, ntrain)) # UnitGaussianNormalizer(y_train)
            
        if no == 'pcann':
            x_normalizer = torch.load(pcaPATH+"param_range-normalizer-%s_res-%s_d-%s-ntrain.pt"%(old_res-1, dX, ntrain))#UnitGaussianNormalizer(XoX_train)
            y_normalizer = torch.load(pcaPATH+"solut_range-normalizer-%s_res-%s_d-%s-ntrain.pt"%(old_res-1, dX, ntrain))#UnitGaussianNormalizer(YoY_train)
        
        #x = torch.from_numpy(X_train0).float().cuda()
        y = torch.from_numpy(Y_train0).float().cuda()
        y_noisy = torch.from_numpy(Y_train0_noisy).float().cuda()


    torch.manual_seed(0)
    np.random.seed(0)

    num_samp = x.size()[0]

    #initialisation = 'random' #'random'#
    out_shape = [num_samp,res,res]
    if pb == 'navierStokes':
        init = 'random'#'random'#'normal'
    if pb == 'structuralMecahnics':
        init = 'random'#'random'#'normal'
    if pb == 'helmholtz':
        init = 'random'#'random'#'normal'
    if pb == 'poisson':
        init = 'ones'#'random'#'normal'
    if pb == 'darcyPWC':
        init = 'ones'#'choice'
    if pb == 'darcyLN':
        init = 'ones'#'random'
    if no == 'pcalin' or no == 'pcann':
        init = 'ones'#'random'
        out_shape = [num_samp,res]

    if init == 'random':
        x_learned = torch.rand(out_shape, device="cuda")
        out_masked = x_learned.requires_grad_()
    if init == 'choice':
        x_learned = torch.from_numpy(np.random.choice([3, 12], size=out_shape)).float().cuda()#.requires_grad_()   
        out_masked = x_learned.requires_grad_()
    if init == 'ones':
        x_learned = torch.ones(out_shape, device="cuda")
        out_masked = x_learned.requires_grad_()
    if init == 'normal':
        x_learned = torch.randn(out_shape, device="cuda")
        out_masked = x_learned.requires_grad_()



    # list_pb     = ['poisson', 'darcyPWC', 'navierStokes', 'structuralMechanics', 'helmholtz']
    list_models =                          [ 'fno', 'pino', 'ufno', 'mwt' , 'pcann', 'pcalin']


    dict_alpha  = {'poisson'             : [2.5e-1, 5.0e-2, 5.0e-2, 5.0e-2,  5.0e-2, 2.5e-2],
                'darcyPWC'            : [2.5e-2, 5.0e-3, 5.0e-3, 5.0e-3,  5.0e-3, 1.0e-3],
                'navierStokes'        : [5.0e-5, m.nan , 5     , 5     ,  m.nan , m.nan ],
                'structuralMechanics' : [2.5e-1, m.nan , 2.5e-2, 1.0e-1,  m.nan , m.nan ],
                'helmholtz'           : [5.0e-1, m.nan , 1     , 1     ,  m.nan , m.nan ]}


    dict_wd     = {'poisson'             : [0     , 0     , 0     , 0     ,  5.0e-4, 1.0e-5],
                'darcyPWC'            : [0     , 0     , 0     , 0     ,  1.0e-4, 1.0e-3],
                'navierStokes'        : [1.0e-4, m.nan , 0     , 0     ,  m.nan , m.nan ],
                'structuralMechanics' : [1.0e-4, m.nan , 1.0e-5, 5.0e-5,  m.nan , m.nan ],
                'helmholtz'           : [0     , m.nan , 0     , 0     ,  m.nan , m.nan ]}

    df_alpha  = pd.DataFrame(dict_alpha,
                    index = list_models)

    df_wd     = pd.DataFrame(dict_wd,
                    index = list_models)

    alpha = df_alpha.at[no, pb]
    wd    = df_wd.at[no, pb]

    optimizer = torch.optim.Adam([out_masked], lr=learning_rate, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    x_normalizer.cuda()
    y_normalizer.cuda()



    # directory = 'files/'+pb+'/noiseRatio=%s'%(noise_ratio)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)


    # if os.path.isfile(directory+'/TrainData_'+TIMESTAMP+'.txt'):
    #     os.remove(directory+'/TrainData_'+TIMESTAMP+'.txt')

    PINO_loss = darcy_PINO_loss if pb[0:5] == 'darcy' else poisson_PINO_loss
    #testx_l2 = 0
    for ep  in range(epochs):
        t1 = default_timer()

        optimizer.zero_grad()

        #out_masked = mask(x_learned)
        #out_masked = x_normalizer.encode(out_masked)
        if no == 'mwt':
            yout = model(torch.cat([out_masked.reshape(num_samp, res, res, 1), grid_mwt.repeat(num_samp,1,1,1)], dim=3)).reshape(num_samp, res, res)
        elif no == 'pcalin' or no == 'pcann':
            yout = model(out_masked)
        else: 
            yout = model(out_masked.reshape(num_samp, res, res, 1)).reshape(num_samp, res, res)
        
        yout = y_normalizer.decode(yout)

        #yout = yout * mollifier
        loss_data = myloss(yout.view(num_samp, -1), y.view(num_samp, -1))
        loss_data_noisy = myloss(yout.view(num_samp, -1), y_noisy.view(num_samp, -1)) 

        if pb != 'darcyPWC' and (no == 'pino' or no == 'fno' or no == 'ufno' or no == 'mwt'): #PDE loss only for PINO, FNO, UFNO, MWT in all problems but darcyPWC.
            loss_f, loss_bd = 0, 0 #PINO_loss(yout, x_normalizer.decode(out_masked)) #Interesting to note that using yout inplace of y_noisy here produces worst results for no noise (It however very slightly improves FNO results).
        else:
            loss_f, loss_bd = 0, 0
            
        #TV_loss = total_variation_loss if (pb == 'poisson' and not(no == 'pcann' or no == 'pcalin')) else total_variance
        TV_loss = total_variance if pb == 'darcyPWC' else tvl2
        loss_TV = TV_loss(x_normalizer.decode(out_masked))
        
        pino_loss       = 0.2*loss_f + loss_data       + alpha*loss_TV
        pino_loss_noisy = 0.2*loss_f + loss_data_noisy + alpha*loss_TV
            
        pino_loss_noisy.backward()
        optimizer.step()
        scheduler.step()
        
        apply_mask2 = False #True if pb == 'darcyPWC' else False

        if apply_mask2:
            out_learned = out_masked
        else:
            out_learned = x_normalizer.decode(out_masked)
        
        if no == 'mwt':
            out_learned = CubicSpline3D(out_learned.detach().cpu().numpy().reshape(num_samp, res, res), old_res, old_res)
            yout = CubicSpline3D(yout.detach().cpu().numpy().reshape(num_samp, res, res), old_res, old_res)
            out_learned = torch.from_numpy(out_learned).float().to(device)
            yout = torch.from_numpy(yout).float().to(device)
            x_mwt = torch.from_numpy(X_train).float().cuda()
            y_mwt = torch.from_numpy(Y_train).float().cuda()
            y_mwt_noisy = torch.from_numpy(Y_train_noisy).float().cuda()

        if no == 'pcalin' or no == 'pcann': #CHANGE
            if apply_mask2:
                out_learned = x_normalizer.decode(out_learned.reshape(num_samp, -1))
            out_learned = pcaX.inverse_transform(out_learned.detach().cpu().numpy().reshape(num_samp, -1)).reshape(num_samp, old_res, old_res)
            YYout        = pcaY.inverse_transform(yout       .detach().cpu().numpy().reshape(num_samp, -1)).reshape(num_samp, old_res, old_res)
            YY           = pcaY.inverse_transform(y          .detach().cpu().numpy().reshape(num_samp, -1)).reshape(num_samp, old_res, old_res)
            YY_noisy     = pcaY.inverse_transform(y_noisy    .detach().cpu().numpy().reshape(num_samp, -1)).reshape(num_samp, old_res, old_res)
            out_learned = torch.from_numpy(out_learned).float().to(device)
            YY = torch.from_numpy(YY).float().to(device)
            YYout = torch.from_numpy(YYout).float().to(device)
            if apply_mask2:
                old_x_normalizer.cuda()
                out_learned = old_x_normalizer.encode(out_learned)
            YY_noisy = torch.from_numpy(YY_noisy).float().to(device)

            X_train = X_train.reshape(num_samp, old_res, old_res)
            Y_train = Y_train.reshape(num_samp, old_res, old_res)
            Y_train_noisy = Y_train_noisy.reshape(num_samp, old_res, old_res)

        if pb == 'darcyPWC':
            mean_out = torch.mean(out_learned)
            out_learned[out_learned>mean_out] = 12
            out_learned[out_learned<=mean_out] = 3

            if no == 'mwt':
                out_learned_mwt = CubicSpline3D(out_learned.detach().cpu().numpy(), res, res)
                out_learned_mwt = torch.from_numpy(out_learned_mwt).float().to(device)
                out_learned_mwt = x_normalizer.encode(out_learned_mwt)
                y_learned = y_normalizer.decode(model(torch.cat([out_learned_mwt.reshape(num_samp, res, res, 1), grid_mwt.repeat(num_samp,1,1,1)], dim=3)).reshape(num_samp, res, res))
                y_learned = CubicSpline3D(y_learned.detach().cpu().numpy().reshape(num_samp, res, res), old_res, old_res)
                y_learned = torch.from_numpy(y_learned).float().to(device)
            elif no == 'pcalin' or no == 'pcann':
                y_learned = pcaX.transform(out_learned.detach().cpu().numpy().reshape(num_samp, -1))
                y_learned = x_normalizer.encode(torch.from_numpy(y_learned).float().to(device))
                y_learned = model(y_learned)
                y_learned = y_normalizer.decode(y_learned)
                y_learned = pcaY.inverse_transform(y_learned.detach().cpu().numpy().reshape(num_samp, -1)).reshape(num_samp, old_res, old_res)
                y_learned = torch.from_numpy(y_learned).float().to(device)
            else:
                y_learned = y_normalizer.decode(model(x_normalizer.encode(out_learned).reshape(num_samp, res, res, 1)).reshape(num_samp, res, res))
            

        if apply_mask2:
            #out_learned = darcy_mask2(out_masked)
            out_learned = 1 / (1 + torch.exp(-out_learned)) 
            out_learned[out_learned>0.5] = 12
            out_learned[out_learned<=0.5] = 3
            #yout = model(out_learned.reshape(num_samp, res, res, 1)).reshape(num_samp, res, res)
            #yout = y_normalizer.decode(yout)
        #yout2 = yout2 * mollifier
        if no == 'mwt':
            testx_l2 = myloss(out_learned.view(num_samp, -1), x_mwt.view(num_samp, -1)).item()
            testy_l2 = myloss(yout.view(num_samp, -1), y_mwt.view(num_samp, -1)).item() #loss_data.item()#
            testy_l2_noisy = myloss(yout.view(num_samp, -1), y_mwt_noisy.view(num_samp, -1)).item() #loss_data.item()#
            if pb == 'darcyPWC':
                testy_l2_learned = myloss(y_learned.view(num_samp, -1), y_mwt.view(num_samp, -1)).item() #loss_data.item()#
                testy_l2_learned_noisy = myloss(y_learned.view(num_samp, -1), y_mwt_noisy.view(num_samp, -1)).item() #loss_data.item()#
            else:
                testy_l2_learned, testy_l2_learned_noisy = 0, 0
                
            accuracy = 100*(out_learned.flatten() == x_mwt.flatten()).float().sum()/len(out_learned.flatten())
        elif no == 'pcalin' or no == 'pcann':   
            testx_l2 = myloss(out_learned.view(num_samp, -1), x.view(num_samp, -1)).item()
            testy_l2 = myloss(YYout.view(num_samp, -1), YY.view(num_samp, -1)).item() #loss_data.item()#
            testy_l2_noisy = myloss(YYout.view(num_samp, -1), YY_noisy.view(num_samp, -1)).item() #loss_data.item()#
            if pb == 'darcyPWC':
                testy_l2_learned = myloss(y_learned.view(num_samp, -1), YY.view(num_samp, -1)).item() #loss_data.item()#
                testy_l2_learned_noisy = myloss(y_learned.view(num_samp, -1), YY_noisy.view(num_samp, -1)).item() #loss_data.item()#
            else:
                testy_l2_learned, testy_l2_learned_noisy = 0, 0

            accuracy = 100*(out_learned.flatten() == x.flatten()).float().sum()/len(out_learned.flatten())
        else:    
            testx_l2 = myloss(out_learned.view(num_samp, -1), x.view(num_samp, -1)).item()
            testy_l2 = myloss(yout.view(num_samp, -1), y.view(num_samp, -1)).item() #loss_data.item()#
            testy_l2_noisy = myloss(yout.view(num_samp, -1), y_noisy.view(num_samp, -1)).item() #loss_data.item()#
            if pb == 'darcyPWC':
                testy_l2_learned = myloss(y_learned.view(num_samp, -1), y.view(num_samp, -1)).item() #loss_data.item()#
                testy_l2_learned_noisy = myloss(y_learned.view(num_samp, -1), y_noisy.view(num_samp, -1)).item() #loss_data.item()#
            else:
                testy_l2_learned, testy_l2_learned_noisy = 0, 0

            accuracy = 100*(out_learned.flatten() == x.flatten()).float().sum()/len(out_learned.flatten())

        accuracy = accuracy.item()

        if  pb == 'darcyPWC':
            if no == 'pcalin' or no == 'pcann':   
                Yout = y_learned
            else:
                yout = y_learned

        
        if ep % write_step == write_step - 1 or ep == 0:
            t2 = default_timer()
            print("epoch: %s, completed in %.2fs. | In. Loss: %.4f, Out. Loss : %.4f, Noiseless Out. Loss %.4f | In. Acc. %.2f %%"%(ep+1, t2-t1, testx_l2, testy_l2_noisy, testy_l2, accuracy))
    #         file = open(directory+'/TrainData_'+TIMESTAMP+'.txt',"a")
    #         file.write(str(ep+1)+" "+str(testx_l2)+" "+str(testy_l2_noisy)+" "+str(testy_l2)+" "+str(accuracy)+"\n")
    #     if pb == 'darcyPWC':
    #         ModelInfos = "_%s-%s_%s-mask2=%s_%03d"%(no, pb, init, apply_mask2, res)+"~res_"+"_%05d"%(ep+1)+"~ep_"+"%.4f~InAcc_"%(accuracy)+str(np.round(testx_l2,6))+"~InputError_"+str(np.round(testy_l2_noisy,6))+"~OutputError_"+\
    #             str(np.round(testy_l2,6))+"~OutputErrorNoiseless_"+str(learning_rate)+"~LR_"+str(wd)+"~Reg_"+str(gamma)+"~gamma_"+str(step_size)+"~Step_"+str(epochs)+"~epochs_"+time.strftime("%Y%m%d-%H%M%S")  
    #     else:
    #         ModelInfos = "_%s-%s_%s-mask2=%s_%03d"%(no, pb, init, apply_mask2, res)+"~res_"+"_%05d"%(ep+1)+"~ep_"+str(np.round(testx_l2,6))+"~InputError_"+str(np.round(testy_l2_noisy,6))+"~OutputError_"+\
    #             str(np.round(testy_l2,6))+"~OutputErrorNoiseless_"+str(learning_rate)+"~LR_"+str(wd)+"~Reg_"+str(gamma)+"~gamma_"+str(step_size)+"~Step_"+str(epochs)+"~epochs_"+time.strftime("%Y%m%d-%H%M%S")  
        

    #     if ep % plot_step == plot_step - 1 or ep == 0:
    #         if no == 'mwt':
    #             plot_comparism(pb, no, noise_ratio, out_learned, yout, X_train, Y_train_noisy, Y_train, ModelInfos, num_samp, old_res, myloss, accuracy)
    #         elif no == 'pcalin' or no == 'pcann':
    #             plot_comparism(pb, no, noise_ratio, out_learned, Yout, X_train, Y_train_noisy, Y_train, ModelInfos, num_samp, old_res, myloss, accuracy)
    #         else:
    #             plot_comparism(pb, no, noise_ratio, out_learned, yout, X_train, Y_train_noisy, Y_train, ModelInfos, num_samp, res, myloss, accuracy)

                
    # #torch.save(x_learned, "files/x_learned"+ModelInfos+".pt")
    # os.rename(directory+'/TrainData_'+TIMESTAMP+'.txt', directory+'/TrainData'+ModelInfos+'.txt')


    # dataTrain = np.loadtxt(directory+'/TrainData'+ModelInfos+'.txt')
    # stepTrain = dataTrain[:,0] #Reading Epoch                   
    # errorIn = dataTrain[:,1] #Reading erros
    # errorOut  = dataTrain[:,2] #Reading erros
    # errorOutNoiseless  = dataTrain[:,3] #Reading erros
    # accuracyIn  = dataTrain[:,4] #Reading erros

    # print("Ploting Loss VS training step...")
    # fig = plt.figure(figsize=(15, 10))
    # #plt.yscale('log')
    # plt.plot(stepTrain, errorIn, label = 'Input Error')
    # plt.plot(stepTrain, errorOut , label = 'Output Error')
    # plt.plot(stepTrain, errorOutNoiseless , label = 'Output Error (Noiseless)')
    # #plt.plot(stepTrain, accuracyIn/100 , label = r'$\times 10^{2}$ Input Accuracy')
    # plt.xlabel('epochs')#, fontsize=16, labelpad=15)
    # plt.ylabel('Error')
    # plt.legend(loc = 'center right')
    # title = "In test error = %.4f, Output test error = %.4f,"%(testx_l2, testy_l2)

    # if pb == 'darcyPWC':
    #     title = "In test error = %.4f, Output test error = %.4f, Input Accuracy = %.2f "%(testx_l2, testy_l2, accuracy)
    #     plt.gca().twinx().plot(stepTrain, accuracyIn , label = 'Accuracy', color = 'red')
    #     plt.xlabel('epochs')#, fontsize=16, labelpad=15)
    #     plt.ylabel('Accuracy')
    #     plt.legend(loc = 'upper right')


    # plt.title(title)

    # directory = 'figures/'+pb
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # plt.savefig(directory+'/noiseRatio=%s'%(noise_ratio)+'/Error_VS_TrainingStep'+ModelInfos+".png", dpi=500)

    #plt.show()

    ACCURACY += accuracy
    TESTX_l2 += testx_l2
    TESTY_l2 += testy_l2
    TESTY_l2_noisy += testy_l2_noisy
    TESTY_l2_learned += testy_l2_learned
    TESTY_l2_learned_noisy += testy_l2_learned_noisy

    file = open(directory+'/Results_%s_samples_'%(ntest)+TIMESTAMP+'.txt',"a")
    file.write("%s %s %s %s %s %s\n"%(accuracy, testx_l2, testy_l2_noisy, testy_l2, testy_l2_learned_noisy, testy_l2_learned))


file.write("\n")
file = open(directory+'/Results_%s_samples_'%(ntest)+TIMESTAMP+'.txt',"a")
file.write("%s %s %s %s %s %s\n"%(ACCURACY/ntest, TESTX_l2/ntest, TESTY_l2_noisy/ntest, TESTY_l2/ntest, TESTY_l2_learned_noisy/ntest, TESTY_l2_learned/ntest))
