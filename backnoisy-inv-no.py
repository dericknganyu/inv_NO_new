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
parser.add_argument('--res', default=64, type=int, help='Specify redolution')
parser.add_argument('--ep', default=500, type = int, help='epochs')#
parser.add_argument('--lr', default=0.001, type = float, help='learning rate')#
parser.add_argument('--wd', default=1e-4, type = float, help='weight decay')
parser.add_argument('--no', default='pino', type=str, help='Specify Neural operator: pino, fno, ufno, pcalin, pcann')
parser.add_argument('--pb', default='darcyPWC', type=str, help='Specify Problem: poison, darcyPWC, darcyLN')
parser.add_argument('--gm', default=0.5, type = float, help='gamma')#
parser.add_argument('--ss', default=100, type = int, help='step size')#
parser.add_argument('--ps', default=2000, type = int, help='plot step')#
parser.add_argument('--init', default='random', type=str, help='Specify initialisation: random, choice, ...')
parser.add_argument('--nr', default=0.0, type = float, help='noise ratio')#
parser.add_argument('--ntest', default=5000, type = int, help='number of testing samples')#
parser.add_argument('--ntrain', default=1000, type = int, help='number of training samples')#
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
#plot_step = epochs/5 #args.ps
#write_step = epochs/200
#init = args.init
noise_ratio = args.nr
ntest = args.ntest
ntrain = args.ntrain
batch_size = args.bs

if no == 'fno':
    modes = 12
    width = 32
    model = FNO2d(modes, modes, width).cuda()

if no == 'pino':
    modes = 12
    width = 32
    model = FNO2d(modes, modes, width).cuda()

if no == 'ufno':
    modes = 12
    width = 32
    model = UFNO2d(modes, modes, width).cuda()

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

if no == "pcalin":
    if pb == 'darcyPWC':
        dX = 100
        dY = 100
        #params = {}
        params["layers"] = [dX , dY]
    if pb == 'poisson':
        dX = 250
        dY = 250
        #params = {}
        params["layers"] = [dX , dY]
    model = pcalin(params).cuda()
if no == "pcann":
    if pb == 'darcyPWC':
        dX = 30
        dY = 30
        p_drop = 0.01
    if pb == 'poisson':
        dX = 200
        dY = 200
        p_drop = 0.001
    model = pcann_snn(in_features=dX, out_features=dY, p_drop=p_drop, use_selu=True).cuda()

myloss = LpLoss(size_average=False)

#---------------------------------------------------------------------------------------------
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)


if pb == 'darcyPWC':
    fileName_ex = "datasets/new_aUP_Square_TrainData=1_TestData=1_Resolution=513X513_Domain=[0,1]X[0,1].hdf5" 
    fileName = "/localdata/Derick/stuart_data/Darcy_421/new_aUP_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
    normPATH = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/normalisers/darcyPWC/UnitGaussianNormalizer/'
    pcaPATH  = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/pca/darcyPWC/UnitGaussianNormalizer/'
if pb == 'darcyLN': 
    fileName_ex = "datasets/aUL_Square_TrainData=1_TestData=1_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
    fileName = "/localdata/Derick/stuart_data/Darcy_421/aUL_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
    normPATH = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/normalisers/darcyLN/UnitGaussianNormalizer/'
    pcaPATH  = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/pca/darcyLN/UnitGaussianNormalizer/'
if pb == 'poisson': 
    fileName_ex = "datasets/fUG_Square_TrainData=1_TestData=1_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
    fileName = "/localdata/Derick/stuart_data/Darcy_421/fUG_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
    normPATH = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/normalisers/poisson/UnitGaussianNormalizer/'
    pcaPATH  = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/pca/poisson/UnitGaussianNormalizer/'



Y_train, X_train, Y_test, X_test = readtoArray(fileName, 1, 1, Nx = 512, Ny = 512)
useless = np.zeros((1, res, res))#Y_TRAIN[ :2, :, :]
print("Adding noise...")
_, X_train, _, X_test = add_noise((useless, X_train, useless, X_test), noise_ratio)

print ("Converting dataset to numpy array and subsamping.")
tt = time.time()
X_train = SubSample(np.array(X_train[ :ntrain, :, :]), res, res)
Y_train = SubSample(np.array(Y_train[ :ntrain, :, :]), res, res)
X_test  = SubSample(np.array(X_test [ :ntest , :, :]), res, res)
Y_test  = SubSample(np.array(Y_test [ :ntest , :, :]), res, res)

print ("    Conversion completed after %.2f minutes"%((time.time()-tt)/60))

if no == 'fno' or no == 'ufno' or no == 'pino':
    x_train = torch.from_numpy(X_train).float()#.cuda()
    y_train = torch.from_numpy(Y_train).float()#.cuda()
    x_test  = torch.from_numpy(X_test ).float()#.cuda()
    y_test  = torch.from_numpy(Y_test ).float()#.cuda()

    x_normalizer = UnitGaussianNormalizer(x_train)
    y_normalizer = UnitGaussianNormalizer(y_train)

if no == 'mwt':
    old_res = res
    res = closest_power(res)
    X_train0 = CubicSpline3D(X_train, res, res)
    Y_train0 = CubicSpline3D(Y_train, res, res)
    X_test0  = CubicSpline3D(X_test , res, res)
    Y_test0  = CubicSpline3D(Y_test , res, res)

    grids_mwt = []
    grids_mwt.append(np.linspace(0, 1, res))
    grids_mwt.append(np.linspace(0, 1, res))
    grid_mwt = np.vstack([xx.ravel() for xx in np.meshgrid(*grids_mwt)]).T
    grid_mwt = grid_mwt.reshape(1,res,res,2)
    grid_mwt = torch.tensor(grid_mwt, dtype=torch.float)#.cuda()

    x_train = torch.from_numpy(X_train0).float()#.cuda()
    y_train = torch.from_numpy(Y_train0).float()#.cuda()
    x_test  = torch.from_numpy(X_test0 ).float()#.cuda()
    y_test  = torch.from_numpy(Y_test0 ).float()#.cuda()

    x_normalizer = UnitGaussianNormalizer(x_train)
    y_normalizer = UnitGaussianNormalizer(y_train)

if no == 'pcalin' or no == 'pcann':

    old_res = res
    res = dX
    X_train = X_train.reshape(ntrain, -1)
    Y_train = Y_train.reshape(ntrain, -1)
    X_test  = X_test.reshape (ntest, -1)
    Y_test  = Y_test.reshape (ntest, -1)   

    pcaX = PCA(n_components = dX, random_state = 0).fit(X_train)
    pcaY = PCA(n_components = dY, random_state = 0).fit(Y_train)  

    X_train0 = pcaX.transform(X_train)
    Y_train0 = pcaY.transform(Y_train)
    X_test0  = pcaX.transform(X_test)
    Y_test0  = pcaY.transform(Y_test)

    x_train = torch.from_numpy(X_train0).float().cuda()
    y_train = torch.from_numpy(Y_train0).float().cuda()
    x_test  = torch.from_numpy(X_test0 ).float().cuda()
    y_test  = torch.from_numpy(Y_test0 ).float().cuda()

    x_normalizer = UnitGaussianNormalizer(x_train)
    y_normalizer = UnitGaussianNormalizer(y_train)



x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)
y_train = y_normalizer.encode(y_train)


#x_normalizer.cuda()
y_normalizer.cuda()

if no == 'fno' or no == 'ufno' or no == 'pino':
    x_train = x_train.reshape(ntrain,res,res,1)
    x_test = x_test.reshape(ntest,res,res,1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

if no == 'mwt':
    x_train = torch.cat([x_train.reshape(ntrain, res, res, 1), grid_mwt.repeat(ntrain,1,1,1)], dim=3)
    x_test  = torch.cat([x_test.reshape (ntest , res, res, 1), grid_mwt.repeat(ntest,1,1,1)], dim=3)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

if no == 'pcalin' or no == 'pcann':
    if no == 'pcalin': 
        if pb == 'poisson':
            batch_size = 1000 
            wd = 0.1
            learning_rate = 0.001
            step_size = 2500
            gamma = 0.01
            epochs = 8000
        if pb == 'darcyPWC':
            batch_size = 200 
            wd = 0.15
            learning_rate = 0.001
            step_size = 2500
            gamma = 0.01
            epochs = 8000
    if no == 'pcann':
        if pb == 'poisson':
            batch_size = 500
            wd = 0.1
            learning_rate = 0.00001
            step_size = 2500
            gamma = 0.1
            epochs = 10000
        if pb == 'darcyPWC':
            batch_size = 100
            wd = 0.1
            learning_rate = 0.0001
            step_size = 2500
            gamma = 0.01
            epochs = 10000
            
    torch.manual_seed(0)
    np.random.seed(0)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=False)#shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

################################################################
# training and evaluation
################################################################
if no == 'pcalin' or no == 'pcann':
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99, dampening=0, weight_decay=wd, nesterov=True)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)
y_normalizer.cuda()

directory = 'files-backnoisy' + '/' + pb + '/' + no
if not os.path.exists(directory):
    os.makedirs(directory)
directory_figs = 'figures-backnoisy' + '/' + pb + '/' + no
if not os.path.exists(directory_figs):
    os.makedirs(directory_figs)

TIMESTAMP = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')
if os.path.isfile(directory + '/lossTrainData_'+TIMESTAMP+'.txt'):
    os.remove(directory + '/lossTrainData_'+TIMESTAMP+'.txt')

PINO_loss = darcy_PINO_loss if pb[0:5] == 'darcy' else poisson_PINO_loss
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    train_f = 0

    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
            
        optimizer.zero_grad()
        out = model(x)
        if no != 'pcalin' and no != 'pcann':
            out = out.reshape(batch_size, res, res)        
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)

        if no == 'pino':
            loss_f, loss_bd = PINO_loss(x.reshape(batch_size, res, res), out) #Interesting to note that using yout inplace of y_noisy here produces worst results for no noise (It however very slightly improves FNO results).
        else:
            loss_f, loss_bd = 0, 0 

        loss_data = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
        
        delta = 0.001 if pb == 'darcyPWC' else 0.2

        loss =  loss_data + delta*loss_f
        loss.backward()

        optimizer.step()

        if no == 'mwt':
            print("Returning to orignial resolution to get test error based on it")
            out = CubicSpline3D(out.detach().cpu().numpy(), old_res, old_res)
            out = torch.from_numpy(out).float().to(device)
            y   = CubicSpline3D(y.detach().cpu().numpy(), old_res, old_res)
            y   = torch.from_numpy(y).float().to(device)

        if no == 'pcalin' or no == 'pcann':
            print("Returning to orignial resolution to get test error based on it")
            out = pcaY.inverse_transform(out.detach().cpu().numpy())
            out = torch.from_numpy(out).float().to(device)

            y = pcaY.inverse_transform(y.detach().cpu().numpy())
            y = torch.from_numpy(y).float().to(device)

        train_l2 += loss_data.item()

        if no == 'pino':
            train_f += loss_f.item()
    if no == 'pcann' and pb == 'poisson': 
        scheduler.step(loss) #Strangely, this produces better results than the case with no loss as argument
    else:
        scheduler.step()


    model.eval()
    test_l2 = 0.0
    abs_err = 0.0
    with torch.no_grad():

        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x)
            if no != 'pcalin' and no != 'pcann':
                out = out.reshape(batch_size, res, res)   
            out = y_normalizer.decode(out)

            if no == 'mwt':
                print("Returning to orignial resolution to get test error based on it")
                out = CubicSpline3D(out.detach().cpu().numpy(), old_res, old_res)
                out = torch.from_numpy(out).float().to(device)
                y   = CubicSpline3D(y.detach().cpu().numpy(), old_res, old_res)
                y   = torch.from_numpy(y).float().to(device)

            if no == 'pcalin' or no == 'pcann':
                print("Returning to orignial resolution to get test error based on it")
                out = pcaY.inverse_transform(out.detach().cpu().numpy())
                out = torch.from_numpy(out).float().to(device)

                y = pcaY.inverse_transform(y.detach().cpu().numpy())
                y = torch.from_numpy(y).float().to(device)


            test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

    train_l2/= ntrain
    if no == 'pino':
        train_f /= ntrain
    abs_err /= ntest
    test_l2 /= ntest

    t2 = default_timer()
    print("epoch: %s, completed in %.4f seconds. Training Loss: %.4f and Test Loss: %.4f"%(ep+1, t2-t1, train_l2, test_l2))

    file = open(directory + '/lossTrainData_'+TIMESTAMP+'.txt',"a")
    file.write(str(ep)+" "+str(train_l2)+" "+str(test_l2)+" "+str(train_f)+"\n")

ModelInfos = '_res=%s_%.6f-relErr-%s-%s-%s-noise'%(res, test_l2, pb, no, noise_ratio)
                            
torch.save(model.state_dict(), directory + "/last_model"+ModelInfos+".pt")

os.rename(directory + '/lossTrainData_'+TIMESTAMP+'.txt', directory + '/lossTrainData'+ModelInfos+'.txt')

dataTrain = np.loadtxt(directory + '/lossTrainData'+ModelInfos+'.txt')
stepTrain = dataTrain[:,0] #Reading Epoch                   
errorTrain = dataTrain[:,1] #Reading erros
errorTest  = dataTrain[:,2] #Reading erros

print("Ploting Loss VS training step...")
fig = plt.figure(figsize=(15, 10))
plt.yscale('log')
plt.plot(stepTrain, errorTrain, label = 'Training Loss')
plt.plot(stepTrain, errorTest , label = 'Test Loss')
plt.xlabel('epochs')#, fontsize=16, labelpad=15)
plt.ylabel('Loss')
plt.legend(loc = 'upper right')
plt.title("lr = %s test error = %s"%(learning_rate, str(np.round(test_l2,6))))
plt.savefig(directory_figs + "/Error_VS_TrainingStep"+ModelInfos+".png", dpi=500)

#def use_model():#(params, model,device,nSample,params):

model.load_state_dict(torch.load(directory + "/last_model"+ModelInfos+".pt"))
model.eval()

print('5')

print()
print()

#Just a file containing data sampled in same way as the training and test dataset
Y_train, X_train, Y_test, X_test = readtoArray(fileName_ex, 1, 1, Nx = 512, Ny = 512)
_      , X_train, _     , X_test = add_noise((useless, X_train, useless, X_test), noise_ratio)

if no == 'fno' or no == 'ufno' or no == 'pino':
    X_train = SubSample(np.array(X_train), res, res)

    print("Starting the Verification with Sampled Example")
    tt = time.time()
    Y_FDM = SubSample(np.array(Y_train), res, res)[0]

    print("      Doing FNO on Example...")
    tt = time.time()
    ff = torch.from_numpy(X_train).float()
    ff = x_normalizer.encode(ff)
    ff = ff.reshape(1,res,res,1).cuda()#torch.cat([ff.reshape(1,res,res,1), grid.repeat(1,1,1,1)], dim=3).cuda()

    Y_NO = y_normalizer.decode(model(ff).reshape(1, res, res)).detach().cpu().numpy()
    Y_NO = Y_NO[0] 
    print("            FNO completed after %s"%(time.time()-tt))

if no == 'mwt':
    X_train = SubSample(np.array(X_train), old_res, old_res)
    X_train_cs = CubicSpline3D(X_train, res, res)

    print("Starting the Verification with Sampled Example")
    tt = time.time()
    Y_FDM = SubSample(np.array(Y_train), old_res, old_res)[0]
    #Y_FDM = CubicSpline3D(Y_FDM, new_res, new_res)

    print("      Doing MWT on Example...")
    tt = time.time()
    ff = torch.from_numpy(X_train_cs).float()
    ff = x_normalizer.encode(ff)
    ff = torch.cat([ff.reshape(1,res,res,1), grid_mwt.repeat(1,1,1,1)], dim=3).cuda() #ff.reshape(1,res,res,1).cuda()#

    Y_NO = model(ff)
    Y_NO = Y_NO.reshape(1, res, res)
    Y_NO = y_normalizer.decode(Y_NO)
    Y_NO = Y_NO.detach().cpu().numpy()
    Y_NO = CubicSpline3D(Y_NO, old_res, old_res)
    Y_NO = Y_NO[0] 
    print("            MWT completed after %s"%(time.time()-tt)) 

if no == 'pcalin' or no == 'pcann':
	X_train = SubSample(X_train, old_res, old_res)
	Y_train = SubSample(Y_train, old_res, old_res)
	ff = np.array(X_train[0])

	print("Starting the Verification with Sampled Example")
	Y_FDM = np.array(Y_train[0])

	print("      Doing PCANN on Example...")
	tt = time.time()

	inPCANN = pcaX.transform(ff.reshape(1, -1))
	inPCANN = torch.from_numpy(inPCANN).float().to(device)
	inPCANN = model(x_normalizer.encode(inPCANN))
	inPCANN = y_normalizer.decode(inPCANN) #changed!!
	inPCANN = inPCANN.detach().cpu().numpy()

	Y_NO = pcaY.inverse_transform(inPCANN).reshape(old_res, old_res)
	print("            PCANN completed after %s"%(time.time()-tt))

myLoss = LpLoss(size_average=False)
print()
print("Ploting comparism of FDM and FNO Simulation results")
fig = plt.figure(figsize=((5+2)*4, 5))

fig.suptitle(r"Plot of $-\nabla \cdot (a(s) \nabla u(s)) = f(s), \partial \Omega = 0$ with $u|_{\partial \Omega}  = 0.$")

colourMap = parula() #plt.cm.jet #plt.cm.coolwarm

plt.subplot(1, 4, 1)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("Input")
plt.imshow(X_train[0], cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

plt.subplot(1, 4, 2)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("Truth")
plt.imshow(Y_FDM, cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

plt.subplot(1, 4, 3)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title('Noisy inv'+no.upper())
plt.imshow(Y_NO, cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

plt.subplot(1, 4, 4)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title("Truth-"+'inv'+no.upper()+", RelL2Err = "+str(round(myLoss.rel_single(Y_NO, Y_FDM).item(), 3)))
plt.imshow(np.abs(Y_FDM - Y_NO), cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()#format=OOMFormatter(-5))

plt.savefig(directory_figs + '/compare'+ModelInfos+'.png',dpi=500)

#plt.show()
