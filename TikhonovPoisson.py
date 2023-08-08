import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from torch import autograd
from utilities.colorMap import parula
import os
import time

from utilities.readData import readtoArray
from utilities.add_noise import *

colourMap = parula()  # plt.cm.jet
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
colourMap = parula()  # plt.cm.jet
torch.manual_seed(1234)
np.random.seed(1234)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.set_default_tensor_type(torch.FloatTensor)


fileName = "datasets/fUG_Square_TrainData=1_TestData=1_Resolution=513X513_Domain=[0,1]X[0,1].hdf5"
normPATH = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/normalisers/poisson/UnitGaussianNormalizer/'
pcaPATH  = '../../../../../../localdata/Derick/stuart_data/Darcy_421/operators/pca/poisson/UnitGaussianNormalizer/'

noise_ratio = 0
F_test, U_test, F_train, U_train = readtoArray(fileName, 1, 1, Nx = 512, Ny = 512)
_, U_train_noisy, _, _      = add_noise(fileName, noise_ratio)
F_test = torch.from_numpy(np.array(F_test)).float().cuda()
U_test = torch.from_numpy(np.array(U_test)).float().cuda()

N_test = 500
U_test = U_test[0:N_test]
F_test = F_test[0:N_test]

#resolution
r = 64 + 1
a = [int(512/(r-1)) * i for i in range(r)]
f_test = F_test[:, a, :]
f_test = f_test[:, :, a].reshape(-1, r ** 2)
U_test =U_test[:,a,:]
U_test = U_test[:, :, a].reshape(-1, r ** 2)



X = torch.linspace(0,1, r)
X1, X2 = torch.meshgrid(X, X)
X1 = X1.reshape(r ** 2, 1)
X2 = X2.reshape(r ** 2, 1)
X = torch.cat((X1, X2), dim=1).to(device)

class FNNLinear(torch.nn.Module):
    def __init__(self, layer_sizes):
        super(FNNLinear, self).__init__()
        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(torch.nn.Linear(layer_sizes[i-1], layer_sizes[i],bias=False))

    def forward(self, x):
        for linear in self.linears[:-1]:
            x = linear(x)
        x = self.linears[-1](x)
        return x


class FNNt(torch.nn.Module):
    def __init__(self, layer_sizes):
        super(FNNt, self).__init__()
        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(torch.nn.Linear(layer_sizes[i-1], layer_sizes[i]))

    def forward(self, x):
        for linear in self.linears[:-1]:
            x = torch.tanh(linear(x))
        x = self.linears[-1](x)
        return x



p = 128
layer_branch = [r ** 2] + [p]
layer_trunck = [2] + [128] * 6 + [p]


Bran_nn = FNNLinear(layer_branch).to(device)
Trun_nn = FNNt(layer_trunck).to(device)

Bran_nn = torch.load("models/soft64branchPoissonLinear.pkl")
Trun_nn = torch.load("models/soft64trunkPoissonLinear.pkl")
finv = torch.ones(r * r,requires_grad=True,device='cuda').float()

k=0
#finv = f_train[k]
u_obs = U_test[k]

xmin, xmax = 0, 1  # -1, 1
ymin, ymax = 0, 1  # -1, 1
fig = plt.figure(figsize=(12, 5))
colourMap = parula()  # plt.cm.jet
plt.subplot(1, 2, 1)
plt.xlabel('x')  # , fontsize=16, labelpad=15)
plt.ylabel('y')  # , fontsize=16, labelpad=15)
plt.title('f')
plt.imshow(finv.reshape(r, r).detach().cpu(), cmap=colourMap, extent=[xmin, xmax, ymin, ymax], origin='lower',
           aspect='auto')  # , vmin=0, vmax=1, )
plt.colorbar()

plt.subplot(1, 2, 2)
plt.xlabel('x')  # , fontsize=16, labelpad=15)
plt.ylabel('y')  # , fontsize=16, labelpad=15)
plt.title("u")
plt.imshow(u_obs.reshape(r, r).cpu(), cmap=colourMap, extent=[xmin, xmax, ymin, ymax], origin='lower',
           aspect='auto')  # , vmin=0, vmax=1, )
plt.colorbar()
plt.show()
plt.savefig("1.png", dpi=500)


lr = 0.002
optimizers2 = torch.optim.Adam([{'params': finv, 'lr': lr}
                              ])
loss_func = torch.nn.MSELoss()

# A: the output of trunk net
A = Trun_nn(X)
X1 =X1.to(device)
X2 = X2.to(device)
A = A.detach()
print('A', A.shape)

for i in range(40000):
    branch_out = Bran_nn(finv).reshape(p,1)
    output = torch.matmul(A, branch_out.reshape(p,1)).reshape(r*r)
    loss1 = loss_func(output, u_obs) ** 0.5
    loss2 = torch.norm(finv)/r
    fs = finv.reshape(r,r)
    dx = (fs[1:r,:]-fs[0:r-1,:])*(r-1)
    lossDx = torch.norm(dx)/r
    dy = (fs[:,1:r] - fs[:,0:r - 1]) * (r - 1)
    lossDy = torch.norm(dy) / r
    Loss = loss1 + 0.002 *loss2 + 0.0001 *lossDx + 0.0001 * lossDy
    optimizers2.zero_grad()
    Loss.backward()
    optimizers2.step()
    if i % 1000==0:
        print(i, Loss.item(),loss1.item(), loss2.item(),lossDx.item(),lossDy.item())

xmin, xmax = 0, 1 #-1, 1
ymin, ymax = 0, 1 #-1, 1
fig = plt.figure(figsize=(12, 5))
colourMap = parula() #plt.cm.jet
plt.subplot(1, 2, 1)
plt.xlabel('x')  # , fontsize=16, labelpad=15)
plt.ylabel('y')  # , fontsize=16, labelpad=15)
plt.title("Approximation")
plt.imshow(finv.detach().reshape(r,r).cpu(), cmap=colourMap, extent=[xmin, xmax, ymin, ymax], origin='lower',
           aspect='auto')  # , vmin=0, vmax=1, )
plt.colorbar()

plt.subplot(1, 2, 2)
plt.xlabel('x')  # , fontsize=16, labelpad=15)
plt.ylabel('y')  # , fontsize=16, labelpad=15)
plt.title("Exact")
plt.imshow(f_test[k].reshape(r,r).cpu() , cmap=colourMap, extent=[xmin, xmax, ymin, ymax], origin='lower',
           aspect='auto')  # , vmin=0, vmax=1, )
plt.colorbar()
plt.show()
plt.savefig("2.png", dpi=500)
error = torch.norm(finv.detach().reshape(r,r).cpu()-f_test[k].reshape(r,r).cpu())/torch.norm(f_test[k].reshape(r,r).cpu())
print('error',error)