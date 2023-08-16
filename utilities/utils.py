import torch
import numpy as np
import scipy.io
import h5py
import sklearn.metrics
import torch.nn as nn
from scipy.ndimage import gaussian_filter

from prettytable import PrettyTable

import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy import ndimage
import os
#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.optim.sgd import SGD
from torch.optim.optimizer import required
import math

from utilities.colorMap import parula

def round_to_multiple(number, multiple, direction='up'):
    if direction == 'nearest':
        result = multiple * round(number / multiple)
    elif direction == 'up':
        result = multiple * math.ceil(number / multiple)
    elif direction == 'down':
        result = multiple * math.floor(number / multiple)
    else:
        result = multiple * round(number / multiple)
    #print(result)
    #return result
    
    result = number+multiple if result == number else result #ensuring that same number is never returned, i.e case where 'number' is a multiple of the variable 'multiple'
    return result


def closest_power(n, base = 2):
    pow = math.log(n,base)
    #pow = math.floor(pow)-1 if pow < 0.584963 else math.floor(pow)
    k = math.floor(pow)
    ref = 3*2**(k-1)
    pow = k if n < ref else k+1

    return 2**pow

def plot_comparism(pb, no, noise_ratio, out_masked, yout, X_train, Y_train_noisy, Y_train, ModelInfos, num_samp, res, myloss, accuracy, ResultsDir=''):
    params = dict()
    params["xmin"], params["xmax"], params["ymin"], params["ymax"] = 0, 1, 0, 1
        
    X_learned = out_masked.reshape(num_samp, res, res).detach().cpu().numpy()
    Y_learned = yout.reshape(num_samp, res, res).detach().cpu().numpy()

    accuracy = 100*np.sum(out_masked.flatten() == X_train.flatten())/len(out_masked.flatten())

    print("   Ploting comparism of FDM and %s-%s Simulation results"%(no, pb))
    fig = plt.figure(figsize=((5+1)*3, (5)*3))
    fig.set_tight_layout(True)

    # if pb[0:5] == 'darcy':
    #     suptitle = r"Plot of $-\nabla \cdot (\lambda(s) \nabla u(s)) = f(s), \partial \Omega = 0$ with $u|_{\partial \Omega}  = 0.$"
    # if pb == 'poisson':
    #     suptitle = "Plot of $- \Delta u = \lambda(x, y)$ on $\Omega = ]0,1[ x ]0,1[$ with $u|_{\partial \Omega}  = 0.$"
        
    # fig.suptitle(suptitle)

    colourMap = plt.cm.magma#parula() #plt.cm.jet #plt.cm.coolwarm

    plt.subplot(3, 3, 1)
    plt.xlabel('x')#, fontsize=16, labelpad=15)
    plt.ylabel('y')#, fontsize=16, labelpad=15)
    plt.title(r"Parameter, $\lambda(s)$")
    plt.imshow(X_train[0], cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
    plt.colorbar()#format=OOMFormatter(-5))

    plt.subplot(3, 3, 2)
    plt.xlabel('x')#, fontsize=16, labelpad=15)
    plt.ylabel('y')#, fontsize=16, labelpad=15)
    plt.title(r"Learned Parameter, $\hat \lambda(s)$")
    plt.imshow(X_learned[0], cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
    plt.colorbar()#format=OOMFormatter(-5))

    if pb == 'darcyPWC':
        title = r"$|\lambda(s) - \hat \lambda(s)|$, RelL2Err = %.4f, Acc = %.2f %%"%(myloss.rel_single(X_learned[0], X_train[0]).item(), accuracy)
    else:
        title = r"$|\lambda(s) - \hat \lambda(s)|$, RelL2Err = %.4f"%(myloss.rel_single(X_learned[0], X_train[0]).item())

    plt.subplot(3, 3, 3)
    plt.xlabel('x')#, fontsize=16, labelpad=15)
    plt.ylabel('y')#, fontsize=16, labelpad=15)
    plt.title(title)
    plt.imshow(np.abs(X_learned[0] - X_train[0]), cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto', norm=matplotlib.colors.LogNorm())#)#, vmin=0, vmax=1, )
    plt.colorbar()#format=OOMFormatter(-5))

    plt.subplot(3, 3, 4)
    plt.xlabel('x')#, fontsize=16, labelpad=15)
    plt.ylabel('y')#, fontsize=16, labelpad=15)
    plt.title(r"Noisy Solution, $\tilde u(s)$")
    plt.imshow(Y_train_noisy[0], cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
    plt.colorbar()#format=OOMFormatter(-5))

    plt.subplot(3, 3, 5)
    plt.xlabel('x')#, fontsize=16, labelpad=15)
    plt.ylabel('y')#, fontsize=16, labelpad=15)
    plt.title(r"Learned Solution, $\hat u(s)$")
    plt.imshow(Y_learned[0], cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
    plt.colorbar()#format=OOMFormatter(-5))

    plt.subplot(3, 3, 6)
    plt.xlabel('x')#, fontsize=16, labelpad=15)
    plt.ylabel('y')#, fontsize=16, labelpad=15)
    plt.title(r"$|\tilde u(s) - \hat u(s)|$, RelL2Err = "+str(round(myloss.rel_single(Y_learned[0], Y_train_noisy[0]).item(), 3)))
    plt.imshow(np.abs(Y_learned[0] - Y_train_noisy[0]), cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto', norm=matplotlib.colors.LogNorm())#)#, vmin=0, vmax=1, )
    plt.colorbar()#format=OOMFormatter(-5))

    plt.subplot(3, 3, 7)
    plt.xlabel('x')#, fontsize=16, labelpad=15)
    plt.ylabel('y')#, fontsize=16, labelpad=15)
    plt.title("Solution, $u(s)$")
    plt.imshow(Y_train[0], cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
    plt.colorbar()#format=OOMFormatter(-5))

    plt.subplot(3, 3, 8)
    plt.xlabel('x')#, fontsize=16, labelpad=15)
    plt.ylabel('y')#, fontsize=16, labelpad=15)
    plt.title(r"Learned Solution, $\hat u(s)$")
    plt.imshow(Y_learned[0], cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
    plt.colorbar()#format=OOMFormatter(-5))

    plt.subplot(3, 3, 9)
    plt.xlabel('x')#, fontsize=16, labelpad=15)
    plt.ylabel('y')#, fontsize=16, labelpad=15)
    plt.title(r"$|u(s) - \hat u(s)|$, RelL2Err = "+str(round(myloss.rel_single(Y_learned[0], Y_train[0]).item(), 3)))
    plt.imshow(np.abs(Y_learned[0] - Y_train[0]), cmap=colourMap, extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]], origin='lower', aspect = 'auto', norm=matplotlib.colors.LogNorm())#)#, vmin=0, vmax=1, )
    plt.colorbar()#format=OOMFormatter(-5))

    directory = ResultsDir +'figures/'+pb+'/noiseRatio=%s'%(noise_ratio)+'/'+no
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory+'/compare'+ModelInfos+'.png',dpi=500)


def FDM_Darcy(u, a, D=1, f=1):
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    dx = D / (size - 1)
    dy = dx

    # ux: (batch, size-2, size-2)
    ux = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
    uy = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dy)

    ax = (a[:, 2:, 1:-1] - a[:, :-2, 1:-1]) / (2 * dx)
    ay = (a[:, 1:-1, 2:] - a[:, 1:-1, :-2]) / (2 * dy)
    uxx = (u[:, 2:, 1:-1] -2*u[:,1:-1,1:-1] +u[:, :-2, 1:-1]) / (dx**2)
    uyy = (u[:, 1:-1, 2:] -2*u[:,1:-1,1:-1] +u[:, 1:-1, :-2]) / (dy**2)

    a = a[:, 1:-1, 1:-1]
    u = u[:, 1:-1, 1:-1]
    # Du = -(ax*ux + ay*uy + a*uxx + a*uyy)

    # inner1 = torch.mean(a*(ux**2 + uy**2), dim=[1,2])
    # inner2 = torch.mean(f*u, dim=[1,2])
    # return 0.5*inner1 - inner2

    aux = a * ux
    auy = a * uy
    auxx = (aux[:, 2:, 1:-1] - aux[:, :-2, 1:-1]) / (2 * dx)
    auyy = (auy[:, 1:-1, 2:] - auy[:, 1:-1, :-2]) / (2 * dy)
    Du = - (auxx + auyy)

    return Du


def darcy_PINO_loss(u, a):
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    lploss = LpLoss(size_average=True)

    index_x = torch.cat([torch.tensor(range(0, size)), (size - 1) * torch.ones(size), torch.tensor(range(size-1, 1, -1)),
                         torch.zeros(size)], dim=0).long()
    index_y = torch.cat([(size - 1) * torch.ones(size), torch.tensor(range(size-1, 1, -1)), torch.zeros(size),
                         torch.tensor(range(0, size))], dim=0).long()

    boundary_u = u[:, index_x, index_y]
    truth_u = torch.zeros(boundary_u.shape, device=u.device)
    loss_bd = lploss.abs(boundary_u, truth_u)

    Du = FDM_Darcy(u, a)
    f = torch.ones(Du.shape, device=u.device)
    loss_f = lploss.rel(Du, f)

    # im = (Du-f)[0].detach().cpu().numpy()
    # plt.imshow(im)
    # plt.show()

    # loss_f = FDM_Darcy(u, a)
    # loss_f = torch.mean(loss_f)
    return loss_f, loss_bd

def FDM_Poisson(u, D=1):
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    dx = D/(size-1)
    dy = dx

    P1 = (1/dx**2) * (-u[:, 1:-1, 2:] + 2*u[:, 1:-1, 1:-1] - u[:, 1:-1, :-2])
    P2 = (1/dy**2) * (-u[:, 2:, 1:-1] + 2*u[:, 1:-1, 1:-1] - u[:, :-2, 1:-1])
 
    return P1 + P2


def poisson_PINO_loss(u, f):
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    f = f.reshape(batchsize, size, size)
    f = f[:, 1:-1, 1:-1]
    lploss = LpLoss(size_average=True)

    index_x = torch.cat([torch.tensor(range(0, size)), (size - 1) * torch.ones(size), torch.tensor(range(size-1, 1, -1)),
                         torch.zeros(size)], dim=0).long()
    index_y = torch.cat([(size - 1) * torch.ones(size), torch.tensor(range(size-1, 1, -1)), torch.zeros(size),
                         torch.tensor(range(0, size))], dim=0).long()

    boundary_u = u[:, index_x, index_y]
    truth_u = torch.zeros(boundary_u.shape, device=u.device)
    loss_bd = lploss.abs(boundary_u, truth_u)

    Du = FDM_Poisson(u)
    rhs = torch.ones(Du.shape, device=u.device)
    #f = torch.ones(Du.shape, device=u.device)
    loss_f = lploss.rel(Du-f+rhs, rhs)
    return loss_f, loss_bd

# def FDM_Darcy(u, a, D=1):
#     batchsize = u.size(0)
#     size = u.size(1)
#     u = u.reshape(batchsize, size, size)
#     a = a.reshape(batchsize, size, size)
#     dx = D / (size - 1)
#     dy = dx

#     ux = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
#     uy = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dy)

#     a = a[:, 1:-1, 1:-1]

#     aux = a * ux
#     auy = a * uy
#     auxx = (aux[:, 2:, 1:-1] - aux[:, :-2, 1:-1]) / (2 * dx)
#     auyy = (auy[:, 1:-1, 2:] - auy[:, 1:-1, :-2]) / (2 * dy)
#     Du = - (auxx + auyy)
#     return Du


# def darcy_loss(u, a):
#     batchsize = u.size(0)
#     size = u.size(1)
#     u = u.reshape(batchsize, size, size)
#     a = a.reshape(batchsize, size, size)
#     lploss = LpLoss(size_average=True)


#     Du = FDM_Darcy(u, a)
#     f = torch.ones(Du.shape, device=u.device)
#     loss_f = lploss.rel(Du, f)

#     return loss_f


def darcy_mask1(x):
    return 1 / (1 + torch.exp(-x)) * 9 + 3
def identity_mask(x):
    return x

def darcy_mask2(x):
    x = 1 / (1 + torch.exp(-x))
    x[x>0.5] = 0
    x[x<=0.5] = 1
    # x = torch.tensor(x>0.5, dtype=torch.float)
    return  x * 9 + 3

def total_variance(x):
    return torch.mean(torch.abs(x[...,:-1] - x[...,1:])) + torch.mean(torch.abs(x[...,:-1,:] - x[...,1:,:]))     

def total_variance_1d(x):
    L2loss = LpLoss()
    # return torch.mean(torch.abs(x[...,:-1] - x[...,1:])) + L2loss.abs(x[...,:-1,:], x[...,1:,:]) #f2nd try
    return torch.mean(torch.abs(x[...,:-1] - x[...,1:])) #first try

def total_variation_loss(img, weight=1):
    bs, h, w = img.size()
    tv_h = torch.pow(img[:, 1:, :]-img[:, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :,1 :]-img[:, :, :-1], 2).sum()
    return weight * (tv_h + tv_w)/(bs * h *w)


def tvl2(x, weight=1):
    L2loss = LpLoss()
    # bs, h, w = img.size()
    # tv_h = torch.norm(img[..., 1:, :]-img[..., :-1, :])
    # tv_w = torch.norm(img[..., 1:]-img[..., :-1])
    return weight * (L2loss.abs(x[...,:-1],  x[...,1:]) + L2loss.abs(x[...,:-1,:], x[...,1:,:]))


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print("Total Trainable Params: %s"%(total_params))
    return total_params

    
        
class PGM(SGD): # Source https://gist.github.com/pmelchior/f371dda15df8e93776e570ffcb0d1494        
    def __init__(self, params, proxs, lr=required, momentum=0, dampening=0,
                 nesterov=False):
        kwargs = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=0, nesterov=nesterov)
        super().__init__(params, **kwargs)

        if len(proxs) != len(self.param_groups):
            raise ValueError("Invalid length of argument proxs: {} instead of {}".format(len(proxs), len(self.param_groups)))

        for group, prox in zip(self.param_groups, list(proxs)):
            group.setdefault('prox', prox)

    def step(self, closure=None):
        # this performs a gradient step
        # optionally with momentum or nesterov acceleration
        super().step(closure=closure)

        for group in self.param_groups:
            prox = group['prox']

            # here we apply the proximal operator to each parameter in a group
            for p in group['params']:
                p.data = prox(p.data)   


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%.3f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
             self.format = r'$\mathdefault{%s}$' % self.format
             
def SubSample(A, ny, nx):
    _, Ny, Nx = A.shape
    Ny = Ny-1
    Nx = Nx-1
    ny = ny-1
    nx = nx-1

    if Nx%nx==0 and Ny%ny==0:
        if Nx==nx and Ny==ny:
            return A
        else:
            stepx = Nx//nx
            stepy = Ny//ny

            idx = np.arange(0, Nx+1, stepx)
            idy = np.arange(0, Ny+1, stepy)
            
            A = A[:, :, idx]
            A = A[:, idy]

            return A
    else:
        raise ValueError("The array cannot be downsampled to the requested shape")
    
def SubSample1D(A, nx):
    _, Nx = A.shape
    Nx = Nx-1
    nx = nx-1

    if Nx%nx==0:
        if Nx==nx:
            return A
        else:
            stepx = Nx//nx

            idx = np.arange(0, Nx+1, stepx)
            
            A = A[:, idx]

            return A
    else:
        raise ValueError("The array cannot be downsampled to the requested shape")

def CubicSpline3D(A, ny, nx):
    batch, Ny, Nx = A.shape
    Ny = Ny-1
    Nx = Nx-1
    ny = ny-1
    nx = nx-1
    if Nx%nx==0 and Ny%ny==0:
        if Nx==nx and Ny==ny:
            return A
        else:
            return SubSample(A, ny+1, nx+1)
    else:
        first = []
        for i in range(batch):
            first.append([i]*((nx+1)*(nx+1)))
        first = np.array(first).flatten()
        downx = np.linspace(0, Nx, nx+1)
        downy = np.linspace(0, Ny, ny+1)
        Xdown, Ydown = np.meshgrid(downx, downy)
        Xdown = np.tile(Xdown.flatten(),batch)
        Ydown = np.tile(Ydown.flatten(),batch)

        A = ndimage.map_coordinates(A, [first, Ydown, Xdown]).reshape(batch,nx+1,nx+1)
        
        return A

        
#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()#.to(device)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()#.to(device) 
               
        num_examples = x.size()[0]
        if num_examples == 0:
            num_examples +=1

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()#.to(device)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()#.to(device)    
            
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms
        
    def rel_single(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()#.to(device)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()#.to(device)

        diff_norms = torch.norm(x - y, self.p)
        y_norms = torch.norm(y, self.p)
        
        return diff_norms/y_norms
            

    def __call__(self, x, y):
        return self.rel(x, y)    
        
        

             
# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class UnitRangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x
        
# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x)#, 0)[0].view(-1)
        mymax = torch.max(x)#, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()

    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()
        

# A simple feedforward neural network
class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x
        
        