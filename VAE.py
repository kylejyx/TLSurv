import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class VAE(nn.Module):
    """
    Class for creating a Variational Autoencoder(VAE) for two modalities
    """
    def __init__(self, d_dims, en1, en2, ds, ls, dp, device): #list of embedding dimensions, embedding network 1, embedding network 2, dense_size, latent dimension, dropout rate, device for training
        super().__init__()

        self.device = device
        self.i1 = en1
        self.i2 = en2
        self.d_dims = d_dims
        
        ## embedding
        ds = d_dims[0]+d_dims[1]
        self.embed_z_mean = nn.Linear(ds, ls)
        self.embed_z_log_sigma = nn.Linear(ds, ls)


        ## decoder
        self.decode1 = nn.Linear(ls, ds)
        self.decBN1 = nn.BatchNorm1d(ds)
        self.dropout = nn.Dropout(p=dp)
        ## split to two output branches
        self.out1 = nn.Linear(ls, d_dims[0])
        self.out2 = nn.Linear(ls, d_dims[1])


    def sample(self, z_mean, z_log_var):
        batch = z_mean.size()[0]
        dim = z_mean.size()[1]
        eps = np.random.normal(size=(batch,dim))
        ret = z_mean + (0.5 * z_log_var).exp() * torch.tensor(eps).float().to(self.device)
        return ret


    def forward(self, input_one, input_two):
        input_one = self.i1(input_one)
        input_two = self.i2(input_two)
        act = nn.ELU() ## elu activation

        x = torch.cat((input_one, input_two), dim=1)

        z_mean = act(self.embed_z_mean(x))
        z_log_sigma = act(self.embed_z_log_sigma(x))
        z = self.sample(z_mean, z_log_sigma)
        intermediates = [z_mean, z_log_sigma, z]

        latent_inputs = intermediates[2]
        x = latent_inputs
        #x = self.dropout(x)

        o1 = self.out1(x)
        o2 = self.out2(x)

        return intermediates[0], intermediates[1], o1, o2, input_one, input_two


def compkernel(x, y):
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]
    tile_x = torch.reshape(x, (x_size, 1, dim))
    tx = tile_x
    for n in range(y_size - 1):
        torch.cat((tx, tile_x), dim=1)

    tile_y = torch.reshape(y, (1, y_size, dim))
    ty = tile_y
    for m in range(x_size - 1):
        torch.cat((ty, tile_y), dim=0)

    kernel_input = (-torch.mean((tx - ty).pow(2), 2)).exp() / float(dim)
    return kernel_input


def mmd(x, y):
    x_kernel = compkernel(x, x)
    y_kernel = compkernel(y, y)
    xy_kernel = compkernel(x, y)
    mmd = torch.mean(x_kernel) + torch.mean(y_kernel) - 2*torch.mean(xy_kernel)
    return mmd


def loss_function(z1, x1, z2, x2, mu, logvar, beta):
    CE = F.mse_loss(z1, x1, reduction='mean')
    CE2 = F.mse_loss(z2, x2, reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    ##MMD = mmd(z1, x1) + mmd(z2, x2)
    return torch.mean(CE + CE2 + beta * KLD)
