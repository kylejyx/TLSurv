import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from pycox.models.loss import NLLLogistiHazardLoss
from Coxnnet import Coxnnet_encoder


class VAESurv(nn.Module):
    """
    Class for creating a TTSurv(VAE)
    """
    def __init__(self, in_1,in_2,d_dims, ds, ls, dropout_p, ns, device): #input dimensions, list of embedding dimensions, dense_size, latent dimension, dropout rate, neuron size in hidden layer for survival network,  device for training

        super().__init__()
        
        self.device = device
        self.i1 = Coxnnet_encoder(in_1, d_dims[0], 0)
        self.i2 = Coxnnet_encoder(in_2, d_dims[1], 0)
        self.d_dims = d_dims
  
        ## embedding
        ds = d_dims[0]+d_dims[1]
        self.embed_z_mean = nn.Linear(ds, ls)
        self.embed_z_log_sigma = nn.Linear(ds, ls)

        self.surv_net = nn.Sequential(
            nn.Linear(ls, ns), nn.ReLU(), nn.BatchNorm1d(ns),
            nn.Dropout(dropout_p),
            nn.Linear(ns, 1, bias=False)
        )
    
    def sample(self, z_mean, z_log_var):
        batch = z_mean.size()[0]
        dim = z_mean.size()[1]
        eps = np.random.normal(size=(batch,dim))
#        return (z_mean.data.cpu() + np.exp(0.5 * z_log_var.data.cpu()) * eps).float().to(self.device)
        ret = z_mean + (0.5 * z_log_var).exp() * torch.tensor(eps).float().to(self.device)
        return ret



    def forward(self, input_one, input_two):
        input_one = self.i1(input_one)
        input_two = self.i2(input_two)
        act = nn.ELU() ## elu activation
        x = torch.cat((input_one,input_two),dim=1)

        z_mean = self.embed_z_mean(x)
        z_log_sigma = self.embed_z_log_sigma(x)
        z = self.sample(z_mean, z_log_sigma)
        #intermediates = [z_mean, z_log_sigma, z]
        #return intermediates[0], intermediates[1], o1, o2
        
        encoded = z 
        phi = self.surv_net(encoded)
        return phi

class LossLogHaz(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        assert (alpha >= 0) and (alpha <= 1), 'Need `alpha` in [0, 1].'
        self.alpha = alpha
        self.loss_surv = NLLLogistiHazardLoss()

        
    def forward(self, phi, idx_durations, events):
        loss_surv = self.loss_surv(phi, idx_durations, events) 
        return self.alpha * loss_surv
