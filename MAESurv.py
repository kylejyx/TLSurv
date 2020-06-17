import torch
from torch import nn
from torch.nn import functional as F
from Coxnnet import Coxnnet_encoder

class MAESurv(nn.Module):
    """
    Class for creating a TTSurv(MAE)
    """
    def __init__(self, in_dims, d_dims, hidden_dims, do, ns, fuse_type='sum'): #List of input dimensions, list of embedding dimensions, latent dimension, dropout rate for survival network, neuron size in hidden layer for survival network, information fusion type('sum' or 'cat')
 
        super().__init__()
        self.num_views = len(in_dims)
        self.in_dims = in_dims
        self.d_dims = d_dims
        self.fuse_type = fuse_type
        self.hidden_dims = hidden_dims
        
        self.in1 = Coxnnet_encoder(in_dims[0], d_dims[0], 0)
        self.in2 = Coxnnet_encoder(in_dims[1], d_dims[1], 0)

        layers_one = [nn.Linear(d_dims[0], hidden_dims[0])]
        for i in range(1, len(hidden_dims)):
            layers_one.append(nn.ReLU())
            layers_one.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            #layers.Dropout(p=0.1)
        
        self.encoder_one = nn.Sequential(*layers_one)
        
        layers_two = [nn.Linear(d_dims[1], hidden_dims[0])]
        for i in range(1, len(hidden_dims)):
            layers_two.append(nn.ReLU())
            layers_two.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            #layers.Dropout(p=0.1)
        
        self.encoder_two = nn.Sequential(*layers_two)
        
        if self.fuse_type == 'sum':
            fuse_dim = hidden_dims[-1]
        elif self.fuse_type == 'cat':
            fuse_dim = hidden_dims[-1] * len(in_dims)
        else:
            raise ValueError(f"fuse_type should be 'sum' or 'cat', but is {fuse_type}")
        self.latent_size = fuse_dim
        self.surv_net = nn.Sequential(
            nn.Linear(self.latent_size, ns), nn.ReLU(), nn.BatchNorm1d(ns), nn.Dropout(do),
            nn.Linear(ns, 1, bias=False),
        )
        
    def forward(self, input_one, input_two):
        input_one = self.in1(input_one)
        input_two = self.in2(input_two)

        encoded_one = self.encoder_one(input_one)
        encoded_two = self.encoder_two(input_two)
        if self.fuse_type == 'sum':
            out = torch.stack([encoded_one, encoded_two], dim=-1).mean(dim=-1)
        else:
            out = torch.cat([encoded_one, encoded_two], dim=-1)
        return self.surv_net(out)

