import torch
import torch.nn as nn

class Coxnnet(nn.Module):
    """
    Class for creating a complete Cox-nnet network
    """
    def __init__(self, in_f, ds, dp): #Input dimension, neuron size in hidden layer, dropout rate
        super().__init__()
        
        self.layer1 = nn.Linear(in_f, ds)
        self.dropout1 = nn.Dropout(p=dp)
        self.layer2 = nn.Linear(ds, 1, bias=False)
        
        
    def forward(self, input):
        act = nn.Tanh()
        
        x=self.layer1(input)
        x=act(x)
        x=self.dropout1(x)
        x=self.layer2(x)
        
        return x


class Coxnnet_encoder(nn.Module):
    """
    Class for creating embedding network
    """
    def __init__(self, in_f, ds, dp):  #Input dimension, neuron size in hidden layer, dropout rate
        super().__init__()
        self.layer1 = nn.Linear(in_f, ds)
        
    def forward(self, input):
        act = nn.Tanh()
        
        x=self.layer1(input)
        x=act(x)
        return x
        
