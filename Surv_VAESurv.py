"""
Code for training TTSurv(VAE)
Example usage: python Surv_VAESurv.py mirna mirna epochs learning_rate l2_regularization batch_size
Please ensure there is no space in the list if a list of candidates is provided for any hyper-parameter.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn
import torch.nn.functional as F
import torchtuples as tt

from pycox.models import CoxPH 
from pycox.evaluation import EvalSurv

import pickle
import pandas as pd
import sys
import io
import os
import gc
import time

from VAESurv import VAESurv
from utils import read_data
from utils import parse_args
DATA_FOLDER = os.path.expanduser('~/LUAD/')
DATA_SUFFIX = "_TT"
INDEX_SET = "2"
STATE_FOLDER = os.path.expanduser('~/states_VAE_LUAD_set2/')
SAVE_FOLDER = os.path.expanduser('~/states_VAE_LUAD_set2/')

LOG_FOLDER = os.path.expanduser('./log_VAESurv_set2/')
PREFIX = ""


device=torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
print(device)
np.random.seed(1234)
_ = torch.manual_seed(123)

def VAESurv_pipeline(mod1, mod2, x1, x2, y1, y2, hyperparameters, save, path):
    d_dims = hyperparameters['D dims']
    dense_size = hyperparameters['Dense size']#number of nodes in dense layers
    latent_size = hyperparameters['Latent size'] # number of nodes (dimensionality) of encoded data
    neuron_size = hyperparameters['Neuron size'] # Dimensions for survival network
    dropout_p = hyperparameters['Dropout']

    in_features_one = x1.shape[1]
    if mod2 != 'None':
        in_features_two = x2.shape[1]
        net = VAESurv(in_features_one, in_features_two, d_dims, dense_size, latent_size, dropout_p, neuron_size, device).to(device)
    else:
        pass
    #Load pre-trained VAE
    if mod2 != 'None':
        PATH=hyperparameters['State file path']+mod1+'+'+mod2+'.pt'
    else:
        PATH=hyperparameters['State file path']+mod1+'.pt'
    net.load_state_dict(torch.load(PATH), strict=False)
    net.eval()
    
    for name, param in net.named_parameters():
        if not('surv_net' in name):
            param.requires_grad = False
    """    
    print('Trainable parameters:')
    for name, param in net.named_parameters():
        if (param.requires_grad):
            print(name)
    """
    net.train()
    model = CoxPH(net, tt.optim.Adam) #loss = 
    
    batch_size = hyperparameters['batch_size'] 
    epochs = hyperparameters['Epoch']
    verbose = True
    model.optimizer.set_lr(hyperparameters['Learning rate'])
    model.optimizer.set('weight_decay', hyperparameters['L2 reg'])
    if mod2 != 'None':
        log = model.fit((x1,x2),(y1,y2), batch_size, epochs, verbose=verbose)
    else:
        log = model.fit(x1,(y1,y2), batch_size, epochs, verbose=verbose)
    net.eval()
    if save:
        PATH = SAVE_FOLDER + "VAESurv_"+ path 
        torch.save(net.state_dict(), PATH)
    return model, log


def VAESurv_evaluate(model, x1, x2, durations, events):
    durations = durations.cpu().numpy()
    events = events.cpu().numpy()
    _ = model.compute_baseline_hazards()
    if x2 is not None:
        surv = model.predict_surv_df((x1, x2))
    else:
        surv = model.predict_surv_df(x1)
    
    ev = EvalSurv(surv, durations, events, censor_surv='km')
    return ev.concordance_td()
    
def CV(mod1, mod2, x1, x2, y1, y2, folds, hyperparameters):
    kf = KFold(n_splits=folds, shuffle=True, random_state=71)
    for train_index, test_index in kf.split(x1):
        x1_train = x1[train_index]
        x1_test = x1[test_index]
        if x2 is not None:
            x2_train = x2[train_index]
            x2_test = x2[test_index]
        else:
            x2_train = None
            x2_test = None
        y1_train = y1[train_index]
        y1_test = y1[test_index]
        y2_train = y2[train_index]
        y2_test = y2[test_index]
        
        p_time = time.time()
        model, log = VAESurv_pipeline(mod1, mod2, x1_train, x2_train, y1_train, y2_train, hyperparameters, False, None)
        a_time = time.time()
        print(f'Training time: {a_time-p_time}')
        
        Cindex = VAESurv_evaluate(model, x1_test, x2_test, y1_test, y2_test)
        print(f"Val C-index: {Cindex}")
                    
def main(mod1, mod2, epochs, learning_rate, l2s, batch_size, network='VAESurv'):
    if mod2 != 'None':
        VAE_params = pickle.load(io.open(STATE_FOLDER+mod1+'+'+mod2+'.dict', 'rb'))        
    else:
        VAE_params = pickle.load(io.open(STATE_FOLDER+mod1+'.dict', 'rb'))        
    
    x1, x2, index = read_data(DATA_FOLDER, mod1, mod2, suffix=DATA_SUFFIX)
    with open(DATA_FOLDER+'train'+INDEX_SET+'.pickle','rb') as f:
        train_index = pickle.load(f)
    with open(DATA_FOLDER+'test'+INDEX_SET+'.pickle','rb') as f:
        test_index = pickle.load(f)
    x1_train = x1.loc[train_index].to_numpy()#x1_train = torch.from_numpy(x1.loc[train_index].to_numpy()).float().to(device)
    in_features_one = x1_train.shape[1]
    x1_test = x1.loc[test_index].to_numpy()#x1_test = torch.from_numpy(x1.loc[test_index].to_numpy()).float().to(device)
    x1_train = torch.from_numpy(x1_train).float().to(device)
    x1_test = torch.from_numpy(x1_test).float().to(device)
    if x2 is not None:
        x2_train = x2.loc[train_index].to_numpy()
        in_features_two = x2_train.shape[1]
        x2_test = x2.loc[test_index].to_numpy()
        x2_train = torch.from_numpy(x2_train).float().to(device)
        x2_test = torch.from_numpy(x2_test).float().to(device)
    else:
        x2_train = None
        in_features_two = None
        x2_test = None
    with open(DATA_FOLDER + 'survival_TT.pickle', 'rb') as f:
        survival=pickle.load(f)
    get_target = lambda df: (df['OS.time'].values, df['OS'].values)
    y1_train, y2_train = get_target(survival.loc[train_index])
    y1_test, y2_test = get_target(survival.loc[test_index])    
    y1_train = torch.from_numpy(y1_train).to(device)
    y2_train = torch.from_numpy(y2_train).to(device)
    y1_test = torch.from_numpy(y1_test).to(device)
    y2_test = torch.from_numpy(y2_test).to(device)
    
    for epoch in epochs:
        for do in [0,0.1,0.2]:
            for l2 in l2s:
                for lr in learning_rate:
                    for bs in batch_size:
                        for ns in [16,32,64,100]:
                            hyperparameters = {'Epoch': epoch,
                            'Dropout': do,
                            'L2 reg': l2,
                            'State file path': STATE_FOLDER,
                            'Learning rate': lr, #tt.optim.Adam
                            'batch_size': bs,
                            'Neuron size': ns
                            }
                            hyperparameters.update(VAE_params)
                            
                            stdOrigin = sys.stdout
                            PATH = mod1+" "+str(mod2)+"_"+str(epoch)+"_"+str(do)+"_"+str(l2)+"_"+str(lr)+"_"+str(bs)+"_"+str(ns)
                            sys.stdout = open(os.path.join(LOG_FOLDER, "VAESurv_"+PATH+".out"), "w")
                            print(hyperparameters)
                           
                            CV(mod1, mod2, x1_train, x2_train, y1_train, y2_train, 5, hyperparameters)
                            
                            p_time = time.time()
                            model, log = VAESurv_pipeline(mod1, mod2, x1_train, x2_train, y1_train, y2_train, hyperparameters, True, PATH)
                            a_time = time.time()
                            print(f'Training time: {a_time-p_time}')
                            Cindex = VAESurv_evaluate(model, x1_test, x2_test, y1_test, y2_test)
                            print(f"Test C-index: {Cindex}")
                            
                            gc.collect()
                            sys.stdout.close()
                            sys.stdout = stdOrigin




if (__name__ == '__main__'):
    print(sys.argv)
    parameters = parse_args(sys.argv)
    main(*parameters)
