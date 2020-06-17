"""
Code for training fusion network for TTSurv(VAE)
Example: python Surv_train_VAE.py mirna mrna epochs batch_size beta dense_size latent_size dropout 
Please ensure there is no space in the list if a list of candidates is provided for any hyper-parameter.
"""
import sys
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
np.random.seed(1234)
_ = torch.manual_seed(123)
from VAE import VAE, loss_function
from Coxnnet import Coxnnet_encoder
from utils import read_data
from utils import parse_args

import pickle
import os
import io
import gc
import time
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

DATA_FOLDER = os.path.expanduser('~/LUAD/')
DATA_SUFFIX = "_TT"
INDEX_SET = "2"
STATE_FOLDER = os.path.expanduser('~/states_Coxnnet_set2/')
SAVE_FOLDER = os.path.expanduser('~/states_VAE_LUAD_set2/')
LOG_FOLDER = os.path.expanduser('./log_VAE_set2/')

def train_VAE(mod1, mod2, d_dims, encoder1, encoder2, x1, x2, epoch, batch_size, beta, dense_size, latent_size, dropout, save, path):
    torch.cuda.empty_cache()
    if mod2 == 'None':
        model = VAE_single(in_features_one, dense_size, latent_size, dropout, device).to(device)
    else:
        model = VAE(d_dims, encoder1, encoder2, dense_size, latent_size, dropout, device).to(device)
    print('Trainable parameters:')
    for name, param in model.named_parameters():
        if (param.requires_grad):
            print(name)
    model.train()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=0)
    for j in range(epoch):
        total_loss = 0.
        for i in range(0, len(x1), batch_size):
            if mod2 == 'None':
                y_pred = model(x1[i:i+batch_size])
                loss = loss_function_single(y_pred[2], y_pred[3], y_pred[0], y_pred[1], beta)
            else:
                y_pred = model(x1[i:i+batch_size], x2[i:i+batch_size])
                loss = loss_function(y_pred[2], y_pred[4], y_pred[3], y_pred[5], y_pred[0], y_pred[1], torch.tensor(beta).to(device))
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()       
        print(f"Epoch: {j+1} loss: {total_loss}")
    if save:
        PATH = SAVE_FOLDER + path 
        hyperparameters = {'D dims': d_dims,
                'Dense size': dense_size,
                'Latent size': latent_size,
                'Beta': beta}
        with open(PATH + ".dict", "wb") as f:
            pickle.dump(hyperparameters, f)
        torch.save(model.state_dict(), PATH + '.pt')
    return model

def evaluate_VAE(model, x1, x2, batch_size):
    testloss = 0.
    with torch.no_grad():
        model.eval()
        for i in range(0, len(x1), batch_size):
            if x2 is None:
                y_pred = model(x1[i:i+batch_size])
                loss = loss_function_single(y_pred[2], y_pred[3], y_pred[0], y_pred[1], 0)
            else:
                y_pred = model(x1[i:i+batch_size], x2[i:i+batch_size])
                loss = loss_function(y_pred[2], y_pred[4], y_pred[3], y_pred[5], y_pred[0], y_pred[1], torch.tensor(0).to(device))
            testloss += loss.item()
    return testloss

def CV(mod1, mod2, e1, e2, x1, x2, d_dims, folds, epoch, bs, beta, ds, ls, do):
    kf = KFold(n_splits=folds, shuffle=True, random_state=66)
    for train_index, test_index in kf.split(x1):
        x1_train = x1[train_index]
        x1_test = x1[test_index]
        if mod2 != "None":
            x2_train = x2[train_index]
            x2_test = x2[test_index]
        else:
            x2_train = None
            x2_test = None
        p_time = time.time()
        model = train_VAE(mod1, mod2, d_dims, e1,e2, x1_train, x2_train, epoch, bs, beta, ds, ls, do, False, None)
        a_time = time.time()
        print(f'Training time: {a_time-p_time}')
        testloss = evaluate_VAE(model, x1_test, x2_test, bs)
        print(f"Validationloss: {testloss}")
                    
def main(mod1, mod2, epochs, batch_sizes, betas, dense_sizes, latent_sizes, dropouts):
    x1, x2, index = read_data(DATA_FOLDER, mod1, mod2, suffix=DATA_SUFFIX)
    with open(DATA_FOLDER+'train'+INDEX_SET+'.pickle','rb') as f:
        train_index = pickle.load(f)
    with open(DATA_FOLDER+'test'+INDEX_SET+'.pickle','rb') as f:
        test_index = pickle.load(f)
    x1_train = x1.loc[train_index].to_numpy()#x1_train = torch.from_numpy(x1.loc[train_index].to_numpy()).float().to(device)
    in_dims = [x1_train.shape[1]]
    x1_test = x1.loc[test_index].to_numpy()#x1_test = torch.from_numpy(x1.loc[test_index].to_numpy()).float().to(device)
    x1_train = torch.from_numpy(x1_train).float().to(device)
    x1_test = torch.from_numpy(x1_test).float().to(device)
    if x2 is not None:
        x2_train = x2.loc[train_index].to_numpy()#x2_train = torch.from_numpy(x2.loc[train_index].to_numpy()).float().to(device)
        x2_test = x2.loc[test_index].to_numpy()#x2_test = torch.from_numpy(x2.loc[test_index].to_numpy()).float().to(device)
        in_dims.append(x2_train.shape[1])
        x2_train = torch.from_numpy(x2_train).float().to(device)
        x2_test = torch.from_numpy(x2_test).float().to(device)
    else:
        x2_train = None
        in_features_two = None
        x2_test = None
    d_dims=[pickle.load(io.open(os.path.join(STATE_FOLDER, mod1+'.dict'), 'rb'))['Dense size']]
    d_dims.append(pickle.load(io.open(os.path.join(STATE_FOLDER, mod2+'.dict'), 'rb'))['Dense size'])
    
    encoder1 = Coxnnet_encoder(in_dims[0], d_dims[0], 0).to(device)
    encoder2 = Coxnnet_encoder(in_dims[1], d_dims[1], 0).to(device)
    #Load pre-trained AE
    PATH=os.path.join(STATE_FOLDER, mod1+'.pt')
    encoder1.load_state_dict(torch.load(PATH), strict=False)
    encoder1.eval()
    
    
    PATH=os.path.join(STATE_FOLDER, mod2+'.pt')
    encoder2.load_state_dict(torch.load(PATH), strict=False)
    encoder2.eval()
    #Freeze pre-trained parameters
    for name, param in encoder1.named_parameters():
        param.requires_grad = False
    for name, param in encoder2.named_parameters():
        param.requires_grad = False
        
    for epoch in epochs:
        for bs in batch_sizes:
            for beta in betas:
                for ds in dense_sizes:
                    for ls in latent_sizes:
                        for do in dropouts:
                            stdOrigin = sys.stdout
                            PATH = mod1+"+"+str(mod2)+"_"+str(epoch)+"_"+str(bs)+"_"+str(beta)+"_"+str(ds)+"_"+str(ls)+"_"+str(do)
                            sys.stdout = open(os.path.join(LOG_FOLDER, "VAE_"+PATH+".out"), "w")
                            CV(mod1, mod2, encoder1,encoder2,x1_train, x2_train, d_dims, 5, epoch, bs, beta, ds, ls, do)
        
                            p_time = time.time()
                            model = train_VAE(mod1, mod2, d_dims, encoder1, encoder2, x1_train, x2_train, epoch, bs, beta, ds, ls, do, True, PATH)
                            a_time = time.time()
                            print(f'Training time: {a_time-p_time}')
                            testloss = evaluate_VAE(model, x1_test, x2_test, bs)
                            print(f"Testloss: {testloss}")
                            gc.collect()
                            sys.stdout.close()
                            sys.stdout = stdOrigin
    
    
    
    
    
    
if (__name__ == '__main__'):
    print(sys.argv)
    parameters = parse_args(sys.argv)
    main(*parameters)
