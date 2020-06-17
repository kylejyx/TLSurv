"""
Code for training fusion network for TTSurv(MAE)
Example: python Surv_train_MAE.py mirna mrna epochs batch_size dense_size learning_rate l2_regularization dropout
Please ensure there is no space in the list if a list of candidates is provided for any hyper-parameter
"""
import sys
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
np.random.seed(1234)
_ = torch.manual_seed(123)

from MAE import MAE
from MAE import Loss_view_similarity
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
SAVE_FOLDER = os.path.expanduser('~/states_MAE_LUAD_set2/')
LOG_FOLDER = os.path.expanduser('./log_MAE_set2/')

def train_MAE(mod1, mod2, in_dims, d_dims, encoder1, encoder2, x, hidden_dims, learning_rate, weight_decay, batch_size, num_epochs, loss_fns, loss_weights, save, path):
    torch.cuda.empty_cache()
    if mod2 == 'None':
        pass
        model = MAE_single(in_features_one, dense_size, latent_size, dropout, device).to(device)
    else:
        
        model = MAE(in_dims, d_dims, hidden_dims, encoder1, encoder2).to(device)
        
    print('Trainable parameters:')
    for name, param in model.named_parameters():
        if (param.requires_grad):
            print(name)
    
    model.train()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)

    #loss_per_epoch = []
    for j in range(num_epochs):
        total_loss = 0.
        for i in range(0, len(x), batch_size):
            y_pred = model(x[i:i+batch_size])

            loss_batch = []
            loss_batch.append(loss_fns[0](y_pred[0], y_pred[2]) * loss_weights[0]) #MSELoss
            loss_batch.append(loss_fns[1](y_pred[1]) * loss_weights[1]) #View similarity

            loss = sum(loss_batch)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch: {j+1} loss: {total_loss}")
        #loss_per_epoch.append((j+1,total_loss))
    if save:
        PATH = SAVE_FOLDER + path
        hyperparameters = {'D dims': d_dims,
            'Hidden dims': hidden_dims}
        with open(PATH + ".dict", "wb") as f:
            pickle.dump(hyperparameters, f)
        torch.save(model.state_dict(), PATH + '.pt')
    return model

def evaluate_MAE(model, x, batch_size, loss_fns, loss_weights):
    loss_weights=[1,0]
    with torch.no_grad():
        model.eval()
        total_loss = 0.
        for i in range(0, len(x), batch_size):
            y_pred = model(x[i:i+batch_size])

            loss_batch = []
            loss_batch.append(loss_fns[0](y_pred[0], y_pred[2]) * loss_weights[0]) #MSELoss
            loss_batch.append(loss_fns[1](y_pred[1]) * loss_weights[1]) #View similarity

            loss = sum(loss_batch)
            total_loss += loss.item()
    return total_loss
    
def CV(mod1, mod2, e1, e2, x, folds, in_dims, d_dims, hidden_dims, lr, wd, bs, epoch, loss_fns, loss_weights):
    kf = KFold(n_splits=folds, shuffle=True, random_state = 66)
    for train_index, test_index in kf.split(x):
        x_train = x[train_index]
        x_test = x[test_index]
        
        p_time = time.time()
        model = train_MAE(mod1, mod2, in_dims, d_dims, e1, e2, x_train, hidden_dims, lr, wd, bs, epoch, loss_fns, loss_weights, False, None)
        a_time = time.time()
        print(f'Training time: {a_time-p_time}')
        testloss = evaluate_MAE(model, x_test, bs, loss_fns, loss_weights)
        print(f"Validationloss: {testloss}")
                    
def main(mod1, mod2, epochs, batch_sizes, hidden_dims, learning_rates, weight_decay, dropouts):
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
        x_train = torch.cat((x1_train, x2_train), dim=1)
        x_test = torch.cat((x1_test, x2_test), dim=1)
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
    hidden_dims=hidden_dims[0]
    loss_fn_reg = torch.nn.MSELoss()
    view_sim_loss_type = 'hub'
    cal_target='mean-feature'
    loss_view_sim = Loss_view_similarity(sections=hidden_dims[-1], loss_type=view_sim_loss_type, explicit_target=True, cal_target=cal_target, target=None)
    loss_fns = [loss_fn_reg, loss_view_sim]
    loss_weights = [1, 1]
    
    for epoch in epochs:
        for bs in batch_sizes:
            for lr in learning_rates:
                for wd in weight_decay:
                    for do in dropouts:
                        stdOrigin = sys.stdout
                        PATH = mod1+"+"+str(mod2)+"_"+str(epoch)+"_"+str(bs)+"_"+str(hidden_dims)+"_"+str(lr)+"_"+str(wd)+"_"+str(loss_weights)+"_"+str(do)
                        sys.stdout = open(os.path.join(LOG_FOLDER, "MAE_"+PATH+".out"), "w")
                        CV(mod1, mod2, encoder1, encoder2, x_train, 5, in_dims, d_dims, hidden_dims, lr, wd, bs, epoch, loss_fns, loss_weights)
    
                        p_time = time.time()
                        model = train_MAE(mod1, mod2, in_dims, d_dims, encoder1, encoder2, x_train, hidden_dims, lr, wd, bs, epoch, loss_fns, loss_weights, True, PATH)
                        a_time = time.time()
                        print(f'Training time: {a_time-p_time}')
                        testloss = evaluate_MAE(model, x_test, bs, loss_fns, loss_weights)
                        print(f"Testloss: {testloss}")
                        gc.collect()
                        sys.stdout.close()
                        sys.stdout = stdOrigin

    
if (__name__ == '__main__'):
    print(sys.argv)
    parameters = parse_args(sys.argv)
    main(*parameters)
