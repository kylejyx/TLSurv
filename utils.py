import numpy as np
import pandas as pd
import pickle
import io
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper


def read_data(folder, mod1, mod2, suffix=""):
    if mod1 == 'mirna' or mod1 == 'mrna' or mod1 == 'cnv':
        mod1_train = pickle.load(io.open(folder + mod1 + suffix+"n.pickle", 'rb'))
    
    else:
        mod1_train = pickle.load(io.open(folder + mod1 + suffix+".pickle", 'rb'))
    x_train_one = mod1_train.astype('float32')
    print(x_train_one.shape)
    
    if mod2 != 'None':
        if mod2 == 'mirna' or mod2 == 'mrna' or mod2 == 'cnv':
            mod2_train = pickle.load(io.open(folder + mod2 + suffix+"n.pickle", 'rb'))
        
        else:
            mod2_train = pickle.load(io.open(folder + mod2 + suffix+".pickle", 'rb'))
        x_train_two = mod2_train.astype('float32')
        print(x_train_two.shape)
        assert all(mod1_train.index == mod2_train.index)
        return x_train_one, x_train_two, mod1_train.index
    return x_train_one, None, mod1_train.index

def to_list(ss):
    r = []
    ss = ss.split(",")
    for j in ss:
        r.append(int(j))
    return r


def parse_args(args):
    results = []
    if (' ' in args[1]):
        mod1, mod2 = args[1].split()
        offset=2
    else:
        mod1 = args[1]
        mod2 = args[2]
        offset=3
    results.append(mod1)
    results.append(mod2)
    for i in range(offset,len(args)):
        if ('[[' in args[i]): #List of list
            term = []
            ss = args[i][1:-1].split("],")
            ss[-1]=ss[-1][:-1] #Remove ']' at the end
            for j in ss:
                term.append(to_list(j[1:]))
            results.append(term)
 
        elif ('[' in args[i]):
            term = []
            ss = args[i][1:-1].split(",")
            int_flag = True
            try:
                int(ss[0])
            except ValueError:
                int_flag = False
            for j in ss:
                term.append(int(j) if int_flag else float(j))
            results.append(term)
        else:
            try:
                int(args[i])
                int_flag = True
            except ValueError:
                int_flag = False
            results.append([int(args[i]) if int_flag else float(args[i])])
    return results

