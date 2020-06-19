import os
import pandas as pd
import numpy as np
import sys

network = sys.argv[1]
print(f"Results for {network}")
if ('Surv' in network or 'nnet' in network):
    names=[]
    values_cv = []
    values_ct = []
    values_i = []
    values_mean = []
    values_std = []
    values_t = []
    results_dir = './'
    for fname in os.listdir(results_dir):
        if fname[-4:] == '.out' and fname.startswith(network):
            with open(os.path.join(results_dir, fname),'r') as f:
                val=[]    
                t=0
                for line in f:
                    if ('Val C-index:' in line):
                        val.append(float(line[13:line.find(',')]))
                    if ('Training time:' in line):
                        t = float(line[15:-1])
                    if ('Test C-index: ' in line):
                        values_cv.append(val)
                        values_mean.append(np.mean(val))
                        values_std.append(np.std(val))
                        values_t.append(t)
                        values_ct.append(float(line[14:line.find(',')]))
                        names.append(fname[len(network)+1:-4])
                        
                    """
                    if ('Integrated' in line):
                        values_i.append(line[24:-1])
                        
                    """
    #values_i=[float(i) for i in values_i]
    print("Overall best C-index: ", max(values_mean))
    print(names[np.argmax(values_mean)])
    #print("Overall best Brier score: ", min(values_i))
    #print(names[np.argmin(values_i)])

    tables={'Name' :names, 'C-index(val)':values_cv, 'Val(mean)':values_mean, 'Val(std)':values_std, 'Training time':values_t, 'C-index(test)':values_ct}#, 'Integrated Brier Score':values_i}
    a = pd.DataFrame(tables)
    a.to_csv(network + '.csv')

    print("Optimal C-index for each modality")
    combined = []
    for i in range(len(values_ct)):
        if names[i].find('_') > 0:
            modality = names[i][:names[i].find('_')] #modality = names[i][:names[i].find('_')] if ('+' in names[i] or ' ' in names[i]) else names[i][:names[i].find('_', names[i].find('_')+1)]
            combined.append((modality, values_ct[i], values_mean[i], values_std[i], values_cv[i], values_t[i], names[i]))


    combined_c = sorted(combined, key=lambda x: x[2], reverse=True)
    #combined_i = sorted(combined, key=lambda x: x[3])

    mods = set()

    for i in range(len(combined_c)):
        if not combined_c[i][0] in mods:
            print(combined_c[i])
            mods.add(combined_c[i][0])
    """
    print("Optimal Integrated Brier Scorefor each modality")

    mods = set()

    for i in range(len(combined_i)):
        if not combined_i[i][0] in mods:
            print(combined_i[i])
            mods.add(combined_i[i][0])
    """
else:
    names=[]
    values_c = []
    values_v = []
    values_mean = []
    values_std = []
    values_t=[]
    results_dir = './'
    for fname in os.listdir(results_dir):
        if fname[-4:] == '.out' and fname.startswith(network):
            with open(os.path.join(results_dir, fname),'r') as f:
                val=[]
                t=0
                for line in f:
                    if ('Validationloss:' in line):
                        val.append(float(line[16:-1]))
                    if ('Training time:' in line):
                        t = float(line[15:-1])
                    if ('Testloss:' in line):
                        values_c.append(line[10:-1])
                        values_v.append(val)
                        values_mean.append(np.mean(val))
                        values_std.append(np.std(val))
                        values_t.append(t)
                        names.append(fname[len(network)+1:-4])
    values_c=[float(i) for i in values_c]

    print("Overall best loss: ", min(values_mean))
    print(names[np.argmin(values_mean)])

    tables={'Name' :names, 'Testloss':values_c, 'Validation loss':values_v, 'Val(mean)':values_mean, 'Val(std)':values_std, 'Training time':values_t}
    a = pd.DataFrame(tables)
    a.to_csv(network + '.csv')

    print("Optimal testloss for each modality")
    combined = []
    for i in range(len(values_c)):
        if names[i].find('_') > 0:
            modality = names[i][:names[i].find('_')]
            combined.append((modality, values_c[i], values_mean[i], values_std[i], values_t[i], names[i]))

    combined_c = sorted(combined, key=lambda x: x[1])

    mods = set()

    for i in range(len(combined_c)):
        if not combined_c[i][0] in mods:
            print(combined_c[i])
            mods.add(combined_c[i][0])
