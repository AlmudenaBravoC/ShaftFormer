from sklearn.model_selection import ParameterGrid
from utils import my_functions as fu
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

#get the data from confirguration
configurations = {'carga': [0.0, 1.0], 'velocidad': [0.0, 1.0], 
    'lado': [0, 1], 'direct': [0, 1], 'corte': [0, 1]}
grid_conf = list(ParameterGrid(configurations))


###### ALL WS
configurations = {'carga': [0.0, 1.0], 'velocidad': [0.0, 1.0], 
    'lado': [0, 1], 'direct': [0, 1], 'corte': [0, 1]}
grid_conf = list(ParameterGrid(configurations))

ws = []
target = []
for i in range(1,5):
    path = f'./../DATASETS/WS{i}_preprocessed_multiclass.pkl'
    len_c = 0
    for conf in grid_conf:
        X_ws1, Y_ws1, feat_ws1, xref_ws1 = fu.preprocess_pickle_configuration(path, lado=conf['lado'], direction=conf['direct'], corte=conf['corte'], carga=conf['carga'], velocidad=conf['velocidad'])
        target.extend(Y_ws1.values)
        len_c += len(Y_ws1)
    
    ws.extend(np.repeat(i, len_c))

## the labels
sns.countplot(x=target, hue=ws)
plt.show()




########## ONLY WS1

path = './../DATASETS/WS1_preprocessed_multiclass.pkl'

conf = grid_conf[0]
data, target, features, xref_ws1 = fu.preprocess_pickle_configuration(path, lado=conf['lado'], direction=conf['direct'], corte=conf['corte'], carga=conf['carga'], velocidad=conf['velocidad'])
target = target.values
confs_num = [len(data)]
for conf in grid_conf[1:]:
    X_ws1, Y_ws1, feat_ws1, xref_ws1 = fu.preprocess_pickle_configuration(path, lado=conf['lado'], direction=conf['direct'], corte=conf['corte'], carga=conf['carga'], velocidad=conf['velocidad'])
    data = np.concatenate((data, X_ws1), axis = 0)
    target = np.concatenate((target, Y_ws1.values), axis=0)
    confs_num.append(len(X_ws1)+confs_num[-1])

dict_target = dict(Counter(target))


## the labels
plt.bar(list(dict_target.keys()), dict_target.values())
plt.show()


## the range
range_vals = np.max(data, axis=1) - np.min(data, axis=1)

# plot the histogram of range values
plt.hist(range_vals, bins=80)
plt.xlabel('Range')
plt.ylabel('Frequency')
plt.title('Distribution of Range Values')
plt.show()


    #another
min_values = np.min(data, axis=1)
max_values = np.max(data, axis=1)
top = 2000

# Create a plot
fig, ax = plt.subplots()
ax.plot(max_values[:top], label='max')
ax.plot(min_values[:top], label='min')

for i in confs_num:
    if i>top: break
    ax.axvline(x=i, color='red', linestyle='--')

# Add labels and legend
ax.set_xlabel('Row index')
ax.set_ylabel('Value')
ax.legend()
plt.show()


##configurations
fig, axes = plt.subplots(1,3,figsize=(16,9))
ini_conf = 0 #min can be 1
top = 2500 #(2000:6confs, 3000:9confs)
for ax in axes.flat:
    values=[]
    for i,t in enumerate(confs_num):
        if i < ini_conf: continue
        else:
            if len(values)>top:next=i; break
            if i==0: values.extend(np.repeat(i, t))
            else: values.extend(np.repeat(i, t-confs_num[i-1]))
    #tar_count = Counter(target[:2000])
    if ini_conf == 0:
        sns.countplot(y=target[:len(values)], hue=values, ax=ax)
    else:
        sns.countplot(y=target[confs_num[ini_conf-1]:confs_num[ini_conf-1]+len(values)], hue=values, ax=ax)
    
    ini_conf = next
    
fig.suptitle('Distribution of the labels in the different configurations', size=20)
plt.show()