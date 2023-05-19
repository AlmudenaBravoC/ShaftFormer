import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from utils.myTools import dotdict
import ast
from model.model import transformerModel

i = ''
conf_num= 2
num_lin = 1
model_type = 'forecasting' 
# args.name_folder = f'shaftformer_{conf_num}cnn_{num_lin}linear_exp{i}'
name_folder = f'./../results/shaftformer_{conf_num}cnn_{num_lin}linear_{model_type}_SAVE{i}'

    
val = np.load(f'{name_folder}/loss_val.npy')
tr = np.load(f'{name_folder}/loss_train.npy')

# plt.plot(np.arange(len(val)), val, c='r', label='validation')
# plt.plot(np.arange(len(val)), tr, label='train')
# plt.legend()
# plt.show()

print('Train:',min(tr))
print('Val:',min(val))


#  The use of a filter to reduce the noise of the signals and visualize better the results
# we need the values of the signal (original and prediction)

#get the arguments
args = dotdict()
args.name_folder = name_folder
file = open(f'./../results/{args.name_folder}/arguments.txt', "r")
for line in file: #open the file that has the arguments saved and create the dictionary
    s1 = line.split(': ')
    if s1[0] != 'devices':
        try:
            args[s1[0]] = ast.literal_eval(s1[1].split(' \n')[0]) #we use this library to mantein the type of the values (identify if it is a int, float, bool or tuple)
        except ValueError:
            args[s1[0]] = s1[1].split(' \n')[0]
    else:
        args[s1[0]] = s1[1].split(' \n')[0]


#%% DATA
data_args = dotdict()

data_args.path = './../DATASETS/WS1_preprocessed_multiclass.pkl'
# data_args.path = 'Trenes/DATASETS/WS1_preprocessed_multiclass.pkl'
data_args.get_class = True #if we want to also get the class of the data
data_args.several_conf = True

model = transformerModel(args, data_args)

print('Predicting---- using the test')
preds, trues = model.predict(future = True, get_values_signal=True)

print(preds.shape) #length , batch, dimension (if it has)
print(trues.shape)