#%% LIBRARIES 

import torch

from utils.myTools import dotdict
from model.model import transformerModel
import os
import ast

#%% ARGUMENTS
args = dotdict()

    #model
args.heads = 3 #number of heads for the transformer
args.nencoder=3 #number of layers in the encocer
args.dropout = 0.1 #dropout
args.train_epochs = 100 #number of epochs to train the model (a maximum number of them)
args.output_attention = False #if we want to print the attention scores ---- TODAVIA NO ESTÁ HECHO PARA QUE SE PUEDAN IMPRIMIR

args.learning_rate = 0.0001 
args.batch_size = 16 #16

args.linear_initialization = 'Non' #We can use ['Non', 'Xavier', 'He', 'Uniform'] --> If uniform, we need to specify the values of a and b
if args.linear_initialization == 'Uniform':
    args.a = -0.1
    args.b = 0.1

args.model_type = "forecasting" #['classification', 'forecasting']
if args.model_type == 'classification':
    args.num_class = 4

args.use_gpu = True if torch.cuda.is_available() else False
args.gpu = 1
args.use_multi_gpu = False
args.devices = '0,1,3'

    #embedding signal 
args.inchannels = 1 #the number of features per time point 
args.outchannels = 96 #the number of features outputted per time point (taking into account the transformation done in the conf --> 64 conv1 + 32 conv2 = 96 total output dimension)
args.kernel = 9 # k is the width of the 1-D sliding kernel
    #embedding configuration
args.inchannels_conf = 5 #we have 5 different features in the conf
args.outchannels_conf = 2**args.inchannels_conf #it will be 32. We do this to try making the model learn one feature for each combination of the configurations

args.feature_range = (-3, 3) #if the values are small, the model does not learn --> prediction is equal to a line
args.conf_cnn = True #if false, we do not consider the configurations
args.two_linear = False #if true we use 2 linear (96 to 32 and then 32 to 1). only if conf_cnn is true

i = 2
conf_num= 2 if args.conf_cnn else 1
num_lin = 2 if args.two_linear else 1
# args.name_folder = f'shaftformer_{conf_num}cnn_{num_lin}linear_exp{i}'
args.name_folder = f'shaftformer_{conf_num}cnn_{num_lin}linear_{args.model_type}_exp{i}'


train = True
if train:
    seguir = False
    while not seguir:
        if os.path.exists(f'./../results/{args.name_folder}'): 
            print(f'folder : {args.name_folder} already exists')
            res = str(input("\tDo you want to rewrite the information? (y / n)  "))
            if res == 'y': seguir = True
            else:
                i += 1 #add a new experiment
                # args.name_folder = f'shaftformer_{conf_num}cnn_{num_lin}linear_exp{i}'
                args.name_folder =f'shaftformer_{conf_num}cnn_{num_lin}linear_{args.model_type}_exp{i}'
        else: seguir = True

else: #we are testing the model saved
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

#%% MODEL
# import os
# os.chdir('export/gts_usuarios/abcerrada/Trenes/ShaftFormer')
model = transformerModel(args, data_args)
if train: 
    model.trainloop()
else: 
    print('Predicting---- using the test')
    model.predict(future = True)
    


# %%