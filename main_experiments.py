#%% LIBRARIES 

import torch

from utils.myTools import dotdict
from model.model import transformerModel
import os
import ast

import numpy as np
import random


#%% ARGUMENTS
args = dotdict()
last_loss = np.Inf

for exp in range(30):
    print(f'\tEXPERIMENT NUMBER __________________________ {exp}')

        #model
    args.heads = random.choice([3,6]) #number of heads for the transformer
    args.nencoder = random.choice([3,4]) #number of layers in the encocer
    args.dropout = round(random.uniform(0.1, 0.5), 2) #dropout

    args.train_epochs = 200 #number of epochs to train the model (a maximum number of them)
    args.output_attention = False #if we want to print the attention scores ---- TODAVIA NO ESTÁ HECHO PARA QUE SE PUEDAN IMPRIMIR

    args.learning_rate = round(random.uniform(0.00001, 0.01), 5)
    args.batch_size = random.choice([16,20,24]) #16
    args.sigma = round(random.uniform(0.1, 0.5), 2)

    if args.batch_size == 24 : args.heads = 3 

    args.linear_initialization = 'Non' #We can use ['Non', 'Xavier', 'He', 'Uniform'] --> If uniform, we need to specify the values of a and b
    if args.linear_initialization == 'Uniform':
        args.a = -0.1
        args.b = 0.1

    args.model_type = "forecasting" #['classification', 'forecasting']
    if args.model_type == 'classification':
        args.num_class = 4

    args.use_gpu = True if torch.cuda.is_available() else False
    args.gpu = 0
    args.use_multi_gpu = False

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

    i = 1
    conf_num= 2 if args.conf_cnn else 1
    num_lin = 2 if args.two_linear else 1
    # args.name_folder = f'shaftformer_{conf_num}cnn_{num_lin}linear_exp{i}'
    args.name_folder = f'shaftformer_{conf_num}cnn_{num_lin}linear_{args.model_type}_exp{i}'         

    #%% DATA
    data_args = dotdict()

    data_args.path = './../DATASETS/WS1_preprocessed_multiclass.pkl'
    # data_args.path = 'Trenes/DATASETS/WS1_preprocessed_multiclass.pkl'
    data_args.get_class = True #if we want to also get the class of the data
    data_args.several_conf = True

    print(f'\n{args.heads},{args.nencoder},{args.dropout},{args.learning_rate},{args.batch_size},{args.sigma}')

    #%% MODEL
    model = transformerModel(args, data_args)
    val, train= model.trainloop()
    

    print('Predicting---- using the test')
    mse = model.predict(future = True)


    ### SAVE ALL THE INFORMATION
    if not os.path.exists(f'./../results/{args.name_folder}'):
        os.makedirs(f'./../results/{args.name_folder}')
        
        f= open(f'./../results/{args.name_folder}/results2.txt', "a")
        f.write('heads,n-enc,dropout,learningRate,batch,sigma,trainLoss,validationLoss,tetsLoss')
    else:
        f = open(f'./../results/{args.name_folder}/results2.txt', "a")
    f.write(f'\n{args.heads},{args.nencoder},{args.dropout},{args.learning_rate},{args.batch_size},{args.sigma},{train},{val},{mse}')
    f.close()

    if last_loss >mse:
        last_loss = mse
        old_file = f'./../results/{args.name_folder}/checkpoint.pth'
        new_file = f'./../results/{args.name_folder}/checkpoint_best.pth'
        os.rena,e(old_file, new_file)

    torch.cuda.empty_cache()

