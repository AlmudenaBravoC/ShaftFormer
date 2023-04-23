import numpy as np
import matplotlib.pyplot as plt


i = 2
conf_num= 2
num_lin = 1
model_type = 'forecasting' 
# args.name_folder = f'shaftformer_{conf_num}cnn_{num_lin}linear_exp{i}'
name_folder = f'./../results/shaftformer_{conf_num}cnn_{num_lin}linear_{model_type}_exp{i}'

    
val = np.load(f'{name_folder}/loss_val.npy')
tr = np.load(f'{name_folder}/loss_train.npy')

# plt.plot(np.arange(len(val)), val, c='r', label='validation')
# plt.plot(np.arange(len(val)), tr, label='train')
# plt.legend()
# plt.show()

print('Train:',min(tr))
print('Val:',min(val))