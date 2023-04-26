from sklearn.model_selection import ParameterGrid
import numpy as np
import torch
import torch.nn as nn
from data.data_loader import CustomDataset, collate_fn, CustomDatasetTarget, collate_fn_target
from torch.utils.data import DataLoader
from torch import optim
import os
import time
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import utils.my_functions as fu
from data.procesamiento import preprocessing

from utils.metrics import metric, MSE
from model.TransformerModel import ShaftFormer


class transformerModel(nn.Module):
    def __init__(self, args, data_args) -> None:
        super().__init__()

        self.args = args
        self.data_args = data_args

        #random.seed(231)
        self.split = False #just to split the data only one time --> save the values of the index
        self.idx_tr =[]
        self.idx_val = []
        self.idx_tst = []

        self.device = self._acquire_device()

        ##check initialization linear transformation:
        if args.linear_initialization not in ['Non','Xavier', 'He', 'Uniform']:
            raise Exception('Linear initialization method not valid')
        elif args.linear_initialization == 'Uniform' and (args.a >= args.b):
            raise Exception('Not valid values for the uniform initialization')
        
         ##check values of model_type:
        if args.model_type not in ['classification', 'forecasting']:
            raise Exception('Not valid model type')
        if args.model_type == 'classification' and not data_args.get_class:
            raise Exception('Get class should be TRUE')
    
        if args.model_type == 'classification':
            args.model_type = 'forecasting'
            self.model = ShaftFormer(args=args, device = self.device)
            try:
                self.model.load_state_dict(torch.load(f'./../results/{self.args.name_folder}/checkpoint_forecasting.pth'))
            except:
                raise Exception('The model for classification need a pretrained model of forecasting, check the name of the pretrained model is checkpoint_forecasting.pth')
            args.model_type = 'classification'
            model_c = ShaftFormer(args=args, device=self.device)
            
            self.model.simple_mlp = model_c.simple_mlp
        else:
            self.model = ShaftFormer(args=args, device=self.device)


    def predict(self, future:bool = True):
        """
        Function that predicts the future values given only src. 
        If the future is true, the model will predict values to future using "auto-regressive decoding"  
        """

        self.model.load_state_dict(torch.load(f'./../results/{self.args.name_folder}/checkpoint.pth', map_location= torch.device('cpu')))
        self.model.to(self.device)

        if self.args.model_type == "forecasting": 
            criterion =  self._select_criterion()


        self.model.eval()
        # self.device = torch.device("cpu")
        
        tst_loader = self._get_data( test = True ) #we get the test loader
        values = []
        
    
        for x in tst_loader:
            if self.data_args.get_class:
                x, class_t, feat, idx = x
                pred, trues = self.model.forward(x=x, feat=feat, test=future)        
            
            loss = criterion(pred.detach().cpu()[:,:,0], trues.detach().cpu())
            # for i in range(len(feat)): #only for the first 4 signals
            #     p = pred[:, i, 0]
            #     t = trues[:, i]
            #     # mae, mse, rmse, mape, mspe = metric(p.cpu().detach().numpy(), t.cpu().detach().numpy())
            #     values.append(MSE(p.cpu().detach().numpy(), t.cpu().detach().numpy()))
            values.append(loss)
        self.plot_signals(pred, trues, target=class_t, name=f'testResult')
            # print('\tMetrics for signal {} \nmse:{:.3f}, mae:{:.3f}, rmse:{:.3f}, mape:{:.3f}, mspe:{:.3f}'.format(i, mse, mae, rmse, mape, mspe))
        return np.mean(values)

    def test(self, x_loader:DataLoader, criterion, last_loss):
        self.model.eval()
        if self.args.use_multi_gpu and self.args.use_gpu: self.model.model.module.eval()
        
        total_loss = []
        trues_cm = []
        pred_cm =[]
        
        for x in x_loader:
            if self.data_args.get_class:
                x, class_t, feat, idx = x
            else:
                x, feat = x
            
            if self.args.model_type == "forecasting":
                pred, tgt = self.model.forward(x=x, feat=feat)
                loss = criterion(pred.detach().cpu()[:,:,0], tgt.detach().cpu())
            else:
                pred_soft = self.model.forward(x=x, feat=feat)
                loss = criterion(pred_soft, class_t.to(self.device))

                trues_cm.extend(class_t.cpu())
                pred = torch.argmax(pred_soft, dim = 1) #get the max index by row
                pred_cm.extend(pred.cpu())
            

            total_loss.append(loss.item())
        total_loss = np.average(total_loss)

        if total_loss < last_loss:
            if self.data_args.get_class and self.args.model_type == 'forecasting':
                x, class_t, feat, idx = next(iter(x_loader))
                pred, trues = self.model.forward(x=x, feat=feat)
                self.plot_signals(pred, trues, target=class_t, name='validationResults')
            elif self.args.model_type == 'classification':
                    cm = confusion_matrix(trues_cm , pred_cm)
                    disp = ConfusionMatrixDisplay(cm)
                    disp.plot()
                    plt.savefig(f'./../results/{self.args.name_folder}/cm_validation.png')
            else:
                x, feat = next(iter(x_loader))
                pred, trues = self.model.forward(x=x, feat=feat)
                self.plot_signals(pred, trues, name='validationResults')

        self.model.train()
        if self.args.use_multi_gpu and self.args.use_gpu: self.model.model.module.train()
        return total_loss

    def trainloop(self):
        #self._save_information() #save the information of the model (arguments) in a txt file
        

        tr_loader, val_loader = self._get_data()

        model_optim = self._select_optimizer()
        if self.args.model_type == "forecasting": 
            criterion =  self._select_criterion()

        else: #classification
            criterion =  self._select_criterion(1)
            trues_cm = []
            pred_cm = []

        last_loss = np.Inf
        last_train = 0
        loss_train = []
        loss_val = []

        for epoch in range(self.args.train_epochs):
            train_loss = []
            
            self.model.train()

            epoch_time = time.time()
            
            for tr in tr_loader:
                model_optim.zero_grad()

                if self.data_args.get_class:
                    tr, class_tr, feat, idx = tr
                else:
                    tr, feat = tr
                
                if self.args.model_type == "forecasting":
                    pred, trues = self.model.forward(x=tr, feat=feat)
                    loss = criterion(pred[:,:,0], trues)
                else: #classification
                    pred_soft = self.model.forward(x=tr, feat=feat, idx = idx)
                    loss = criterion(pred_soft, class_tr.to(self.device))

                    trues_cm.extend(class_tr.cpu())
                    pred = torch.argmax(pred_soft, dim = 1) #get the max index by row
                    pred_cm.extend(pred.cpu())

                train_loss.append(loss.item())
                
            
            print("Epoch: {} cost time: {} ".format(epoch+1, time.time()-epoch_time))
            
            loss.backward()
            model_optim.step()

            loss_train.append(np.average(train_loss))
            loss_val.append(self.test(x_loader=val_loader, criterion=criterion, last_loss= last_loss))
            print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} ".format(
                    epoch + 1, loss_train[epoch], loss_val[epoch]))

            if loss_val[epoch] < last_loss:
                best_model_path = f'./../results/{self.args.name_folder}/checkpoint.pth'
                if self.args.use_multi_gpu and self.args.use_gpu: 
                    torch.save(self.model.state_dict(), best_model_path)
                else: torch.save(self.model.state_dict(), best_model_path)
                print('Model updated')

                if self.data_args.get_class and self.args.model_type == 'forecasting':
                    # tr, class_tr, feat = next(iter(tr_loader))
                    # pred, trues = self.model.forward(x=tr, feat= feat)
                    self.plot_signals(pred, trues, target=class_tr)
                elif self.args.model_type == 'classification':
                    cm = confusion_matrix(trues_cm , pred_cm)
                    disp = ConfusionMatrixDisplay(cm)
                    disp.plot()
                    plt.savefig(f'./../results/{self.args.name_folder}/cm_train.png')
                else:
                    tr, class_tr = next(iter(tr_loader))
                    pred, trues = self.model.forward(x=tr)
                    self.plot_signals(pred, trues)

                patience = 0

                last_loss = loss_val[epoch]
                last_train = loss_train[epoch]
            else:
                patience += 1
                if patience >= 5 and loss_val[epoch] > loss_val[epoch-1]: #just break if the patience is big and the loss is not decreasing
                    # model_optim.param_groups[0]['lr'] /= 10.0
                    # print(f"\n\tUPDATING THE LEARNING RATE: {model_optim.param_groups[0]['lr']}")
                    # patience = 0
                    print('BREAK ---------- ')
                    break
        
        #SAVE THE INFORMATION ABOUT THE LOSS
        np.save(f'./../results/{self.args.name_folder}/loss_train.npy', loss_train)
        np.save(f'./../results/{self.args.name_folder}/loss_val.npy', loss_val)

        # return self.model
        return last_loss, last_train

  


    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    #### PLOT FUNCTION
    def plot_signals(self, pred, trues, name:str = 'trainResults', target=None, row=2, col=2):
        fig, axes = plt.subplots(row, col ,figsize=(12,5)) #just save the example of the last batch trained

        for i, ax in enumerate(axes.flat):
            ax.plot(trues.cpu().detach().numpy()[:200,i], label='GroundTruth')
            ax.plot(pred.cpu().detach().numpy()[:200,i, 0], label='Prediction')
            if target != None:
                ax.set_title(target[i].item())
        fig.suptitle(name, fontsize=16)
        plt.savefig(f'./../results/{self.args.name_folder}/{name}.png')

    ## JUST IN CASE WE DECIDE TO TRY SEVERAL OF THEM
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self, forecasting = 0):
        if forecasting == 0: criterion =  nn.MSELoss()
        elif forecasting == 1: criterion = nn.CrossEntropyLoss()
        return criterion

    def _save_information(self):
        if not os.path.exists(f'./../results/{self.args.name_folder}'):
            os.makedirs(f'./../results/{self.args.name_folder}')

        f = open(f'./../results/{self.args.name_folder}/arguments.txt', "w")
        for a in self.args:
            f.write(f'{a}: {self.args[a]} \n')
        f.close()

    ## OTHER FUNCTIONS FOR THE DATA
    def _split_data(self, n_signals, train_split = 0.7, test_split = 0.1):
        if not self.split :
            #Knowing the number of signals we hace, we split between train / validation / test
            n_train = int(n_signals* train_split) # 0 to n_train
            n_test = int(n_signals * test_split) #n_train to n_train+n_test
            n_val = n_signals - (n_train + n_test) #the last n_val

            range_list = list(range(n_signals))
            random.shuffle(range_list) #shuffle the list (this is just for when we use several configurations, so we take different configurations for train / test / validation)

        
            self.idx_tr = range_list[:n_train]
            self.idx_val = range_list[-n_val:]
            self.idx_tst = range_list[n_train:n_train+n_test]

            #save the indexs of the test
            np.save(f'./../results/{self.args.name_folder}/test_index.npy', self.idx_tst)
            # np.save(f'./../results/{self.args.name_folder}/val_index.npy', self.idx_val)
            # np.save(f'./../results/{self.args.name_folder}/train_index.npy', self.idx_tr)

            self.split = True
        return

    def _get_data(self, test = False ):
        """
        test: if is False it will return the train dataloader and the validation dataloader
        """

        #get the data from confirguration
        configurations = {'carga': [0.0, 1.0], 'velocidad': [0.0, 1.0], 
            'lado': [0, 1], 'direct': [0, 1], 'corte': [0, 1]}
        grid_conf = list(ParameterGrid(configurations))

        conf = grid_conf[0]
        data, target, features, xref_ws1 = fu.preprocess_pickle_configuration(self.data_args.path, lado=conf['lado'], direction=conf['direct'], corte=conf['corte'], carga=conf['carga'], velocidad=conf['velocidad'])
        target = target.values
        features = features.values
        if self.data_args.several_conf: #we add more configurations
            for conf in grid_conf[1:]: 
                X_ws1, Y_ws1, feat_ws1, xref_ws1 = fu.preprocess_pickle_configuration(self.data_args.path, lado=conf['lado'], direction=conf['direct'], corte=conf['corte'], carga=conf['carga'], velocidad=conf['velocidad'])
                data = np.concatenate((data, X_ws1), axis = 0)
                features = np.concatenate((features, feat_ws1.values), axis = 0)
                target = np.concatenate((target, Y_ws1.values), axis=0)

        n_signals, len_seq = data.shape #number of signals and the length of the sequence

        if not self.split and not test: 
            self._split_data(n_signals=n_signals)
            print(f'Train: {len(self.idx_tr)} \nValidation: {len(self.idx_val)} \nTest: {len(self.idx_tst)} \n')

        #do the preprocessing (min-max scaler)
        data_new = preprocessing(n_signals, data, feature_range = self.args.feature_range)

        #traspose the signal to have --> [sequence len, number of signals]. Map the labels and transform features to tensor
        dataT = torch.tensor(np.array(data_new).T, dtype=torch.float32)
        targetT = torch.tensor(list(map(lambda x: {'eje sano':0, 'd1':1, 'd2':2, 'd3':3}[x], target)))
        featuresT = torch.tensor(features, dtype=torch.float32)
        
        if not test:
            tr = dataT[:, self.idx_tr]
            val = dataT[:, self.idx_val]
            feat_tr = featuresT[self.idx_tr, :]
            feat_val = featuresT[self.idx_val, :]

            if self.data_args.get_class: #we add the target so we can use it 
                tr_class = targetT[self.idx_tr]
                val_class = targetT[self.idx_val]
                dataset = CustomDatasetTarget(tr, tr_class, feat_tr, self.idx_tr)
                dataset2 = CustomDatasetTarget(val, val_class, feat_val, self.idx_val)
                tr_loader = DataLoader(dataset, batch_size = self.args.batch_size, collate_fn=collate_fn_target)
                val_loader = DataLoader(dataset2, batch_size = self.args.batch_size, collate_fn=collate_fn_target)
            else:
                dataset = CustomDataset(tr)
                dataset2 = CustomDataset(val)
                tr_loader = DataLoader(dataset, batch_size = self.args.batch_size, collate_fn=collate_fn)
                val_loader = DataLoader(dataset2, batch_size = self.args.batch_size, collate_fn=collate_fn)

            return tr_loader, val_loader

        else:
            self.idx_tst = np.load(f'./../results/{self.args.name_folder}/test_index.npy')
            # self.idx_tst = np.load(f'./../results/{self.args.name_folder}/train_index.npy')
            tst = dataT[:, self.idx_tst]
            feat_tst = featuresT[self.idx_tst,:]
            if self.data_args.get_class:
                tst_class = targetT[self.idx_tst]
                dataset = CustomDatasetTarget(tst, tst_class, feat_tst, self.idx_tst)
                ts_loader = DataLoader(dataset, batch_size = self.args.batch_size, collate_fn=collate_fn_target, shuffle=False)
            else:
                dataset = CustomDataset(tst)
                ts_loader = DataLoader(dataset, batch_size = self.args.batch_size, collate_fn=collate_fn, shuffle=False)

            return ts_loader
