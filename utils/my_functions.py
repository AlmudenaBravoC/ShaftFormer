# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_curve, roc_auc_score, auc,confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,  OneHotEncoder, label_binarize
from sklearn.model_selection import ParameterGrid
from scipy import signal
from skimage.transform import resize
import argparse
from itertools import cycle
from torchmetrics import AUROC, ConfusionMatrix

path = './model/'

# DEFINE FUNCTIONS USED
def subsam(sig, length):
    subsampled = signal.resample(x=sig[:length], num=2000)
    return subsampled

def normalize(X):
    scaler = MinMaxScaler()
    scaler.fit(X)
    norm_data = scaler.transform(X)
    return norm_data

def save_roc_curve(fper, tper, auc, path , title='', name=''):
    fig, ax = plt.subplots( nrows=1, ncols=1 )  
    ax.plot(fper, tper, color='red', label='ROC curve and AUC = %0.3f' %auc)
    ax.plot([0, 1], [0, 1], color='green', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic Curve'+ title)
    ax.legend()
    fig.savefig(path + name + '.png')   
    plt.close(fig)
      
def loss_curve(loss_tr, loss_val, name, path):
    fig, ax = plt.subplots( nrows=1, ncols=1 )  
    ax.plot(loss_tr, label='Training set')
    ax.plot(loss_valid, label='Validation set')
    ax.set_title('Training and Validation Loss')
    fig.savefig(path +name+ '.png')   
    plt.close(fig)
    
def save_roc_multiclass(y_test, y_score, name, n_classes = 4, path = path):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
        # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    fig, ax = plt.subplots( nrows=1, ncols=1 ) 
    ax.plot(fpr["micro"],tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )
    
    ax.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )
    
    colors = cycle(["aqua", "darkorange", "cornflowerblue","green"])
    for i, color in zip(range(n_classes), colors):
        ax.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )
    
    ax.plot([0, 1], [0, 1], "k--", lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Some extension of Receiver operating characteristic to multiclass")
    ax.legend(loc="lower right")
    fig.savefig(path + name + '.png')   
    plt.close(fig)

def get_auc(model, x, xref, feat, Y, path_plot, title='', name=''):
    probs = torch.exp(model.forward(torch.Tensor(x).to(model.device),torch.Tensor(xref).to(model.device),torch.Tensor(feat.values)).to(model.device))
    probs = probs.detach().numpy()
    y = Y
    auc = roc_auc_score(y, probs[:,1])
    fper, tper, thresholds = roc_curve(y, probs[:,1])
    save_roc_curve(fper, tper, auc, path_plot, title, name)
    y_pred = np.argmax(probs, axis=1)
    cm = confusion_matrix(y, y_pred)
    return auc,cm  

def compute_spec(x):
    #f, t, Sxx = signal.spectrogram(x, fs=12800)
    f, t, Sxx_im = signal.spectrogram(x, fs=12800, mode='complex')
    f, t, Sxx_mag = signal.spectrogram(x, fs=12800, mode='magnitude')
    #S = np.stack((Sxx,Sxx_im, Sxx_mag), axis=-1)
    S = np.stack((np.real(Sxx_im),np.imag(Sxx_im), Sxx_mag), axis=-1)
    return S

def spectrogram(xsig, xr):
    d1 = pd.DataFrame()
    d1['x_ws1'] = list(xsig)
    X_spec_WS1 = pd.DataFrame()
    X_spec_WS1['Sx'] = d1['x_ws1'].map(lambda x: compute_spec(x))
    X_spec_WS1['Sx'] = X_spec_WS1['Sx'].map(lambda x: resize(x, (64, 64), mode = 'constant'))
    x = np.stack(X_spec_WS1['Sx'])#.astype(None)
    
    d2 = pd.DataFrame()
    d2['x_ref_ws1'] = list(xr)
    X_spec_ref_WS1 = pd.DataFrame()
    X_spec_ref_WS1['Sx'] = d2['x_ref_ws1'].map(lambda x: compute_spec(x))
    X_spec_ref_WS1['Sx'] = X_spec_WS1['Sx'].map(lambda x: resize(x, (64, 64), mode = 'constant'))
    xref = np.stack(X_spec_ref_WS1['Sx'])#.astype(None)
    
    return x, xref
def plot_roc_curve(fper, tper, auc, title=''):
    plt.figure()
    plt.plot(fper, tper, color='red', label='ROC curve and AUC = %0.3f' %auc)
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve'+ title)
    plt.legend()
    plt.show()
    
def plot_spec(x):
    f, t, Sxx = signal.spectrogram(x, fs=12800)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram representation')
    plt.show()
    
def preprocess_pickle(path, features = False):
    data = pd.read_pickle(path)
    # We also extract the values of the reference we are going to include in the model
    xref = data[data['Label']=='eje sano']
    xref = xref.sample(frac=0.304)
    xref = xref['Subsampled']
      
    # remove this values of xref from the data
    data = data.drop(xref.index)
    
    # not use time values
    #xref = list(map(lambda x: np.array([r[1] for r in x]), xref.values))
    xref = np.stack(xref).astype(None)
    
    X = np.stack(data['Subsampled'].values).astype(None)
    feat = data[['Lado','Direction','Corte','Load','Velocidad']]
    Y = data['Label']
    
    
    if features:
        return X, Y, feat, xref
    else:
        return X, Y
    
def preprocess_pickle_configuration(path, lado, direction, corte, carga, velocidad):
    data = pd.read_pickle(path)
    data['Load'] = data['Load'].round()


    # NOW FILTER FOR SPECIFIC CONFIGURATION
    data = data[(data['Lado']==lado) & (data['Direction']==direction) & (data['Load']==carga) 
                & (data['Corte']==corte) & (data['Velocidad']==velocidad)]
    
    
    # We also extract the values of the reference we are going to include in the model
    xref = data[data['Label']=='eje sano']
    xref = xref.sample(frac=0.304)
    xref = xref['Subsampled']
      
    # remove this values of xref from the data
    data = data.drop(xref.index)
    
    # not use time values
    #xref = list(map(lambda x: np.array([r[1] for r in x]), xref.values))
    xref = np.stack(xref).astype(None)
    
    X = np.stack(data['Subsampled'].values).astype(None)
    feat = data[['Lado','Direction','Corte','Load','Velocidad']]
    Y = data['Label']
    
    return X, Y, feat, xref
    