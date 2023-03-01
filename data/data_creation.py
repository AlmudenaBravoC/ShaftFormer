import pandas as pd
import numpy as np

import utils.my_functions as fu


def data_creation(data_args:dict):
    """
    saves the new data in the data folder in a csv file
    """
    X_ws1, Y_ws1, feat_ws1, xref_ws1 = fu.preprocess_pickle_configuration(data_args.path, lado=0, direction=0, corte=0, carga=0, velocidad=0)

    print(X_ws1.shape) #(282, 2000)
    print(Y_ws1.shape) #(282,)

    if data_args.data_type == 0: n_signals, n_sequence = X_ws1.shape
    else: n_signals, n_sequence = X_ws1.shape #AHORA MISMO NO HAY NINGUNA DIFERENCIA
    d = pd.DataFrame(X_ws1)
    d_new = pd.melt(d, value_vars=d.columns) #change the pd to a melt dataframe, that is, concat all signals
    d_new.drop(['variable'], axis=1, inplace=True) 
    d_new.rename(columns={'value':'T'}, inplace=True) #name the column target

    ## add the column group and class (eje sano etc)
    d_new['group'] = np.repeat(range(n_signals), n_sequence)
    d_new['time_idx'] = np.tile(np.arange(n_sequence), n_signals)
    if data_args.include_class: d_new['class'] = np.repeat(Y_ws1.values, n_sequence)

    d_new.to_csv(data_args.save_file+data_args.name_file)