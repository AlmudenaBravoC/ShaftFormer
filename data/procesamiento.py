from sklearn.preprocessing import MinMaxScaler
import numpy as np

def minMax(signal, feature_range):
    scaler = MinMaxScaler(feature_range=feature_range)
    signal = signal.reshape(-1,1)
    scaler.fit(signal)

    #save the necessary values
    min_  = scaler.min_
    scale_ = scaler.scale_

    #compute the transformation of the signal
    sig_processed = scaler.transform(signal)

    return sig_processed.squeeze(), min_, scale_

def preprocessing(n_signals, data, feature_range = (-1, 1), folder_name="data"):
    data_new = []
    inform_signal = []
    for i in range(n_signals):
        sig , min_, scale_ = minMax(data[i,:], feature_range)
        data_new.append(sig)
        inform_signal.append((min_, scale_))
    np.save(f'{folder_name}/min_max_scaler.npy', inform_signal)

    return data_new