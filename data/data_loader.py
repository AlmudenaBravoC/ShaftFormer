import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data  # shape: [s, b, d]
        self.s, self.b = data.shape
        
    def __len__(self):
        return self.b
    
    def __getitem__(self, index):
        return self.data[:, index]  # return (s, d) and a dummy target tensor

class CustomDatasetTarget(Dataset):
    def __init__(self, data, target, feat, idx):
        self.data = data  # shape: [s, b]
        self.s, self.b = data.shape
        self.feat = feat
        self.idx = torch.tensor(idx)

        self.target = target
        
    def __len__(self):
        return self.b
    
    def __getitem__(self, index):
        return self.data[:, index], self.target[index], self.feat[index,:], self.idx[index]

def collate_fn(batch):
    inputs = [item[0] for item in batch]  # list of (s, d) tensors
    return torch.stack(inputs, dim=1)  # return input tensor of shape (s, b, d) and target tensor of shape (b,)

def collate_fn_target(batch):
    inputs = [item[0] for item in batch]  # list of (s, d) tensors
    targets = [item[1] for item in batch]  # list of target tensors
    features = [item[2] for item in batch] #list of the features of the tensor
    idxs = [item[3] for item in batch] #list of the index of the signal
    return torch.stack(inputs, dim=1), torch.stack(targets), torch.stack(features, dim=0), torch.stack(idxs)
