import torch
from torch.utils.data import DataLoader, Dataset
from utils.tools import StandardScaler
import numpy as np
import pandas as pd

#calculate max_len
def cal_max_len(data):
    max_len = 0
    for d in data:
        max_len = max(len(d), max_len)
    print('sequence_max_len:: ', max_len)
    for p in range(3):
        if max_len % 4 == 0:
            break
        else: max_len += 1
    print('modified sequence_max_len:: ', max_len)

    return max_len

#equalize length
def equal(data, max_len):
    dataset = []
    for d in data:
        d = np.array(d)
        if len(d) < max_len:
            dataset.append(torch.cat([torch.tensor(d), torch.zeros((max_len - d.shape[-2]),d.shape[-1])]))
        else:
            dataset.append(torch.tensor(d[:max_len]))
    return dataset

#dataset
class FDDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index], self.data[index]

    def __len__(self):
        return len(self.data)

class FDDataset_test(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

#read_data
def read_data(data_path, batch_size, fold, cols):
    scaler = StandardScaler()
    df = pd.read_csv(data_path)
    df = df[['MaterialID']+cols+['is_test']+['target']]
    raw_train_data = df[df['is_test']==0]
    raw_test_data = df[df['is_test']==1]
    
    scaler.fit(raw_train_data.iloc[:, 1:-2])
    train_col_data = scaler.transform(raw_train_data.iloc[:, 1:-2])
    test_col_data = scaler.transform(raw_test_data.iloc[:, 1:-2])

    raw_train_data = pd.concat([raw_train_data.iloc[:, :1], train_col_data, raw_train_data.iloc[:, -2:]],1)
    raw_test_data = pd.concat([raw_test_data.iloc[:, :1], test_col_data, raw_test_data.iloc[:, -2:]],1)
    
    all_train_data = []
    for ID in list(set(raw_train_data['MaterialID'].values)):
        all_train_data.append(raw_train_data[raw_train_data['MaterialID']==ID].iloc[:, 1:-2])

    input_size = np.array(all_train_data[0]).shape[-1]  # input_size

    test_data = []
    test_label = []
    for ID in list(set(raw_test_data['MaterialID'].values)):
        test_data.append(raw_test_data[raw_test_data['MaterialID']==ID].iloc[:, 1:-2])
        test_label.append(raw_test_data[raw_test_data['MaterialID']==ID].iloc[-1, -1])

    train_data = all_train_data[:int(len(all_train_data)*0.2*(fold-1))]+all_train_data[int(len(all_train_data)*0.2*fold):]
    valid_data = all_train_data[int(len(all_train_data)*0.2*(fold-1)):int(len(all_train_data)*0.2*fold)]

    max_len = cal_max_len(train_data)
    train_data = equal(train_data, max_len)
    valid_data = equal(valid_data, max_len)
    test_data = equal(test_data, max_len)

    #dataset
    train_dataset = FDDataset(train_data)
    valid_dataset = FDDataset(valid_data)
    test_dataset = FDDataset_test(test_data, test_label)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    return train_loader, valid_loader, test_loader, input_size, max_len, scaler

#read all data
def read_all_data(data_path, max_len, scaler, cols):
    df = pd.read_csv(data_path)
    df = df[['MaterialID']+cols+['is_test']+['target']]
    
    df_col = scaler.transform(df.iloc[:, 1:-2])
    df = pd.concat([df.iloc[:, :1], df_col, df.iloc[:, -2:]],1)

    all_data = []
    all_label = []
    for ID in list(set(df['MaterialID'].values)):
        all_data.append(df[df['MaterialID']==ID].iloc[:, 1:-2])
        all_label.append(df[df['MaterialID']==ID].iloc[-1, -1])

    all_data = equal(all_data, max_len)    

    all_dataset = FDDataset_test(all_data, all_label)
    all_loader = DataLoader(all_dataset, batch_size=1)

    return all_loader




