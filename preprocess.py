import random
import numpy as np
import pandas as pd 
import torch
from utils import minmax_torch_vector
from torch.utils.data import DataLoader, TensorDataset

def sr_data_reader(p = .001):
    
    df = pd.read_csv(
            'learning_traces.13m.csv',
            header=0, 
            skiprows=lambda i: i>0 and random.random() > p
    )
    # timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    df.drop_duplicates(inplace=True)
    # convert seconds to days
    df["delta_days"] = np.round(df.loc[:,"delta"].copy()/(60*60*24),1)
    df.drop(columns = ["delta"],inplace=True)
    # add cushion to zero
    df.loc[df["p_recall"] == 0,"p_recall"] = df.loc[df["p_recall"] == 0,"p_recall"] + 1e-3
    # define observed half life
    df["h"] = -1*df["delta_days"]/(np.log2(df["p_recall"]) + 1e-3)

    return df


def sr_data_loader(df,batch_size,simple=True):

    if simple:
        df = df[['p_recall','delta_days','h','history_seen','history_correct']]
        x = minmax_torch_vector(torch.tensor(df[['history_seen', 'history_correct']].values, dtype=torch.float32))
    else:
        raise NotImplementedError
    
    p = minmax_torch_vector(torch.tensor(df['p_recall'].values, dtype=torch.float32))
    h = minmax_torch_vector(torch.tensor(df['h'].values, dtype=torch.float32))
    delta = minmax_torch_vector(torch.tensor(df['delta_days'].values, dtype=torch.float32))
    dataset = TensorDataset(x, p, h, delta)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def leitner_data(df):
    """returns leitner feature matrix and delta vector"""
    df['history_incorrect'] = df['history_seen'] - df['history_correct']
    leitner_x = minmax_torch_vector(torch.tensor(df[['history_correct', 'history_incorrect']].values, dtype=torch.float32))
    delta = minmax_torch_vector(torch.tensor(df['delta_days'].values, dtype=torch.float32))

    return leitner_x, delta