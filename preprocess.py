import random
import numpy as np
import pandas as pd 
import torch
from utils import minmax_torch_vector
from torch.utils.data import DataLoader, TensorDataset

# if we do the clipping pre-normalization
# we should no longer minmax scale p
# should we still minmax scale h though?

def hclip(series,min_half_life = .010 ,max_half_life = 274.0):
    return series.clip(lower=min_half_life, upper=max_half_life)

def pclip(series,min_p_recall = .0001 ,max_p_recall = .9999):
    return series.clip(lower=min_p_recall, upper=max_p_recall)

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
    df['p_recall'] = pclip(df['p_recall'])
    # define observed half life
    df["h"] = hclip(-1*df["delta_days"]/(np.log2(df["p_recall"])))
    # clip half life (15 min - 274 day max)

    #   p = pclip(float(row['p_recall']))
    #     t = float(row['delta'])/(60*60*24)  # convert time delta to days
    #     h = hclip(-t/(math.log(p, 2)))

    return df


def sr_data_loader(df,batch_size,simple=True):

    if simple:
        df = df[['p_recall','delta_days','h','history_seen','history_correct']]
        x = minmax_torch_vector(torch.tensor(df[['history_seen', 'history_correct']].values, dtype=torch.float32))
    else:
        raise NotImplementedError
    
    p = torch.tensor(df['p_recall'].values, dtype=torch.float32)
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