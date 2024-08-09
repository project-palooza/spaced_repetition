import torch

def standardize_torch_vector(vec): # z-score 
    mean = vec.mean(dim=0, keepdim=True)
    std = vec.std(dim=0, keepdim=True)
    vec_standardized = (vec - mean) / std
    return vec_standardized

def minmax_torch_vector(vec):
    min_ = vec.min(dim=0)
    max_ = vec.max(dim=0)
    vec_minmax = (vec - min_.values)/(max_.values - min_.values)
    return vec_minmax

def mean_absolute_percentage_error(y_true, y_pred):
    return torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100

def root_mean_squared_error(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

def r_squared(y_true,y_pred):
    y_bar = torch.mean(y_true)

    residual_sum_of_squares = torch.sum((y_true - y_pred) ** 2)
    total_sum_of_squares = torch.sum((y_true - y_bar) ** 2)

    r_squared = 1 - residual_sum_of_squares/total_sum_of_squares

    return r_squared

def mape_rmse_rsq(y_true,y_pred):
    mape = mean_absolute_percentage_error(y_true,y_pred)
    rmse = root_mean_squared_error(y_true,y_pred)
    rsq = r_squared(y_true,y_pred)

    return mape,rmse,rsq