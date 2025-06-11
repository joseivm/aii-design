import pandas as pd
import numpy as np
import os
from pathlib import Path
import time
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sktime.transformations.panel.catch22 import Catch22
import multiprocessing as mp
import itertools


from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

# Input files/dirs
DATA_DIR = os.path.join(PROJECT_DIR,'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR,'processed')

# Output files/dirs
TS_TRANSFORMS_DIR = os.path.join(DATA_DIR,'time-series-transforms')

# Global Variables
ZONES = ['C1','C2','C3','N1','N2','N3','NE1','NE2','NE3','S1','S2']
TEST_YEARS = np.arange(2015, 2023)

# Note: shouldn't need to sample or to test set idx thing
# TODO: implement train, val, test, thin, change load_loss_data to simplified version, and re run.

##### Data loading #####
def load_loss_data():
    fpath = os.path.join(PROCESSED_DATA_DIR,'Thailand_loss_data.csv')
    df = pd.read_csv(fpath)
    df.set_index('ObsID',inplace=True)
    return df

def load_ts_data():
    fpath = os.path.join(DATA_DIR,'processed','Thailand_ts_data.csv')
    df = pd.read_csv(fpath)
    df['ObsID'] = df['TCode'].astype(str) + '-' + df['Year'].astype(str)
    variables = np.sort(df.Variable.unique())
    cols = ['ObsID'] + [str(i) for i in range(1,181)]
    dfs = []
    for variable in variables:
        tdf = df.loc[df.Variable == variable, cols]
        tdf.set_index('ObsID',inplace=True)
        dfs.append(tdf)
    return dfs

def load_train_test_panel_data(test_year, zone, train_ratio=2):
    ldf = load_loss_data()
    cols = ['Loss','WeightSum','LossClass','LossGroup']

    ldf = ldf.loc[ldf.Zone == zone,:]
    train_df = ldf.loc[ldf.Year != test_year, cols]
    test_df = ldf.loc[ldf.Year == test_year, cols]

    train_df, val_df = train_test_split(train_df, test_size=0.4, stratify=train_df['Loss'] > 0)

    train_pos = train_df.loc[train_df.Loss > 0, :]
    replace = train_ratio*(train_pos.shape[0]) > train_df.loc[train_df.Loss == 0, :].shape[0]
    train_neg = train_df.loc[train_df.Loss == 0, :].sample(
        train_ratio*len(train_pos), replace=replace)
    
    train_df = pd.concat([train_pos, train_neg])
    return train_df, val_df, test_df

##### Data Processing #####
def scale_data(X_train, X_val, X_test, transformer=QuantileTransformer):
    for channel in range(X_train.shape[1]):
        trf = transformer()
        X_train[:, channel, :] = trf.fit_transform(X_train[:, channel, :])
        X_val[:, channel, :] = trf.transform(X_val[:, channel, :])
        X_test[:, channel, :] = trf.transform(X_test[:, channel, :])

    return X_train, X_val, X_test

def standardize_data(X_train, X_val, X_test):
    trf = StandardScaler()
    X_train = trf.fit_transform(X_train)
    X_val = trf.transform(X_val)
    X_test = trf.transform(X_test)

    return X_train, X_val, X_test

def catch22_transform(X_train):
    X_trains = []
    for channel in range(X_train.shape[1]):
        # print(channel)
        trf = Catch22()
        # if channel != 2:
        #     trf = Catch22()
        # else:
        #     trf = Catch22(features=[i for i in range(22) if i not in [11,17]])

        trf_X_train = trf.fit_transform(X_train[:, [channel], :])
        X_trains.append(trf_X_train)
    return np.concatenate(X_trains, axis=1)

##### Time-series feature extraction #####
def create_ts_feature_data(params):
    transform, test_year, zone = params['transform'], params['test_year'], params['zone']
    # load time-series and panel data
    train_pdf, val_pdf, test_pdf = load_train_test_panel_data(test_year, zone)
    train_pdf, train_ts, val_pdf, val_ts, test_pdf, test_ts = create_train_test_ts_data(train_pdf, val_pdf, test_pdf)

    # fit transform on train data
    X_train_ts, X_val_ts, X_test_ts = transform_data(train_ts, val_ts, test_ts, transform)

    # transform and save data
    save_dir = os.path.join(TS_TRANSFORMS_DIR,'Thailand', zone, transform)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    X_train_ts_filename = os.path.join(save_dir, f"X_train_{test_year}.npy")
    X_train_idx_filename = os.path.join(save_dir, f"X_train_idx_{test_year}.npy")
    X_val_ts_filename = os.path.join(save_dir, f"X_val_{test_year}.npy")
    X_val_idx_filename = os.path.join(save_dir, f"X_val_idx_{test_year}.npy")
    X_test_ts_filename = os.path.join(save_dir, f"X_test_{test_year}.npy")
    X_test_idx_filename = os.path.join(save_dir, f"X_test_idx_{test_year}.npy")
    np.save(X_train_ts_filename, X_train_ts)
    np.save(X_train_idx_filename, train_pdf.index)
    np.save(X_val_ts_filename, X_val_ts)
    np.save(X_val_idx_filename, val_pdf.index)
    np.save(X_test_ts_filename, X_test_ts)
    np.save(X_test_idx_filename, test_pdf.index)

def transform_data(train_ts, val_ts, test_ts, transform):
    if transform == 'ts_raw':
        # train_ts, val_ts, test_ts = scale_data(train_ts, val_ts, test_ts)
        X_train_ts = np.reshape(train_ts,(train_ts.shape[0],-1))
        X_val_ts = np.reshape(val_ts,(val_ts.shape[0],-1))
        X_test_ts = np.reshape(test_ts,(test_ts.shape[0],-1))
    elif transform == 'catch22':
        # train_ts, val_ts, test_ts = scale_data(train_ts, val_ts, test_ts)
        X_train_ts = catch22_transform(train_ts)
        X_val_ts = catch22_transform(val_ts)
        X_test_ts = catch22_transform(test_ts)
    else:
        # train_ts, val_ts, test_ts = scale_data(train_ts, val_ts, test_ts)
        trf = MiniRocketMultivariate()
        X_train_ts = trf.fit_transform(train_ts)
        X_val_ts = trf.transform(val_ts)
        X_test_ts = trf.transform(test_ts)
    
    X_train_ts, X_val_ts, X_test_ts = standardize_data(X_train_ts, X_val_ts, X_test_ts)
    return X_train_ts, X_val_ts, X_test_ts

def create_train_test_ts_data(train_pdf, val_pdf, test_pdf):
    ts_dfs = load_ts_data()
    ts_indices = [df.index for df in ts_dfs]
    full_data_indices = reduce(np.intersect1d, ts_indices)

    train_pdf = train_pdf.loc[train_pdf.index.isin(full_data_indices),:]
    val_pdf = val_pdf.loc[val_pdf.index.isin(full_data_indices),:]
    test_pdf = test_pdf.loc[test_pdf.index.isin(full_data_indices),:]

    train_ts_data_matrices = tuple(df.loc[train_pdf.index,:].to_numpy() for df in ts_dfs)
    train_ts_X = np.stack(train_ts_data_matrices, axis=1)

    val_ts_data_matrices = tuple(df.loc[val_pdf.index,:].to_numpy() for df in ts_dfs)
    val_ts_X = np.stack(val_ts_data_matrices, axis=1)

    test_ts_data_matrices = tuple(df.loc[test_pdf.index,:].to_numpy() for df in ts_dfs)
    test_ts_X = np.stack(test_ts_data_matrices, axis=1)
    return train_pdf, train_ts_X, val_pdf, val_ts_X, test_pdf, test_ts_X

def create_param_dicts(test_years, transforms):
    dicts = []
    for yr, transform, zone in itertools.product(test_years, transforms, ZONES):
        ndict = {'test_year': yr, 'transform': transform, 'zone': zone}
        dicts.append(ndict)

    return dicts

if __name__ == '__main__':
    years = np.arange(2015, 2023)
    transforms = ['ts_raw','catch22']
    param_dicts = create_param_dicts(years, transforms)
    # for params in param_dicts:
    #     print(params)
        # create_ts_feature_data(params)
    with mp.Pool(3) as pool:
            pool.map(create_ts_feature_data, param_dicts)
