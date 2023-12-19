import pandas as pd
import numpy as np
import os
import re
from functools import reduce
from sklearn.linear_model import HuberRegressor, RANSACRegressor, TheilSenRegressor
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sktime.transformations.panel.catch22 import Catch22
import time

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

# Input files/dirs
RAW_DATA_DIR = os.path.join(PROJECT_DIR,'data','raw')
PRISM_DATA_DIR = os.path.join(RAW_DATA_DIR,'PRISM')
NASS_DATA_DIR = os.path.join(RAW_DATA_DIR,'NASS')

# Output files/dirs
PRED_DATA_DIR = os.path.join(PROJECT_DIR,'data','processed','Illinois')
TRANSFORMS_DIR = os.path.join(PROJECT_DIR,'data','time-series-transforms')

##### Data Loading ##### 
def load_yield_data(state):
    filename = f"{state} yields new.csv"
    filepath = os.path.join(NASS_DATA_DIR,filename)
    df = pd.read_csv(filepath)

    relevant_cols = ['Year','County ANSI','County','Value']
    county_obs = df.groupby('County')['Year'].nunique().reset_index(name='N')
    full_data_counties = county_obs.loc[county_obs.N >= 93, 'County']

    df = df.loc[df.County.isin(full_data_counties),relevant_cols]

    df['County ANSI'] = df['County ANSI'].astype(int).astype(str)
    df['County ANSI'] = df['County ANSI'].apply(lambda x: '0'+x if len(x) == 2 else x)
    df['County ANSI'] = df['County ANSI'].apply(lambda x: '00'+x if len(x) == 1 else x)

    df['CountyCode'] = '17' + df['County ANSI']
    return df

def load_prism_data(state):
    fnames = os.listdir(os.path.join(PRISM_DATA_DIR,state))
    filepaths = [os.path.join(PRISM_DATA_DIR,state,f) for f in fnames]
    dfs = [pd.read_csv(filepath) for filepath in filepaths]
    pdf = pd.concat(dfs)
    pdf = pdf.loc[pdf.Name.notna(),:]
    pdf['CountyCode'] = pdf.Name.astype(int).astype(str)
    pdf['Date'] = pd.to_datetime(pdf['Date'],format="%Y-%m")
    pdf['Year'] = pdf.Date.dt.year
    pdf['Month'] = pdf.Date.dt.month.astype(int)
    col_names = {'Elevation (ft)':'Elevation','ppt (inches)':'ppt',
                    'tmin (degrees F)':'tmin','tmax (degrees F)':'tmax',
                    'tdmean (degrees F)':'tdmean','vpdmin (hPa)':'vpdmin',
                    'vpdmax (hPa)':'vpdmax'}
    pdf.rename(columns=col_names,inplace=True)

    weather_vars = ['ppt','tmin','tmax','tdmean','vpdmin','vpdmax']
    weather_dfs = []
    for var in weather_vars:
        tdf = pd.pivot(pdf,index=['CountyCode','Year'],columns='Month',values=var).reset_index()
        new_cols = [f"{var}{col}" if isinstance(col,int) else col for col in tdf.columns]
        tdf.columns = new_cols
        weather_dfs.append(tdf)

    pdf = reduce(lambda x,y: pd.merge(x,y, on=['CountyCode','Year']),weather_dfs)
    pdf['CountyYear'] = pdf['CountyCode'] + '-' + pdf.Year.astype(str)
    return pdf

def add_detrended_values(df):
    df['1'] = 1
    df['t'] = df['Year'] - 1924
    df['t^2'] = df['t']**2

    for county in df.County.unique():
        X = df.loc[df.County == county,['1','t','t^2']]
        y = df.loc[df.County == county,'Value']

        huber = HuberRegressor().fit(X,y)
        ransac = RANSACRegressor(random_state=1).fit(X,y)
        ts = TheilSenRegressor(random_state=1).fit(X,y)
        df.loc[df.County == county,'Huber Trend'] = huber.predict(X)
        df.loc[df.County == county,'RANSAC Trend'] = ransac.predict(X)
        df.loc[df.County == county,'TS Trend'] = ts.predict(X)

        df.loc[df.County == county, 'Huber Index'] = df.loc[(df.County == county) & (df.Year == 2018),
                                                            'Huber Trend'].mean()/df['Huber Trend']
        df.loc[df.County == county, 'RANSAC Index'] = df.loc[(df.County == county) & (df.Year == 2018),
                                                            'RANSAC Trend'].mean()/df['RANSAC Trend']
        df.loc[df.County == county, 'TS Index'] = df.loc[(df.County == county) & (df.Year == 2018),
                                                            'TS Trend'].mean()/df['TS Trend']

    df['Huber Value'] = df['Value']*df['Huber Index']
    df['RANSAC Value'] = df['Value']*df['RANSAC Index']
    df['TS Value'] = df['Value']*df['TS Index']

    return df.drop(columns=['1','t','t^2'])

def add_detrended_values_old(df,random_state=50):
    df['1'] = 1
    df['t'] = df['Year'] - 1924
    df['t^2'] = df['t']**2

    X = df[['1','t','t^2']]
    y = df['Value']

    huber = HuberRegressor().fit(X,y)
    ransac = RANSACRegressor(random_state=random_state).fit(X,y)
    df['Huber Trend'] = huber.predict(X)
    df['RANSAC Trend'] = ransac.predict(X)

    df['Huber Index'] = df.loc[df.Year == 2018,'Huber Trend'].mean()/df['Huber Trend']
    df['RANSAC Index'] = df.loc[df.Year == 2018,'RANSAC Trend'].mean()/df['RANSAC Trend']

    df['Huber Value'] = df['Value']*df['Huber Index']
    df['RANSAC Value'] = df['Value']*df['RANSAC Index']
    return df.drop(columns=['1','t','t^2'])

##### Transformations #####
def save_transformed_data(state, transformer, trf_name):
    # Load data
    ldf = create_loss_data(state)
    pdf = load_prism_data(state)
    pdf = pdf.loc[pdf.CountyYear.isin(ldf.index),:]

    # turn it into a ts tensor
    ts_data, obs_idx = create_raw_ts_tensors(pdf)

    # split into train/val/test (this includes the standardizing)
    train_mask = obs_idx.str[-4:].astype(int) <= 1991
    val_mask = (obs_idx.str[-4:].astype(int) > 1991) & (obs_idx.str[-4:].astype(int) <= 2003)
    test_mask = obs_idx.str[-4:].astype(int) > 2003

    train_idx = obs_idx.loc[train_mask]
    val_idx = obs_idx.loc[val_mask]
    test_idx = obs_idx.loc[test_mask]

    train_ts = ts_data[train_mask,:,:]
    val_ts = ts_data[val_mask,:,:]
    test_ts = ts_data[test_mask,:,:]

    train_ts, val_ts, test_ts = scale_data(train_ts,val_ts,test_ts)

    # transform
    if transformer is not None:
        trf = transformer()
        X_train_ts = trf.fit_transform(train_ts)
        X_val_ts = trf.transform(val_ts)
        X_test_ts = trf.transform(test_ts)
    else:
        X_train_ts, X_val_ts, X_test_ts = train_ts, val_ts, test_ts

    # scaler = StandardScaler()
    # X_train_ts = scaler.fit_transform(X_train_ts)
    # X_val_ts = scaler.transform(X_val_ts)
    # X_test_ts = scaler.transform(X_test_ts)

    y_train = ldf.loc[train_idx,'TSLoss']
    y_val = ldf.loc[val_idx,'TSLoss']
    y_test = ldf.loc[test_idx,'TSLoss']

    # save
    save(X_train_ts, y_train, X_val_ts, y_val, X_test_ts, y_test, trf_name)

def save_chen_data(state):
    ldf = create_loss_data(state)
    pdf = load_prism_data(state)

    df = ldf.merge(pdf,on=['CountyCode','Year'])
    df = df.sort_values('CountyYear')
    feats = [col for col in df.columns if re.search('[0-9]',col)]
    train_data = df.loc[df.Year <= 1991,:]
    val_data = df.loc[(df.Year > 1991) & (df.Year <= 2003),:]
    test_data = df.loc[df.Year > 2003,:]

    scaler = StandardScaler()
    X_train, y_train = train_data.loc[:,feats], train_data['TSLoss']
    X_val, y_val = val_data.loc[:,feats], val_data['TSLoss']
    X_test, y_test = test_data.loc[:,feats], test_data['TSLoss']

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    save(X_train, y_train, X_val, y_val, X_test, y_test,'chen')

def create_loss_data(state):
    df = load_yield_data(state)
    df = add_detrended_values(df)
    df['CountyYear'] = df['CountyCode'] + '-' + df['Year'].astype(str)
    df.set_index('CountyYear',inplace=True)
    df['Huber Bushel Loss'] = df['Huber Value'].max() - df['Huber Value']
    df['RANSAC Bushel Loss'] = df['RANSAC Value'].max() - df['RANSAC Value']
    df['TS Bushel Loss'] = df['TS Value'].max() - df['TS Value']
    df['HuberLoss'] = df['Huber Bushel Loss']*3.5
    df['RANSACLoss'] = df['RANSAC Bushel Loss']*3.5
    df['TSLoss'] = df['TS Bushel Loss']*3.5
    return df.loc[df.TSLoss.notna(),:]

def save(X_train, y_train, X_val, y_val, X_test, y_test, trf_name):
    X_train_filename = f"X_train_{trf_name}.npy"
    X_train_full_path = os.path.join(TRANSFORMS_DIR,X_train_filename)

    X_val_filename = f"X_val_{trf_name}.npy"
    X_val_full_path = os.path.join(TRANSFORMS_DIR,X_val_filename)

    X_test_filename = f"X_test_{trf_name}.npy"
    X_test_full_path = os.path.join(TRANSFORMS_DIR,X_test_filename)

    y_train_filename = f"y_train_{trf_name}.npy"
    y_train_full_path = os.path.join(TRANSFORMS_DIR,y_train_filename)

    y_val_filename = f"y_val_{trf_name}.npy"
    y_val_full_path = os.path.join(TRANSFORMS_DIR,y_val_filename)

    y_test_filename = f"y_test_{trf_name}.npy"
    y_test_full_path = os.path.join(TRANSFORMS_DIR,y_test_filename)

    np.save(X_train_full_path, X_train)
    np.save(X_val_full_path, X_val)
    np.save(X_test_full_path, X_test)

    np.save(y_train_full_path, y_train)
    np.save(y_val_full_path, y_val)
    np.save(y_test_full_path, y_test)

def scale_data(X_train, X_val, X_test, transformer=QuantileTransformer):
    for channel in range(X_train.shape[1]):
        trf = transformer()
        X_train[:, channel, :] = trf.fit_transform(X_train[:, channel, :])
        X_val[:, channel, :] = trf.transform(X_val[:, channel, :])
        X_test[:, channel, :] = trf.transform(X_test[:, channel, :])

    return X_train, X_val, X_test

def create_raw_ts_tensors(pdf):
    # This will just take the ts data and return it in tensor format along with it's indices
    obs_order = pdf['CountyYear'].sort_values()
    weather_vars = ['ppt','tmin','tmax','tdmean','vpdmin','vpdmax']
    weather_dfs = []
    for var in weather_vars:
        cols = ['CountyYear'] + [col for col in pdf.columns if var in col]
        tdf = pdf.loc[:,cols].set_index('CountyYear').sort_index().to_numpy()
        weather_dfs.append(tdf)
    
    ts_data = np.stack(weather_dfs,axis=1)
    return ts_data, obs_order

state = 'Illinois'
# save_transformed_data(state,None,'raw')
save_chen_data(state)
transformer_dict = {'rocket':MiniRocketMultivariate,'catch22':Catch22, 'raw':None}
for name, transformer in transformer_dict.items():
    start = time.time()
    save_transformed_data(state,transformer,name)
    end = time.time()
    total = (end-start)/60
    print(f"{name}: {total} mins")