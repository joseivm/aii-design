import pandas as pd
import numpy as np
import os
import re
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.linear_model import HuberRegressor, RANSACRegressor, TheilSenRegressor
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sktime.transformations.panel.catch22 import Catch22
import time
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

# Input files/dirs
RAW_DATA_DIR = os.path.join(PROJECT_DIR,'data','raw')
PRISM_DATA_DIR = os.path.join(RAW_DATA_DIR,'PRISM')
NASS_DATA_DIR = os.path.join(RAW_DATA_DIR,'NASS')

# Output files/dirs
TRANSFORMS_DIR = os.path.join(PROJECT_DIR,'data','time-series-transforms')

##### Data Loading ##### 
def load_yield_data(state):
    filename = f"{state} yields.csv"
    filepath = os.path.join(NASS_DATA_DIR,filename)
    df = pd.read_csv(filepath)

    relevant_cols = ['Year','County ANSI','County','Value']
    full_data_counties = df.loc[(df.Year == 2022) & (df['County ANSI'].notna()),'County']
            
    df = df.loc[df.County.isin(full_data_counties),relevant_cols]

    df['County ANSI'] = df['County ANSI'].astype(int).astype(str)
    df['County ANSI'] = df['County ANSI'].apply(lambda x: '0'+x if len(x) == 2 else x)
    df['County ANSI'] = df['County ANSI'].apply(lambda x: '00'+x if len(x) == 1 else x)

    df['CountyCode'] = '17' + df['County ANSI']
    return df

def load_prism_data():
    fnames = os.listdir(os.path.join(PRISM_DATA_DIR))
    filepaths = [os.path.join(PRISM_DATA_DIR,f) for f in fnames if '.csv' in f]
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
    df['t'] = df['Year'] - df['Year'].min()-1
    df['t^2'] = df['t']**2

    current_yr = df.Year.max()

    for county in df.County.unique():
        X = df.loc[df.County == county,['1','t','t^2']]
        y = df.loc[df.County == county,'Value']

        huber = HuberRegressor().fit(X,y)
        ransac = RANSACRegressor(random_state=1).fit(X,y)
        ts = TheilSenRegressor(random_state=1).fit(X,y)
        df.loc[df.County == county,'Huber Trend'] = huber.predict(X)
        df.loc[df.County == county,'RANSAC Trend'] = ransac.predict(X)
        df.loc[df.County == county,'TS Trend'] = ts.predict(X)

        df.loc[df.County == county, 'Huber Index'] = df.loc[(df.County == county) & (df.Year == current_yr),
                                                            'Huber Trend'].mean()/df['Huber Trend']
        df.loc[df.County == county, 'RANSAC Index'] = df.loc[(df.County == county) & (df.Year == current_yr),
                                                            'RANSAC Trend'].mean()/df['RANSAC Trend']
        df.loc[df.County == county, 'TS Index'] = df.loc[(df.County == county) & (df.Year == current_yr),
                                                            'TS Trend'].mean()/df['TS Trend']

    trend_vals = df.loc[df.Year == current_yr,['County','TS Trend','Huber Trend','RANSAC Trend']]
    # trend_vals = df.loc[df.Year == current_yr,['County','TS Trend','RANSAC Trend']]
    df = df.merge(trend_vals,suffixes=('',' current'),on='County')
    df['Huber Value'] = df['Value'] - df['Huber Trend'] + df['Huber Trend current']
    df['RANSAC Value'] = df['Value'] - df['RANSAC Trend'] + df['RANSAC Trend current']
    df['TS Value'] = df['Value'] - df['TS Trend'] + df['TS Trend current']

    # df['Huber Value'] = df['Value']*df['Huber Index']
    # df['RANSAC Value'] = df['Value']*df['RANSAC Index']
    # df['TS Value'] = df['Value']*df['TS Index']

    return df.drop(columns=['1','t','t^2'])

def add_detrended_values_old(df,random_state=50):
    df['1'] = 1
    df['t'] = df['Year'] - 1924
    df['t^2'] = df['t']**2

    X = df[['1','t','t^2']]
    y = df['Value']

    huber = HuberRegressor().fit(X,y)
    ransac = RANSACRegressor(random_state=random_state).fit(X,y)
    ts = TheilSenRegressor(random_state=1).fit(X,y)
    df['Huber Trend'] = huber.predict(X)
    df['RANSAC Trend'] = ransac.predict(X)
    df['TS Trend'] = ts.predict(X)

    df['Huber Value'] = df['Value'] - df['Huber Trend'] + df.loc[df.Year == 2018,'Huber Trend'].mean()
    df['RANSAC Value'] = df['Value'] - df['RANSAC Trend'] + df.loc[df.Year == 2018,'RANSAC Trend'].mean()
    df['TS Value'] = df['Value'] - df['TS Trend'] + df.loc[df.Year == 2018,'TS Trend'].mean()
    # df['Huber Index'] = df.loc[df.Year == 2018,'Huber Trend'].mean()/df['Huber Trend']
    # df['RANSAC Index'] = df.loc[df.Year == 2018,'RANSAC Trend'].mean()/df['RANSAC Trend']

    # df['Huber Value'] = df['Value']*df['Huber Index']
    # df['RANSAC Value'] = df['Value']*df['RANSAC Index']

    return df.drop(columns=['1','t','t^2'])

##### Transformations #####
def save_transformed_data(state, transformer, trf_name, length):
    # Load data
    ldf = create_loss_data(state,length)
    pdf = load_prism_data()
    pdf = pdf.loc[pdf.CountyYear.isin(ldf.index),:]

    # turn it into a ts tensor
    ts_data, obs_idx = create_raw_ts_tensors(pdf)

    # split into train/val/test (this includes the standardizing)
    train_cutoff = get_train_cutoff(ldf)
    train_mask = obs_idx.str[-4:].astype(int) <= train_cutoff
    val_mask = (obs_idx.str[-4:].astype(int) > train_cutoff) & (obs_idx.str[-4:].astype(int) <= 2007)
    test_mask = obs_idx.str[-4:].astype(int) > 2007

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

    scaler = StandardScaler()
    X_train_ts = scaler.fit_transform(X_train_ts)
    X_val_ts = scaler.transform(X_val_ts)
    X_test_ts = scaler.transform(X_test_ts)

    y_train = ldf.loc[train_idx,'TSLoss']
    y_val = ldf.loc[val_idx,'TSLoss']
    y_test = ldf.loc[test_idx,'TSLoss']

    # save
    save(X_train_ts, y_train, X_val_ts, y_val, X_test_ts, y_test, trf_name, state,length)
    save_dir = os.path.join(TRANSFORMS_DIR,state, trf_name)
    
def save_chen_data(state,length):
    ldf = create_loss_data(state,length)
    pdf = load_prism_data()

    df = ldf.merge(pdf,on=['CountyCode','Year'])
    df = df.sort_values('CountyYear')
    feats = [col for col in df.columns if re.search('[0-9]',col)]
    train_cutoff = get_train_cutoff(ldf)
    train_data = df.loc[df.Year <= train_cutoff,:]
    val_data = df.loc[(df.Year > train_cutoff) & (df.Year <= 2007),:]
    test_data = df.loc[df.Year > 2007,:]

    scaler = StandardScaler()
    X_train, y_train = train_data.loc[:,feats], train_data['TSLoss']
    X_val, y_val = val_data.loc[:,feats], val_data['TSLoss']
    X_test, y_test = test_data.loc[:,feats], test_data['TSLoss']

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"State: {state} Length: {length} Size: {X_train.shape[0]}")
    save_dates(train_data['CountyYear'],val_data['CountyYear'], test_data['CountyYear'],state,length)
    save(X_train, y_train, X_val, y_val, X_test, y_test,'chen',state,length)

def debug():
    # One way to do this would be to keep only the counties that would have enough in the 20 case
    # This would drop 2 in the Missouri case.
    states = ['Iowa','Missouri','Indiana']
    lengths = [i*10 for i in range(2,9)] + [83]
    for state in states:
        for length in lengths:
            ldf = create_loss_data(state,length)
            pdf = load_prism_data()

            df = ldf.merge(pdf,on=['CountyCode','Year'])
            df = df.sort_values('CountyYear')
            feats = [col for col in df.columns if re.search('[0-9]',col)]
            train_cutoff = get_train_cutoff(ldf)
            train_data = df.loc[df.Year <= train_cutoff,:]
            val_data = df.loc[(df.Year > train_cutoff) & (df.Year <= 2007),:]
            test_data = df.loc[df.Year > 2007,:]
            avg = test_data['TSLoss'].mean()
            print(f"{state} Length: {length} Loss: {avg}")

def create_loss_data(state, length):
    df = load_yield_data(state)
    df = add_detrended_values(df)
    df['Huber Bushel Loss'] = df['Huber Value'].max() - df['Huber Value']
    df['RANSAC Bushel Loss'] = df['RANSAC Value'].max() - df['RANSAC Value']
    df['TS Bushel Loss'] = df['TS Value'].max() - df['TS Value']
    df['HuberLoss'] = df['Huber Bushel Loss']*3.5
    df['RANSACLoss'] = df['RANSAC Bushel Loss']*3.5
    df['TSLoss'] = df['TS Bushel Loss']*3.5
    df = get_relevant_years(df,length)
    df['CountyYear'] = df['CountyCode'] + '-' + df['Year'].astype(str)
    df.set_index('CountyYear',inplace=True)
    return df.loc[df.TSLoss.notna(),:]

def get_relevant_years(df, length):
    if length is not None:
        cutoff_year = 2007-length + 1
        df = df.loc[df.Year >= cutoff_year,:]

    # making sure they have enough obs for length 20 case
    county_obs = df.loc[(df.Year >= 1988)&(df.Year <= 2007),:].groupby('County').size().reset_index(name='N')
    full_data_counties = county_obs.loc[county_obs['N'] >= 2*20/3,'County'] 

    df = df.loc[df.County.isin(full_data_counties),:]
    return df

def get_train_cutoff(ldf):
    min_year = int(ldf.Year.min())
    max_year = 2007
    years = np.array([i for i in range(min_year,max_year+1)])
    train, val = train_test_split(years,train_size=0.80,shuffle=False)
    return train.max()

def get_train_val_cutoffs(ldf):
    min_year = ldf.Year.min()
    max_year = ldf.Year.max()
    years = np.array([i for i in range(min_year,max_year+1)])
    train, test = train_test_split(years,train_size=0.7,shuffle=False)
    val, test = train_test_split(test,train_size=0.5,shuffle=False)
    return train.max(), val.max()

def save(X_train, y_train, X_val, y_val, X_test, y_test, trf_name, state, length):
    length = str(length)
    save_dir = os.path.join(TRANSFORMS_DIR,state,trf_name)
    Path(save_dir).mkdir(exist_ok=True)
    X_train_filename = f"X_train_{trf_name}_L{length}.npy"
    X_train_full_path = os.path.join(save_dir, X_train_filename)

    X_val_filename = f"X_val_{trf_name}_L{length}.npy"
    X_val_full_path = os.path.join(save_dir, X_val_filename)

    X_test_filename = f"X_test_{trf_name}_L{length}.npy"
    X_test_full_path = os.path.join(save_dir, X_test_filename)

    y_train_filename = f"y_train_{trf_name}_L{length}.npy"
    y_train_full_path = os.path.join(save_dir, y_train_filename)

    y_val_filename = f"y_val_{trf_name}_L{length}.npy"
    y_val_full_path = os.path.join(save_dir, y_val_filename)

    y_test_filename = f"y_test_{trf_name}_L{length}.npy"
    y_test_full_path = os.path.join(save_dir, y_test_filename)

    np.save(X_train_full_path, X_train)
    np.save(X_val_full_path, X_val)
    np.save(X_test_full_path, X_test)

    np.save(y_train_full_path, y_train)
    np.save(y_val_full_path, y_val)
    np.save(y_test_full_path, y_test)

def save_dates(train_years, val_years, test_years, state, length):
    length = str(length)
    save_dir = os.path.join(TRANSFORMS_DIR,state)
    train_yrs_fname = f"train_years_L{length}.npy"
    train_yrs_full_path = os.path.join(save_dir, train_yrs_fname)

    val_yrs_fname = f"val_years_L{length}.npy"
    val_yrs_full_path = os.path.join(save_dir, val_yrs_fname)

    test_yrs_fname = f"test_years_L{length}.npy"
    test_yrs_full_path = os.path.join(save_dir, test_yrs_fname)

    np.save(train_yrs_full_path, train_years)
    np.save(val_yrs_full_path, val_years)
    np.save(test_yrs_full_path, test_years)

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


# states = ['Iowa','Indiana','Missouri','Illinois']
states = ['Indiana']
# state = 'Missouri'
# lengths = [i*10 for i in range(2,9)] + [83]
lengths = [80]
# save_transformed_data(state,None,'raw')
for state in states:
    for length in lengths:
        print(length)
        save_chen_data(state,length)
        # transformer_dict = {'catch22':Catch22}
        # for name, transformer in transformer_dict.items():
        #     start = time.time()
        #     save_transformed_data(state,transformer,name,length)
        #     end = time.time()
        #     total = (end-start)/60
        #     print(f"{name}: {total} mins")