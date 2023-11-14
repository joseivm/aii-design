import pandas as pd
import numpy as np
import os
from functools import reduce
from sklearn.linear_model import HuberRegressor, RANSACRegressor
from sklearn.preprocessing import StandardScaler

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

def create_prediction_data(state):
    ldf = load_loss_data(state)
    pdf = load_prism_data()

    df = ldf.merge(pdf,on=['CountyCode','Year'])
    feats = [col for col in df.columns if col not in ['CountyCode','Year','HuberLoss']]
    train_data = df.loc[df.Year <= 1991,:]
    val_data = df.loc[(df.Year > 1991) & (df.Year <= 2003),:]
    test_data = df.loc[df.Year > 2003,:]

    scaler = StandardScaler()
    train_X, train_y = train_data.loc[:,feats], train_data['HuberLoss']
    val_X, val_y = val_data.loc[:,feats], val_data['HuberLoss']
    test_X, test_y = test_data.loc[:,feats], test_data['HuberLoss']

    train_X = scaler.fit_transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)

    train_X_fname = os.path.join(PRED_DATA_DIR,'train_X.npy')
    val_X_fname = os.path.join(PRED_DATA_DIR,'val_X.npy')
    test_X_fname = os.path.join(PRED_DATA_DIR,'test_X.npy')

    train_y_fname = os.path.join(PRED_DATA_DIR,'train_y.npy')
    val_y_fname = os.path.join(PRED_DATA_DIR,'val_y.npy')
    test_y_fname = os.path.join(PRED_DATA_DIR,'test_y.npy')

    np.save(train_X_fname,train_X)
    np.save(val_X_fname,val_X)
    np.save(test_X_fname,test_X)

    np.save(train_y_fname, train_y)
    np.save(val_y_fname,val_y)
    np.save(test_y_fname,test_y)

def load_loss_data(state):
    df = load_yield_data(state)
    df = add_detrended_values(df)
    df['Huber Bushel Loss'] = df['Huber Value'].max() - df['Huber Value']
    df['RANSAC Bushel Loss'] = df['RANSAC Value'].max() - df['RANSAC Value']
    df['HuberLoss'] = df['Huber Bushel Loss']*3.5
    df['RANSACLoss'] = df['RANSAC Bushel Loss']*3.5
    return df[['CountyCode','Year','HuberLoss']]

def load_yield_data(state):
    filename = f"{state} yields 3.csv"
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

def add_detrended_values(df):
    df['1'] = 1
    df['t'] = df['Year'] - 1924
    df['t^2'] = df['t']**2

    for county in df.County.unique():
        X = df.loc[df.County == county,['1','t','t^2']]
        y = df.loc[df.County == county,'Value']

        huber = HuberRegressor().fit(X,y)
        ransac = RANSACRegressor().fit(X,y)
        df.loc[df.County == county,'Huber Trend'] = huber.predict(X)
        df.loc[df.County == county,'RANSAC Trend'] = ransac.predict(X)

        df.loc[df.County == county, 'Huber Index'] = df.loc[(df.County == county) & (df.Year == 2018),
                                                            'Huber Trend'].mean()/df['Huber Trend']
        df.loc[df.County == county, 'RANSAC Index'] = df.loc[(df.County == county) & (df.Year == 2018),
                                                            'RANSAC Trend'].mean()/df['RANSAC Trend']

    df['Huber Value'] = df['Value']*df['Huber Index']
    df['RANSAC Value'] = df['Value']*df['RANSAC Index']

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

def find_best_random_state(df):
    best_i = 0
    best_mean_diff = 100
    for i in np.arange(45,145):
        tdf = add_detrended_values(df,i)
        mean_diff = np.abs(tdf['RANSAC Value'].mean()-tdf['Huber Value'].mean())
        if mean_diff < best_mean_diff:
            best_mean_diff = mean_diff
            best_i = i

    print(best_i)

def load_prism_data():
    fnames = os.listdir(PRISM_DATA_DIR)
    filepaths = [os.path.join(PRISM_DATA_DIR,f) for f in fnames]
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
    return pdf

create_prediction_data('Illinois')