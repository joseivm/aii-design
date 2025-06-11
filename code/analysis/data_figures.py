import pandas as pd
import numpy as np
import os
import re
from functools import reduce
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor, RANSACRegressor, TheilSenRegressor

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

def create_loss_data(state, length):
    df = load_yield_data(state)
    df = add_detrended_values(df)
    df['State'] = state
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

def historical_yield_plots():
    length = 83
    ildf = create_loss_data('Illinois',length)
    indf = create_loss_data('Indiana',length)
    iadf = create_loss_data('Iowa',length)
    modf = create_loss_data('Missouri',length)


    bdf = pd.concat([ildf,indf,iadf,modf])
    bdf.to_csv(os.path.join(PROJECT_DIR,'data','processed','midwest_yield_data.csv'))

historical_yield_plots()

# plt.close()
# tst = bdf.groupby(['State','Year'])['Value'].mean().reset_index(name='Yield')
# sns.relplot(data=tst, x='Year',y='Yield',kind='line')
# plt.show()
