import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import cvxpy as cp
import matplotlib.pyplot as plt
import os
import time

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

# Input files/dirs
RAW_DATA_DIR = PROJECT_DIR + '/data/raw'
SURVEY_DATA_DIR = PROJECT_DIR + '/data/raw/Survey Data'
panel_admin_cw_filename = RAW_DATA_DIR + '/Kenya Geodata/panel_admin_cw.csv'
NDVI_covariate_data = PROJECT_DIR + '/data/processed/NDVI/covariate_matrix.csv'

# Output files/dirs
FIGURES_DIR = PROJECT_DIR + '/output/figures'
TABLES_DIR = PROJECT_DIR + '/output/tables'

def load_survey_data():
    livestock_filename = SURVEY_DATA_DIR + '/S6C Livestock Losses.dta'
    ldf = pd.read_stata(livestock_filename,convert_categoricals=False)
    ldf.rename(columns={'s6q20a':'Year','s6q20b':'Month','s6q24':'Losses','s6q25':'Adult Losses'},inplace=True)

    household_info_filename = SURVEY_DATA_DIR + '/S0A Household Identification Information.dta'
    hdf = pd.read_stata(household_info_filename,convert_categoricals=False)

    hh_villages = hdf.loc[hdf.location != '',:].groupby(['hhid','location']).size().reset_index(name='N')
    ldf = ldf.merge(hh_villages,left_on='hhid',right_on='hhid')

    panel_geo_cw = pd.read_csv(panel_admin_cw_filename)
    panel_geo_cw['location'] = panel_geo_cw['location'].str.upper()
    ldf = ldf.merge(panel_geo_cw,left_on='location',right_on='location')
    
    ldf = ldf.loc[ldf.Month > 0,:]
    ldf['Day'] = 1
    ldf['Date'] = pd.to_datetime(ldf[['Year','Month','Day']])
    ldf = add_seasons(ldf)
    return ldf.groupby(['Cluster','Location','Season'])['Losses'].sum().reset_index(name='Losses')

def add_seasons(df):
    season_dict = {'LRLD':['03-01','09-30'],'SRSD':['10-01','02-28']}
    
    season_col = {'LRLD':'Season','SRSD':'Season'}

    years = range(2008,2017)
    for year in years:
        for season, dates in season_dict.items():
            season_start_year = year-1 if season in ['LRLD Pre','LRLD Full'] else year
            start_date, end_date = dates
            start_date = str(season_start_year)+'-'+start_date 
            start_date = pd.to_datetime(start_date,format='%Y-%m-%d')

            season_end_year = year+1 if season in ['SRSD','SRSD Full'] else year
            end_date = str(season_end_year)+'-'+end_date 
            end_date = pd.to_datetime(end_date,format='%Y-%m-%d')

            season_name = season + ' ' + str(year)
            col = season_col[season]
            df.loc[(df.Date >= start_date) & (df.Date <= end_date),col] = season_name 
    return df

def make_regression_data():
    df = pd.read_csv(NDVI_covariate_data)
    ldf = load_survey_data()
    df = df.merge(ldf,left_on=['Location','Season'],right_on=['Location','Season'])
    df['SRSD'] = df.Season.str.contains('SRSD')
    df['Good Regime'] = df.PosNDVI >= 0
    return df

def create_prediction_models():
    df = make_regression_data()
    udf = df.loc[df.Cluster == 'Upper',:]
    ldf = df.loc[df.Cluster == 'Lower',:]

    train_x = udf[['PosZNDVI','NegZNDVI','PreNDVI','SRSD','Good Regime']]
    train_y = udf['Losses']
    upper_model = LinearRegression().fit(train_x,train_y)
    upper_model.score(train_x,train_y)

    train_x = ldf[['PosZNDVI','NegZNDVI','PreNDVI','SRSD','Good Regime']]
    train_y = ldf['Losses']
    lower_model = LinearRegression().fit(train_x,train_y)
    lower_model.score(train_x,train_y)
