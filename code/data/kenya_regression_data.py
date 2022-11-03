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
survey_dates = {'R1':'2009-10-01','R2':'2010-10-01','R3':'2011-10-01','R4':'2012-10-01',
                'R5':'2013-10-01','R6':'2015-10-01'}

# Output files/dirs
PROCESSED_DATA_DIR = PROJECT_DIR + '/data/processed'
FIGURES_DIR = PROJECT_DIR + '/output/figures'
TABLES_DIR = PROJECT_DIR + '/output/tables'
kenya_reg_data_filename  = PROCESSED_DATA_DIR + '/Kenya/kenya_reg_data.csv'
kenya_hh_data_filename = PROCESSED_DATA_DIR + '/Kenya/kenya_hh_data.csv'

##### Data Loading #####
def load_livestock_data(groupby_cols):
    livestock_stock_filename = SURVEY_DATA_DIR + '/S6A Livestock Stock.dta'
    sdf = pd.read_stata(livestock_stock_filename,convert_categoricals=False)
    sdf.rename(columns={'s6q2':'NumAnimals'},inplace=True)
    sdf.loc[sdf.NumAnimals == -77,'NumAnimals'] = np.nan 
    sdf['Round'] = 'R' + sdf['round'].astype(int).astype(str)
    sdf['Date'] = sdf.Round.apply(lambda x: survey_dates[x])
    sdf['Month'] = 10
    sdf['Year'] = sdf.Date.apply(lambda x: int(x[:4]))
    sdf = sdf.groupby(groupby_cols)['NumAnimals'].sum().reset_index()
    return sdf

def load_intake_data(groupby_cols):
    intake_filename = SURVEY_DATA_DIR + '/S6D Livestock Intake.dta'
    idf = pd.read_stata(intake_filename,convert_categoricals=False)
    idf.rename(columns={'s6q30':'NumIntake','s6q27a':'Year','s6q27b':'Month'},inplace=True)
    idf.loc[idf.NumIntake == -77,'NumIntake'] = np.nan
    idf = idf.groupby(groupby_cols)['NumIntake'].sum().reset_index()
    return idf

def load_offtake_data(groupby_cols):
    offtake_filename = SURVEY_DATA_DIR + '/S6E Livestock Offtake.dta'
    odf = pd.read_stata(offtake_filename,convert_categoricals=False)
    odf.rename(columns={'s6q39':'NumOfftake','s6q36a':'Year','s6q36b':'Month'},inplace=True)
    odf.loc[odf.NumOfftake == -77,'NumOffake'] = np.nan
    odf = odf.groupby(groupby_cols)['NumOfftake'].sum().reset_index()
    return odf

def load_birth_data(groupby_cols):
    births_filename = SURVEY_DATA_DIR + '/S6F Livestock Births.dta'
    bdf = pd.read_stata(births_filename,convert_categoricals=False)
    bdf.rename(columns={'s6q47':'NumBorn','s6q45a':'Year','s6q45b':'Month'},inplace=True)
    bdf.loc[bdf.NumBorn == -77,'NumBorn'] = np.nan
    bdf = bdf.groupby(groupby_cols)['NumBorn'].sum().reset_index()
    return bdf

def load_slaughter_data(groupby_cols):
    slaughter_filename = SURVEY_DATA_DIR + '/S6G Livestock Slaughter.dta'
    sldf = pd.read_stata(slaughter_filename,convert_categoricals=False)
    sldf.rename(columns={'s6q51':'NumSlaughtered','s6q49a':'Year','s6q49b':'Month'},inplace=True)
    sldf.loc[sldf.NumSlaughtered == -77,'NumSlaughtered'] = np.nan
    sldf = sldf.groupby(groupby_cols)['NumSlaughtered'].sum().reset_index()
    return sldf

def load_loss_data(groupby_cols):
    livestock_filename = SURVEY_DATA_DIR + '/S6C Livestock Losses.dta'
    ldf = pd.read_stata(livestock_filename,convert_categoricals=False)
    ldf.rename(columns={'s6q20a':'Year','s6q20b':'Month','s6q24':'Losses'},inplace=True)
    ldf.loc[ldf.Losses < 0,'Losses'] = np.nan
    ldf = ldf.groupby(groupby_cols)['Losses'].sum().reset_index()
    if ('Year' in groupby_cols) and ('Month' in groupby_cols):
        ldf = ldf.loc[ldf.Month > 0,:]
        ldf['Day'] = 1
        ldf['Date'] = pd.to_datetime(ldf[['Year','Month','Day']])
    return ldf

def get_herd_size():
    groupby_cols = ['hhid','Year','Month']
    household_info_filename = SURVEY_DATA_DIR + '/S0A Household Identification Information.dta'
    hdf = pd.read_stata(household_info_filename,convert_categoricals=False)

    hhids = hdf.hhid.unique()
    years = [i for i in range(2008,2016)]
    months = [i for i in range(1,13)]
    hdf_data = {'Year':years,'Month':months,'hhid':hhids}
    index = pd.MultiIndex.from_product(hdf_data.values(),names=hdf_data.keys())
    hdf = pd.DataFrame(index=index).reset_index()

    sdf = load_livestock_data(groupby_cols)
    idf = load_intake_data(groupby_cols)
    bdf = load_birth_data(groupby_cols)
    odf = load_offtake_data(groupby_cols)
    sldf = load_slaughter_data(groupby_cols)
    ldf = load_loss_data(groupby_cols)

    for df in [sdf,idf,odf,bdf,sldf,ldf]:
        hdf = hdf.merge(df,left_on=groupby_cols,right_on=groupby_cols,how='left')

    hdf['NetGain'] = hdf['NumBorn']+hdf['NumIntake']-hdf['NumOfftake']-hdf['NumSlaughtered']-hdf['Losses']
    hdf.fillna(value=0,inplace=True)
    hdf['Day'] = 1
    hdf['Date'] = pd.to_datetime(hdf[['Year','Month','Day']])
    hdf['HerdSize'] = np.nan
    for year in range(2009,2016):
        # if year == 2014:
            # hdf.loc[hdf.Date == '2014-10-01','NumAnimals'] = hdf.loc[hdf.Date == '2014-09-01','HerdSize'].to_numpy()
        for hhid in hdf.hhid.unique():
            start_date = str(year)+'-10-01'
            start_date = pd.to_datetime(start_date,format='%Y-%m-%d')

            end_date = str(year+1) + '-09-01'
            end_date = pd.to_datetime(end_date,format='%Y-%m-%d')

            num_animals = hdf.loc[(hdf.Date >= start_date)&(hdf.Date <= end_date) & (hdf.hhid == hhid),'NumAnimals']
            net_gains = hdf.loc[(hdf.Date >= start_date)&(hdf.Date <= end_date) & (hdf.hhid == hhid),'NetGain']
            animal_stock = num_animals + net_gains
            hdf.loc[(hdf.Date >= start_date)&(hdf.Date <= end_date) & (hdf.hhid == hhid),'HerdSize'] = animal_stock.cumsum()

    return hdf[['Year','Month','hhid','HerdSize']]

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

##### Covariate Matrix #####
def make_household_data():
    ldf = load_loss_data(groupby_cols= ['hhid','Year','Month'])

    household_info_filename = SURVEY_DATA_DIR + '/S0A Household Identification Information.dta'
    hdf = pd.read_stata(household_info_filename,convert_categoricals=False)

    hh_villages = hdf.loc[hdf.location != '',:].groupby(['hhid','location']).size().reset_index(name='N')
    ldf = ldf.merge(hh_villages,left_on='hhid',right_on='hhid')
    ldf.rename(columns={'location':'Location'},inplace=True)

    panel_geo_cw = pd.read_csv(panel_admin_cw_filename)
    panel_geo_cw['Location'] = panel_geo_cw['Location'].str.upper()
    ldf = ldf.merge(panel_geo_cw,left_on='Location',right_on='Location')

    ldf = add_seasons(ldf)
    hdf = get_herd_size()
    merge_cols = ['Year','Month','hhid']
    ldf = ldf.merge(hdf,left_on=merge_cols,right_on=merge_cols)

    groupby_cols = ['hhid','Cluster','NDVILocation','Season']
    herd_sizes = ldf.groupby(groupby_cols)['HerdSize'].max().reset_index()
    losses = ldf.groupby(groupby_cols)['Losses'].sum().reset_index()
    hhdf = herd_sizes.merge(losses,left_on=groupby_cols,right_on=groupby_cols)
    hhdf.rename(columns={'Losses':'LivestockLosses'},inplace=True)
    return hhdf

def get_mortality_rates():
    ldf = load_loss_data(groupby_cols= ['hhid','Year','Month'])

    household_info_filename = SURVEY_DATA_DIR + '/S0A Household Identification Information.dta'
    hdf = pd.read_stata(household_info_filename,convert_categoricals=False)

    hh_villages = hdf.loc[hdf.location != '',:].groupby(['hhid','location']).size().reset_index(name='N')
    ldf = ldf.merge(hh_villages,left_on='hhid',right_on='hhid')
    ldf.rename(columns={'location':'Location'},inplace=True)

    panel_geo_cw = pd.read_csv(panel_admin_cw_filename)
    panel_geo_cw['Location'] = panel_geo_cw['Location'].str.upper()
    ldf = ldf.merge(panel_geo_cw,left_on='Location',right_on='Location')

    ldf = add_seasons(ldf)
    hdf = get_herd_size()
    merge_cols = ['Year','Month','hhid']
    ldf = ldf.merge(hdf,left_on=merge_cols,right_on=merge_cols)

    groupby_cols = ['Cluster','NDVILocation','Location','Season','Month']
    average_monthly_mortality = ldf.groupby(groupby_cols)['Losses'].mean().reset_index()
    average_monthly_herd_size = ldf.groupby(groupby_cols)['HerdSize'].mean().reset_index()

    merge_cols = ['Cluster','NDVILocation','Location','Season']
    seasonal_mortality = average_monthly_mortality.groupby(merge_cols)['Losses'].sum().reset_index()
    seasonal_max_herd_size = average_monthly_herd_size.groupby(merge_cols)['HerdSize'].max().reset_index()
    sdf = seasonal_mortality.merge(seasonal_max_herd_size,left_on=merge_cols,right_on=merge_cols)
    sdf['MortalityRate'] = sdf['Losses']/sdf['HerdSize']
    sdf.replace([np.inf,-np.inf],np.nan,inplace=True)
    return sdf.loc[sdf.MortalityRate.notna(),:]

def make_regression_data():
    df = pd.read_csv(NDVI_covariate_data)
    df.rename(columns={'Location':'NDVILocation'},inplace=True)
    ldf = get_mortality_rates()
    merge_cols = ['NDVILocation','Season']
    df = df.merge(ldf,left_on=merge_cols,right_on=merge_cols)
    df['SRSD'] = df.Season.str.contains('SRSD')
    df['Good Regime'] = df.PostNDVI >= 0
    df = df.loc[~df.SRSD,:]
    df = df.loc[~df.NDVILocation.isin(['Loiyangalani','Karare']),:]
    return df

reg_data = make_regression_data()
reg_data.to_csv(kenya_reg_data_filename,index=False)

hh_data = make_household_data()
hh_data.to_csv(kenya_hh_data_filename,index=False)
