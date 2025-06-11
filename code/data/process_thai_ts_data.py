import pandas as pd
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

# Input files/dirs
THAI_DATA_DIR = os.path.join(PROJECT_DIR,'data','raw','Thailand')
PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR,'data','processed')


##### Data loading #####

# We'll start off with a nationwide model. It might make more sense to start off with regional models, since
# that's the standard in the literature. 
# TODO: test and re run

def load_loss_data():
    fpath = os.path.join(PROCESSED_DATA_DIR,'Thailand_loss_data.csv')
    df = pd.read_csv(fpath)
    df['PlantDate'] = pd.to_datetime(df['PlantDate'])
    df['HarvestDate'] = pd.to_datetime(df['HarvestDate'])
    return df

def load_chirps():
    chirps_files = [f for f in os.listdir(THAI_DATA_DIR) if 'CHIRPS' in f]
    dfs = []
    for f in chirps_files:
        fpath = os.path.join(THAI_DATA_DIR,f)
        df = pd.read_csv(fpath)
        rain_cols = [col for col in df.columns if 'precipitation' in col]
        df = pd.melt(df,id_vars=['ADM3_PCODE'], value_vars = rain_cols, 
                        var_name='Date', value_name='Precipitation')
        df['Date'] = df.Date.str.slice(stop=8)
        df['Date'] = pd.to_datetime(df['Date'],format="%Y%m%d")
        dfs.append(df)

    df = pd.concat(dfs)
    df['TCode'] = 'T' + df['ADM3_PCODE'].str.slice(start=2)
    df.drop(columns=['ADM3_PCODE'],inplace=True)
    return df.loc[df.Precipitation.notna(),:]

def load_etp():
    etp_files = [f for f in os.listdir(THAI_DATA_DIR) if 'ETP' in f]
    dfs = []
    for f in etp_files:
        fpath = os.path.join(THAI_DATA_DIR,f)
        df = pd.read_csv(fpath)
        evap_cols = [col for col in df.columns if 'Evap' in col]
        df = pd.melt(df,id_vars=['ADM3_PCODE'], value_vars = evap_cols, 
                        var_name='Date', value_name='ETP')
        df['Date'] = df.Date.str.slice(stop=6)
        df['Date'] = pd.to_datetime(df['Date'],format="%Y%m")
        dfs.append(df)

    df = pd.concat(dfs)
    df['TCode'] = 'T' + df['ADM3_PCODE'].str.slice(start=2)
    df.drop(columns=['ADM3_PCODE'],inplace=True)
    return df.loc[df.ETP.notna(),:]

def load_lst():
    etp_files = [f for f in os.listdir(THAI_DATA_DIR) if 'LST' in f]
    dfs = []
    for f in etp_files:
        fpath = os.path.join(THAI_DATA_DIR,f)
        df = pd.read_csv(fpath)
        temp_cols = [col for col in df.columns if 'LST' in col]
        df = pd.melt(df,id_vars=['ADM3_PCODE'], value_vars = temp_cols, 
                        var_name='Date', value_name='Temperature')
        df['Date'] = df.Date.str.slice(stop=10)
        df['Date'] = pd.to_datetime(df['Date'],format="%Y_%m_%d")
        dfs.append(df)

    df = pd.concat(dfs)
    df['TCode'] = 'T' + df['ADM3_PCODE'].str.slice(start=2)
    df.drop(columns=['ADM3_PCODE'],inplace=True)
    return df.loc[df.Temperature.notna(),:]

def remove_geo_data():
    col_dict = {'CHIRPS':'precipitation','LST':'LST_1KM', 'ETP':'Evap_tavg'}
    years = ['2015_2016','2017_2018','2019_2020','2021_2022']
    for data_source, col_name in col_dict.items():
        for year in years:
            fname = os.path.join(RS_DATA_DIR,f"Thailand_{year}_{data_source}.csv")
            df = pd.read_csv(fname)
            cols = [col for col in df.columns if col_name in col]
            cols += ['ADM3_PCODE']
            outpath = os.path.join(RS_DATA_DIR,f"{data_source}_{year}.csv")
            df[cols].to_csv(outpath,index=False)
    
##### Process Data #####
def make_processed_loss_data():
    fpath = os.path.join(THAI_DATA_DIR,'Tambon_loss_data.csv')
    df = pd.read_csv(fpath)
    df['PlantDate'] = pd.to_datetime(df['median_plant_date'])
    df['PlantDate'] -= pd.DateOffset(days=14)
    df['HarvestDate'] = df.PlantDate + pd.DateOffset(days=194)
    df['ProvinceCode'] = df.Tambon.apply(lambda x: x.split('-')[0])
    df['AmphurCode'] = df.Tambon.apply(lambda x: x.split('-')[1])
    df['TambonCode'] = df.Tambon.apply(lambda x: x.split('-')[2])
    df.loc[df.AmphurCode.str.len() < 2, 'AmphurCode'] = '0' + df.loc[df.AmphurCode.str.len() < 2, 'AmphurCode']
    df.loc[df.TambonCode.str.len() < 2, 'TambonCode'] = '0' + df.loc[df.TambonCode.str.len() < 2, 'TambonCode']
    df['TCode'] = 'T' + df.ProvinceCode + df.AmphurCode + df.TambonCode
    df['ObsID'] = df['TCode'] + '-' + df.Year.astype(str)
    df['SuperZone'] = df['Zone'].str.slice(stop=-1)
    df = add_loss_class(df, 5, 'LossClass')
    df = add_loss_class(df, 3,'LossGroup')
    cols = ['Zone','SuperZone','Year','Loss','LossClass','LossGroup','WeightedLoss','WeightSum','PlantDate','HarvestDate','ObsID','TCode']
    outpath = os.path.join(PROCESSED_DATA_DIR,'Thailand_loss_data.csv')
    df.loc[:,cols].to_csv(outpath,index=False)

def add_loss_class(df,n_classes,col_name='LossClass'):
    eps = 0.0001
    bins = [0] + [eps + x*(1-eps)/n_classes for x in range(n_classes+1)]
    bins[-1] = 1
    df[col_name] = pd.cut(df.Loss,bins=bins,labels=False,duplicates='drop',include_lowest=True)
    return df

def make_time_series_data():
    ldf = load_loss_data()

    cdf = load_chirps()
    etp = load_etp()
    lst = load_lst()

    years = np.arange(2015,2023)
    ydfs = []
    for yr in years:
        ydf = ldf.loc[ldf.Year == yr,:]

        ycdf = adjust_dates(cdf,ydf)
        yetp = adjust_dates(etp,ydf)
        ylst = adjust_dates(lst,ydf)

        cdfs = group_by_ts_days(ycdf)
        etps = group_by_ts_days(yetp)
        lsts = group_by_ts_days(ylst)

        # Stretch out shorter time series so they are all the same length
        all_dfs = cdfs + lsts + etps
        max_length = np.max([df.shape[1] for df in all_dfs])

        cdfs = [uniform_scaling(tdf,max_length) for tdf in cdfs]
        lsts = [uniform_scaling(tdf,max_length) for tdf in lsts]
        etps = [uniform_scaling(tdf,max_length) for tdf in etps]

        ycdf = pd.concat(cdfs).reset_index()
        ylst = pd.concat(lsts).reset_index()
        yetp = pd.concat(etps).reset_index()

        ycdf['Variable'] = 'Precipitation'
        ylst['Variable'] = 'Temperature'
        yetp['Variable'] = 'Evap'

        df = pd.concat([ycdf,ylst,yetp])
        df['Year'] = yr
        ydfs.append(df)

    df = pd.concat(ydfs)
    fpath = os.path.join(PROJECT_DIR,'data','processed','Thailand_ts_data.csv')
    df.to_csv(fpath,index=False)

def group_by_ts_days(df):
    value_cols = ['Temperature','ETP','Precipitation']
    value_col = [col for col in df.columns if col in value_cols][0]
    date_df = df.groupby('TCode')['Time'].unique().reset_index(name='SetOfDays')
    date_df['SetOfDays'] = date_df['SetOfDays'].apply(lambda x: str(np.sort(x)))
    unique_sets = date_df['SetOfDays'].unique()
    same_set_dfs = []
    for day_set in unique_sets:
        set_tcodes = date_df.loc[date_df['SetOfDays'] == day_set,'TCode']
        set_df = df.loc[df.TCode.isin(set_tcodes),:]
        set_df = pd.pivot(set_df,index='TCode',columns='Time',values=value_col).reset_index()
        set_df.set_index('TCode',inplace=True)
        same_set_dfs.append(set_df)
    return same_set_dfs

def adjust_dates(df,ldf):
    cols = [col for col in df.columns] + ['Time']
    df = df.merge(ldf,on='TCode')
    df = df.loc[(df.Date >= df.PlantDate) & (df.Date <= df.HarvestDate),:]
    df['Time'] = (df['Date'] - df['PlantDate'])/np.timedelta64(1,'D') + 1
    df['Time'] = df.Time.astype(int)
    return df.loc[:,cols]

def uniform_scaling(df, desired_length):
    data = df.to_numpy()
    data_length = data.shape[1]
    scaling_indexes  = np.arange(1,desired_length+1)*(data_length/desired_length)
    scaling_indexes = np.round(scaling_indexes,2)
    scaling_indexes = np.ceil(scaling_indexes).astype(int)
    scaling_indexes -= 1
    stretched_data = data[:, scaling_indexes]
    new_cols = [str(i) for i in range(1,desired_length + 1)]
    new_df = pd.DataFrame(data=stretched_data,columns=new_cols,index=df.index)
    return new_df

make_processed_loss_data()
make_time_series_data()