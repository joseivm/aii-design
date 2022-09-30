import pandas as pd
import numpy as np
import pyhdf.SD as ph
import re
import os
import h5py
import geopandas as gpd
import pyproj
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import time

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

# Input files/dirs
RAW_DATA_DIR = PROJECT_DIR + '/data/raw'
CLEAN_DATA_DIR = PROJECT_DIR + '/data/processed/NDVI'
coord_filename = CLEAN_DATA_DIR + '/coords.npy'
village_cw_filename = CLEAN_DATA_DIR + '/village_pixel_cw.csv'

# Output files/dirs
covariate_matrix_filename = CLEAN_DATA_DIR + '/covariate_matrix.csv'

# 1. Read in data, get NDVI part 2. subset that based on coordinates
# 3. save it with relevant attributes, without transforming yet
# 4. save resulting thing

##### Data loading #####
def load_ndvi_data():
    NDVI_DATA_DIR = CLEAN_DATA_DIR + '/HDF'
    hdf_files = [fname for fname in os.listdir(NDVI_DATA_DIR) if '.hdf5' in fname]
    hdf_files = [os.path.join(NDVI_DATA_DIR ,fname) for fname in hdf_files]
    ndvi_dfs = [process_ndvi_file(fname) for fname in hdf_files]
    ndf = pd.concat(ndvi_dfs)
    return ndf

def load_kenya_geodata():
    filename = RAW_DATA_DIR + '/Kenya Geodata/gadm41_KEN_3.shp'
    kdf = gpd.read_file(filename)
    locations_of_interest = ['Marsabit Central','Sagante/Jaldesa','Laisamis','Loiyangalani','Maikona']
    # kdf.loc[kdf.NAME_1 == 'Marsabit',:].apply(lambda x: ax.annotate(text=x['NAME_3'],
    #     xy=x.geometry.centroid.coords[0], ha='center'), axis=1)
    kdf.rename(columns={'NAME_3':'Location'},inplace=True)
    return kdf.loc[kdf.NAME_1 == 'Marsabit',['Location','geometry']]

def process_ndvi_file(filename):
    print(filename)
    f = h5py.File(filename,'r')
    ndvi = f['NDVI']
    data = get_ndvi_data(ndvi)

    ndf = pd.DataFrame({'NDVI':data.flatten()})
    ndf = ndf.reset_index().rename(columns={'index':'PID'})
    
    vdf = pd.read_csv(village_cw_filename)

    df = vdf.merge(ndf,left_on='PID',right_on='PID')

    file_date = filename[-15:-5]
    df['Date'] = pd.to_datetime(file_date,format='%Y-%m-%d')
    df['Year'] = df['Date'].dt.year
    df['Dekad'] = df['Date'].dt.dayofyear//10+1
    df = add_seasons(df)
    return df    

def get_ndvi_data(dataset):
    data = dataset[:,:].astype(float)
    attributes = dataset.attrs
    valid_range = attributes['valid_range']
    fill_value = attributes['_FillValue']
    scale_factor = attributes['scale_factor']
    add_offset = attributes['add_offset']

    invalid = np.logical_or(data < valid_range[0], data > valid_range[1])
    invalid = np.logical_or(invalid, data == fill_value)
    data[invalid] = np.nan
    data = (data - add_offset) * scale_factor
    data = np.ma.masked_array(data, np.isnan(data))
    return data

def add_seasons(df):
    season_dict = {'LRLD':['03-01','09-30'],'SRSD':['10-01','02-28'],'LRLD Pre':['10-01','02-28'],
      'LRLD Full':['10-01','09-30'],'SRSD Pre':['03-01','09-30'],'SRSD Full':['03-01','02-28']}
    
    season_col = {'LRLD':'Season','SRSD':'Season','LRLD Pre':'Preseason','SRSD Pre':'Preseason',
        'LRLD Full':'Full Season','SRSD Full':'Full Season'}

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
   
def create_village_pixel_mapping():
    NDVI_DATA_DIR = CLEAN_DATA_DIR + '/HDF'
    hdf_files = [fname for fname in os.listdir(NDVI_DATA_DIR) if '.hdf5' in fname]
    hdf_files = [os.path.join(NDVI_DATA_DIR ,fname) for fname in hdf_files]

    fname = hdf_files[0]
    f = h5py.File(fname,'r')
    ndvi = f['NDVI']
    data = get_ndvi_data(ndvi)
    coords = np.load(coord_filename)

    ndvi_points = gpd.GeoSeries(map(Point,coords.reshape(-1,2)))
    ndf = gpd.GeoDataFrame({'geometry':ndvi_points,'NDVI':data.flatten()})
    ndf.crs = pyproj.CRS.from_epsg(4326).to_wkt()
    kdf = load_kenya_geodata()

    df = kdf.sjoin(ndf,how='inner')
    df.rename(columns={'index_right':'PID'},inplace=True)
    df[['Location','PID']].to_csv(village_cw_filename,index=False)

##### Covariate Matrix #####
def create_covariate_matrix():
    ndf = load_ndvi_data()
    pdf = ndf.groupby(['PID','Dekad'])['NDVI'].agg(PixelMean='mean',PixelStd='std').reset_index()
    ndf = ndf.merge(pdf,left_on=['PID','Dekad'],right_on=['PID','Dekad'])
    ndf['ZNDVI'] = (ndf['NDVI']-ndf['PixelMean'])/ndf['PixelStd']
    ndf['NegZNDVI'] = np.abs(np.minimum(ndf['ZNDVI'],0))
    ndf['PosZNDVI'] = np.maximum(ndf['ZNDVI'],0)
    pos_ndvi = ndf[ndf.ZNDVI.notna()].groupby(['Full Season','Location'])['PosZNDVI'].sum().reset_index(name='PosZNDVI')
    neg_ndvi = ndf[ndf.ZNDVI.notna()].groupby(['Full Season','Location'])['NegZNDVI'].sum().reset_index(name='NegZNDVI')
    pre_ndvi = ndf[ndf.ZNDVI.notna()].groupby(['Preseason','Location'])['ZNDVI'].sum().reset_index(name='PreNDVI')

    cov = ndf.groupby(['Location','Season','Full Season','Preseason']).size().reset_index(name='N')
    cov = cov.merge(pos_ndvi,left_on=['Location','Full Season'],right_on=['Location','Full Season'])
    cov = cov.merge(neg_ndvi,left_on=['Location','Full Season'],right_on=['Location','Full Season'])
    cov = cov.merge(pre_ndvi,left_on=['Location','Preseason'],right_on=['Location','Preseason'])
    cov.drop(columns=['N'],inplace=True)
    return cov

cov = create_covariate_matrix()
cov.to_csv(covariate_matrix_filename,index=False)

