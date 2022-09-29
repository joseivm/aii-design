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
import datetime

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

RAW_DATA_DIR = PROJECT_DIR + '/data/raw'
CLEAN_DATA_DIR = PROJECT_DIR + '/data/processed/NDVI'
coord_filename = CLEAN_DATA_DIR + '/coords.npy'
sinusoidal_projection = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

filename = CLEAN_DATA_DIR + '/2010-01-17.hdf5'

# 1. Read in data, get NDVI part 2. subset that based on coordinates
# 3. save it with relevant attributes, without transforming yet
# 4. save resulting thing

def load_ndvi_data(filename):
    f = h5py.File(filename,'r')
    ndvi = f['NDVI']
    data = get_ndvi_data(ndvi)
    coords = np.load(coord_filename)

    ndvi_points = gpd.GeoSeries(map(Point,coords.reshape(-1,2)))
    ndf = gpd.GeoDataFrame({'geometry':ndvi_points,'NDVI':data.flatten()})
    ndf.crs = pyproj.CRS.from_epsg(4326).to_wkt()
    kdf = load_kenya_geodata()

    df = kdf.sjoin(ndf,how='inner')
    df.rename(columns={'index_right':'PID'},inplace=True)

    file_date = filename[-15:-5]
    df['Date'] = datetime.date.fromisoformat(file_date)
    df['Dekad'] = df['Date'].dt.dayofyear//10+1


def add_seasons(df):
    season_dict = {'LRLD':['03-01','09-30'],'SRSD':['10-01','02-28'],'LRLD Pre':['10-01','02-28'],
      'LRLD Full':['10-01','09-30'],'SRSD Pre':['03-01','09-30'],'SRSD Full':['03-01','02-28']}
    
    season_col = {'LRLD':'Season','SRSD':'Season','LRLD Pre':'Preseason','SRSD Pre':'Preseason',
        'LRLD Full':'Full Season','SRSD Full':'Full Season'}

    years = df['Year'].unique()
    for year in years:
        for season, dates in season_dict.keys():
            season_start_year = year-1 if ('Pre' in season) or ('Full' in season) else year
            start_date, end_date = dates
            start_date = str(season_start_year)+'-'+start_date 
            start_date = datetime.date.fromisoformat(start_date)

            end_date = str(year)+'-'+end_date 
            end_date = datetime.date.fromisoformat(end_date)

            season_name = season + ' ' + year
            col = season_col[season]
            df.loc[(df.Date >= start_date) & (df.Date <= end_date),col] = season_name 
    return df

def load_ndvi_data(filename):
    f = h5py.File(filename,'r')
    ndvi = f['NDVI']
    data, coords = get_ndvi_data(ndvi)

    kdf = load_kenya_geodata()
    bounding_square = kdf.unary_union.envelope
    min_x, min_y, max_x, max_y = bounding_square.bounds

    x_mask = (coords[:,:,0] >= min_x) & (coords[:,:,0] <= max_x)
    y_mask = (coords[:,:,1] >= min_y) & (coords[:,:,1] <= max_y)
    ndvi_mask = x_mask & y_mask
    relevant_indices = np.transpose(np.nonzero(ndvi_mask))
    min_x_idx = min(relevant_indices[:,0])
    max_x_idx = max(relevant_indices[:,0])+1
    min_y_idx = min(relevant_indices[:,1])
    max_y_idx = max(relevant_indices[:,1])+1

    relevant_coords = coords[min_x_idx:max_x_idx,min_y_idx:max_y_idx]
    relevant_data = data[min_x_idx:max_x_idx,min_y_idx:max_y_idx]



    top_left = coords[0,0,:]
    top_right = coords[0,-1,:]
    bottom_left = coords[-1,0,:]
    bottom_right = coords[-1,-1,:]
    square = Polygon([top_left,top_right,bottom_right,bottom_left])

    ndvi_square = gpd.GeoDataFrame()
    ndvi_square['geometry'] = None
    ndvi_square.at[0,'geometry'] = square

    kdf = load_kenya_geodata()
    bounding_square = kdf.unary_union.envelope
    bounding_square = gpd.GeoDataFrame({'geometry':bounding_square,'name':['bounding square']})
    min_x, min_y, max_x, max_y = bounding_square.bounds

    x_mask = (coords[:,:,0] >= min_x) & (coords[:,:,0] <= max_x)
    y_mask = (coords[:,:,1] >= min_y) & (coords[:,:,1] <= max_y)
    ndvi_mask = x_mask & y_mask
    relevant_indices = np.transpose(np.nonzero(ndvi_mask))
    min_x_idx = min(relevant_indices[:,0])
    max_x_idx = max(relevant_indices[:,0])+1
    min_y_idx = min(relevant_indices[:,1])
    max_y_idx = max(relevant_indices[:,1])+1

    relevant_coords = coords[min_x_idx:max_x_idx,min_y_idx:max_y_idx]
    relevant_data = data[min_x_idx:max_x_idx,min_y_idx:max_y_idx]

    ndvi_points = gpd.GeoSeries(map(Point,relevant_coords.reshape(-1,2)))
    ndf = gpd.GeoDataFrame({'geometry':ndvi_points,'NDVI':relevant_data.flatten()})

    fig, ax = plt.subplots()
    ndvi_square.plot(ax=ax,facecolor='blue')
    kdf.loc[kdf.NAME_1 == 'Marsabit',:].plot(ax=ax,facecolor='red')
    plt.show()

    archive_metadata = f['NDVI'].attrs['ArchiveMetadata']
    core_metadata = f['NDVI'].attrs['CoreMetadata']
    struct_metadata = f['NDVI'].attrs['StructMetadata']

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

def load_kenya_geodata():
    filename = RAW_DATA_DIR + '/Kenya Geodata/gadm41_KEN_3.shp'
    kdf = gpd.read_file(filename)
    locations_of_interest = ['Marsabit Central','Sagante/Jaldesa','Laisamis','Loiyangalani','Maikona']
    # kdf.loc[kdf.NAME_1 == 'Marsabit',:].apply(lambda x: ax.annotate(text=x['NAME_3'],
    #     xy=x.geometry.centroid.coords[0], ha='center'), axis=1)
    return kdf.loc[kdf.NAME_1 == 'Marsabit',:]
   
hdf_files = [fname for fname in os.listdir(CLEAN_DATA_DIR) if '.hdf5' in fname]
hdf_files = [os.path.join(CLEAN_DATA_DIR ,fname) for fname in hdf_files]

