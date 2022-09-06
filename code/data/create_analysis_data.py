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

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

RAW_DATA_DIR = PROJECT_DIR + '/data/raw'
CLEAN_DATA_DIR = PROJECT_DIR + '/data/processed/NDVI/HDF'
sinusoidal_projection = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

filename = CLEAN_DATA_DIR + '/2010-01-17.hdf5'

# 1. Read in data, get NDVI part 2. subset that based on coordinates
# 3. save it with relevant attributes, without transforming yet
# 4. save resulting thing
# 
# 
# 

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
    units = attributes['units']
    add_offset = attributes['add_offset']

    invalid = np.logical_or(data < valid_range[0], data > valid_range[1])
    invalid = np.logical_or(invalid, data == fill_value)
    data[invalid] = np.nan
    data = (data - add_offset) * scale_factor
    data = np.ma.masked_array(data, np.isnan(data))

    grid_metadata = attributes['StructMetadata']
    ul_regex = re.compile(r'''UpperLeftPointMtrs=\(
                            (?P<upper_left_x>[+-]?\d+\.\d+)
                            ,
                            (?P<upper_left_y>[+-]?\d+\.\d+)
                            \)''', re.VERBOSE)

    match = ul_regex.search(grid_metadata)
    x0 = float(match.group('upper_left_x'))
    y0 = float(match.group('upper_left_y'))

    lr_regex = re.compile(r'''LowerRightMtrs=\(
                            (?P<lower_right_x>[+-]?\d+\.\d+)
                            ,
                            (?P<lower_right_y>[+-]?\d+\.\d+)
                            \)''', re.VERBOSE)
    match = lr_regex.search(grid_metadata)
    x1 = float(match.group('lower_right_x'))
    y1 = float(match.group('lower_right_y'))
            
    nx, ny = data.shape
    x = np.linspace(x0, x1, nx, endpoint=False)
    y = np.linspace(y0, y1, ny, endpoint=False)
    xv, yv = np.meshgrid(x, y)    

    sinu = pyproj.Proj("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
    wgs84 = pyproj.Proj("EPSG:4326") 
    transformer = pyproj.Transformer.from_proj(sinu,wgs84)
    lon, lat= transformer.transform(xv, yv)
    coords = np.vstack(([lat.T],[lon.T])).T
    return data, coords

def load_kenya_geodata():
    filename = RAW_DATA_DIR + '/Kenya Geodata/gadm41_KEN_3.shp'
    kdf = gpd.read_file(filename)
    locations_of_interest = ['Marsabit Central','Sagante/Jaldesa','Laisamis','Loiyangalani','Maikona']
    # kdf.loc[kdf.NAME_1 == 'Marsabit',:].apply(lambda x: ax.annotate(text=x['NAME_3'],
        # xy=x.geometry.centroid.coords[0], ha='center'), axis=1)
    return kdf.loc[kdf.NAME_1 == 'Marsabit',:]
   
hdf_files = [fname for fname in os.listdir(CLEAN_DATA_DIR) if '.hdf5' in fname]
hdf_files = [os.path.join(CLEAN_DATA_DIR ,fname) for fname in hdf_files]

