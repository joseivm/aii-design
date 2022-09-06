import pandas as pd
import numpy as np
import pyhdf.SD as ph
import re
import os
import h5py
import pyproj
import geopandas as gpd

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

RAW_DATA_DIR = PROJECT_DIR + '/data/raw'
MODIS_RAW_DATA_DIR = RAW_DATA_DIR + '/MODIS'
MODIS_CLEAN_DATA_DIR = PROJECT_DIR + '/data/processed/NDVI'

class MetaDataParser:
    def __init__(self):
        self.object_searching = True
        self.current_object = ''
        self.object_regex = re.compile("^\s*OBJECT\s*")
        self.value_regex = re.compile("^\s*VALUE\s*")
        self.end_object_regex = re.compile("^\s*END_OBJECT\s*")
        self.metadata_dict = {}

    def has_object(self,line):
        object_match = self.object_regex.search(line)
        return object_match is not None

    def has_value(self,line):
        value_match = self.value_regex.search(line)
        return value_match is not None

    def has_end_object(self,line):
        end_object_match = self.end_object_regex(line)
        return end_object_match is not None

    def get_object(self,line):
        if self.has_object(line):
            obj = line.split('=')[1]
            obj = obj.strip()
        else:
            obj = ''
        return obj

    def get_value(self,line):
        if self.has_value(line):
            value = line.split('=')[1]
            value = value.strip()
        else:
            value = ''
        return value

    def add_metadata(self,line):
        self.metadata_dict[self.current_object] = self.get_value(line)

    def parse(self,lines):
        for line in lines:
            if self.has_object(line):
                self.current_object = self.get_object(line)
            elif self.has_value(line):
                self.add_metadata(line)

        return self.metadata_dict

def process_modis_file(filename):
    hdf = ph.SD(filename,ph.SDC.READ)
    ndvi_dataset = hdf.select('250m 16 days NDVI')
    data, coords = get_data_and_coordinates(hdf)
    data, coords = subset_ndvi_data(data,coords)

    hdf_attr = hdf.attributes()
    core_metadata = hdf_attr['CoreMetadata.0'].strip('\x00')
    archive_metadata = hdf_attr['ArchiveMetadata.0'].strip('\x00')
    struct_metadata = hdf_attr['StructMetadata.0'].strip('\x00')
    mdp = MetaDataParser()
    core_metadata_dict = mdp.parse(core_metadata.split('\n'))

    date = core_metadata_dict['RANGEBEGINNINGDATE'].strip('"')

    new_filename = MODIS_CLEAN_DATA_DIR + '/HDF/'+ date+'.hdf5'
    new_file = h5py.File(new_filename,'w')
    dataset = new_file.create_dataset('NDVI',data=data)
    dataset.attrs['CoreMetaData'] = core_metadata
    dataset.attrs['ArchiveMetadata'] = archive_metadata
    dataset.attrs['StructMetadata'] = struct_metadata
    for key, item in ndvi_dataset.attributes().items():
        dataset.attrs[key] = item
    new_file.close()

def save_coordinates(filename):
    hdf = ph.SD(filename,ph.SDC.READ)
    data, coords = get_data_and_coordinates(hdf)
    data, coords = subset_ndvi_data(data,coords)
    outfile = MODIS_CLEAN_DATA_DIR + '/coords.npy'
    np.save(outfile,coords)

def get_data_and_coordinates(hdf):
    data = hdf.select('250m 16 days NDVI').get()
    attributes = hdf.attributes()
    grid_metadata = attributes['StructMetadata.0']
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

def subset_ndvi_data(data,coords):
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
    return relevant_data, relevant_coords

def load_kenya_geodata():
    filename = RAW_DATA_DIR + '/Kenya Geodata/gadm41_KEN_3.shp'
    kdf = gpd.read_file(filename)
    return kdf.loc[kdf.NAME_1 == 'Marsabit',:]

def main():
    modis_files = [fname for fname in os.listdir(MODIS_RAW_DATA_DIR) if '.hdf' in fname]
    modis_files = [os.path.join(MODIS_RAW_DATA_DIR,fname) for fname in modis_files]
    [process_modis_file(fname) for fname in modis_files]
    save_coordinates(modis_files[0])

main()