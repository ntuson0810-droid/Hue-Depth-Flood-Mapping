import os
import re

core_utils_code = """
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from tqdm import tqdm

ESA_MANNING_LOOKUP = {
    10:  0.120, 20:  0.060, 30:  0.035, 40:  0.040, 50:  0.020,
    60:  0.025, 70:  0.012, 80:  0.030, 90:  0.060, 95:  0.100, 100: 0.025,
}
DEFAULT_MANNING = 0.040
NODATA_VAL = -9999

def lulc_to_manning(lulc_array):
    manning = np.full(lulc_array.shape, DEFAULT_MANNING, dtype=np.float32)
    for cls, n_val in ESA_MANNING_LOOKUP.items():
        manning[lulc_array == cls] = n_val
    return manning

def create_manning_raster_from_lulc(lulc_path, output_path):
    if os.path.exists(output_path):
        return
    with rasterio.open(lulc_path) as src:
        lulc = src.read(1)
        meta = src.meta.copy()
    manning = lulc_to_manning(lulc)
    meta.update(dtype='float32', nodata=-9999, compress='lzw', count=1)
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(manning, 1)

def create_interaction_features(data_dict, eps=1e-6):
    feats = []
    names = []
    def add(name, arr):
        feats.append(arr.astype(np.float32))
        names.append(name)

    if 'dsm_bathy' in data_dict and 'distance_r' in data_dict: add('dem_x_dist', data_dict['dsm_bathy'] * data_dict['distance_r'])
    if 'dsm_bathy' in data_dict and 'slope' in data_dict: add('dem_x_slope', data_dict['dsm_bathy'] * data_dict['slope'])
    if 'dsm_bathy' in data_dict and 'twi' in data_dict: add('dem_x_twi', data_dict['dsm_bathy'] * data_dict['twi'])
    if 'precip_s_1' in data_dict and 'slope' in data_dict: add('rain_x_slope', data_dict['precip_s_1'] * data_dict['slope'])
    if 'precip_s_1' in data_dict and 'twi' in data_dict: add('rain_x_twi', data_dict['precip_s_1'] * data_dict['twi'])
    if 'flow_accum' in data_dict and 'slope' in data_dict: add('flow_x_slope', data_dict['flow_accum'] * data_dict['slope'])
    if 'distance_r' in data_dict and 'twi' in data_dict: add('dist_x_twi', data_dict['distance_r'] * data_dict['twi'])
    if 'precip_s_1' in data_dict and 'flow_accum' in data_dict: add('rain_x_flow', data_dict['precip_s_1'] * data_dict['flow_accum'])
    if 'dsm_bathy' in data_dict: add('dem_squared', data_dict['dsm_bathy'] ** 2)
    if 'distance_r' in data_dict: add('dist_squared', data_dict['distance_r'] ** 2)

    if 'manning_n' in data_dict and 'slope' in data_dict:
        slope_rad = np.radians(np.clip(data_dict['slope'], 0, 89))
        slope_grade = np.tan(slope_rad)
        n_safe = np.maximum(data_dict['manning_n'], 0.005)
        add('conveyance', (1.0 / n_safe) * np.sqrt(np.maximum(slope_grade, eps)))
        add('resistance', np.clip(n_safe / np.sqrt(np.maximum(slope_grade, eps)), 0, 100))
    if 'manning_n' in data_dict and 'hand30_100' in data_dict: add('manning_x_hand', data_dict['manning_n'] * data_dict['hand30_100'])
    if 'manning_n' in data_dict and 'flow_accum' in data_dict: add('manning_x_flow', data_dict['manning_n'] * data_dict['flow_accum'])
    return feats, names

def extract_values_at_points(tif_path, lons, lats):
    with rasterio.open(tif_path) as src:
        coords = [(lon, lat) for lon, lat in zip(lons, lats)]
        return np.array([val[0] for val in src.sample(coords)])

def clip_raster_by_shapefile(input_tif, shapefile_path, output_tif):
    import rasterio.mask
    gdf = gpd.read_file(shapefile_path)
    with rasterio.open(input_tif) as src:
        if gdf.crs != src.crs: gdf = gdf.to_crs(src.crs)
        geom = [shapes['geometry'] for _, shapes in gdf.iterrows()]
        out_image, out_transform = rasterio.mask.mask(src, geom, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})
    with rasterio.open(output_tif, "w", **out_meta) as dest:
        dest.write(out_image)

def classify_flood_depth(depth):
    cls = np.zeros_like(depth, dtype=np.uint8)
    cls[(depth > 0.001) & (depth <= 0.5)] = 1
    cls[(depth > 0.5) & (depth <= 1.0)] = 2
    cls[(depth > 1.0) & (depth <= 1.5)] = 3
    cls[(depth > 1.5) & (depth <= 2.0)] = 4
    cls[depth > 2.0] = 5
    return cls

def create_flood_classification_map(depth_raster_path, output_classified_path, output_legend_path=None):
    with rasterio.open(depth_raster_path) as src:
        depth = src.read(1)
        meta = src.meta.copy()
    cls = classify_flood_depth(depth)
    cls[depth == NODATA_VAL] = 255
    meta.update(dtype='uint8', nodata=255)
    with rasterio.open(output_classified_path, 'w', **meta) as dst:
        dst.write(cls, 1)
"""

os.makedirs('src/core', exist_ok=True)
with open('src/core/__init__.py', 'w') as f: f.write('')
with open('src/core/utils.py', 'w', encoding='utf-8') as f: f.write(core_utils_code)
print("Created src/core/utils.py")
