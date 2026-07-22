#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SedHydro
---------
A Multi-Fraction Erosion and Sediment Transport 
Framework with Hydrology-Model-Agnostic Coupling.

Developed by the CAPE Team
University of Calgary

Repository:
https://github.com/<organization>/SedHydro

"""

#%% import packages

import os
import sys
import re
import platform
import tomllib

import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio

from netCDF4 import Dataset, num2date
#%% from toml file

# =====================================================
# CONFIG FILE
# =====================================================

toml_file = "/Users/armanhaddadchi/Library/CloudStorage/OneDrive-UniversityofCalgary/ErosionModel/git_SedHydro/directory_settings_Notikenwin_validation.toml"

# if len(sys.argv) > 1:
#     toml_file = sys.argv[1]
# else:
#     toml_file = "directory_settings_Smoky_validation.toml"
# =====================================================
# SMALL HELPERS
# =====================================================

def load_directory_settings(toml_name=toml_file):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    toml_path = os.path.join(script_dir, toml_name)

    with open(toml_path, "rb") as f:
        return tomllib.load(f)


def get_os_suffix():
    system = platform.system().lower()
    if system.startswith("win"):
        return "_win"
    elif system == "linux":
        return "_linux"
    else:
        return "_mac"


def cfg_list_or_none(section, key):
    values = section.get(key, None)
    if values is None:
        return None
    return list(values)


def get_cfg_path(cfg_section, key_base):

    suffix = get_os_suffix()
    key = f"{key_base}{suffix}"

    if key in cfg_section:
        return os.path.normpath(cfg_section[key])

    if key_base in cfg_section:
        return os.path.normpath(cfg_section[key_base])

    raise KeyError(
        f"Missing key '{key}' or fallback key '{key_base}' in TOML."
    )


def safe_chdir(path_to_use):

    if os.path.isdir(path_to_use):
        os.chdir(path_to_use)
    else:
        print(
            f"Warning: working directory not found:"
            f" {path_to_use}"
        )


# =====================================================
# READ CONFIG
# =====================================================

cfg = load_directory_settings()

# =====================================================
# SETTINGS
# =====================================================

model_mode = cfg["settings"].get("model_mode", "calibration").lower()

start_date = cfg["settings"]["start_date"]
end_date = cfg["settings"]["end_date"]

cell_size = cfg["settings"]["cell_size"]

'''
cold_region=True → applies SWE attenuation
cold_region=False → original erosion model
'''
cold_region = cfg["settings"].get(
    "cold_region",
    True
)

# =====================================================
# WORKING DIRECTORY
# =====================================================

working_directory = get_cfg_path(
    cfg["common"],
    "working_directory"
)

safe_chdir(working_directory)

# =====================================================
# MAIN DIRECTORIES
# =====================================================

main_directory = get_cfg_path(
    cfg["common"],
    "main_directory"
)

catchment_folder = cfg["common"]["catchment_folder"]

catchment_path = os.path.join(
    main_directory,
    catchment_folder
)

main_dir_routing = get_cfg_path(
    cfg["common"],
    "main_dir_routing"
)

# =====================================================
# CATCHMENT SETTINGS
# =====================================================

setting_directory = cfg["catchment"]["setting_directory"]

cat_shp_path = cfg["catchment"]["cat_shp_path"]

cat_hru_path = cfg["catchment"]["cat_hru_path"]

dem_path = cfg["catchment"]["dem_path"]

rainfall_directory = cfg["catchment"]["rainfall_directory"]

SUMMA_path = cfg["catchment"]["SUMMA_path"]

mizu_path = cfg["catchment"]["mizu_path"]

river_shp_file = cfg["catchment"]["river_shp_file"]

# =====================================================
# REFERENCE FILES
# =====================================================

landclass_directory = get_cfg_path(cfg["reference"],
    "landclass_directory")

geol_path = get_cfg_path(cfg["reference"],
    "geol_path")

geolclass_directory = get_cfg_path(cfg["reference"],
    "geolclass_directory")

landcover_path = get_cfg_path(cfg["reference"],
    "landcover_path")

sand_path = get_cfg_path(cfg["reference"],
    "sand_path")

silt_path = get_cfg_path(cfg["reference"],
    "silt_path")

# reach_bed_file = cfg["reference"]["reach_bed_file"]

sediment_size_file = cfg["reference"]["sediment_size_file"]

routing_constants_toml = get_cfg_path(cfg["reference"],
    "routing_constants_toml")

# =====================================================
# OBSERVED + COEFFICIENTS
# =====================================================

coefs_path = get_cfg_path(cfg["common"],
    "coefs_path")

coefs = pd.read_csv(coefs_path)
"""
df_Q_SSC_obs[discharge] -- unit: m3/s
df_Q_SSC_obs[SSC] -- unit: g/m3

"""

observed_path = get_cfg_path(cfg["observed"],
    "observed_path")

# =====================================================
# OUTPUT
# =====================================================

output_dir = get_cfg_path(cfg["output"],
    "output_dir")

output_optimisation_file_name1b = cfg["output"].get(
    "output_optimisation_file_name1b",
    "optimised1b_parameters")

output_optimisation_file_name3 = cfg["output"].get(
    "output_optimisation_file_name3",
    "optimised3_combined_parameters")

output_model_sed_final_pkl = cfg["output"].get(
    "output_model_sed_final_pkl",
    "model_sed_final_run1.pkl"
)

# =====================================================
# STORAGE SETTINGS
# =====================================================

use_storage = cfg.get(
    "storage",
    {}
).get("use_storage", False)

storage_data_type = cfg.get(
    "storage",
    {}
).get("storage_data_type","area")

river_storage = None

if use_storage:

    storage_file_dir = get_cfg_path(cfg["storage"],
        "storage_file_dir")

    river_storage = pd.read_csv(storage_file_dir)

# =====================================================
# ADD PATHS
# =====================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

if main_dir_routing not in sys.path:
    sys.path.insert(0, main_dir_routing)


#%% read Validation data
# This section is only activated when model_mode = "validation" in TOML

if model_mode == "validation":

    start_date_validation = cfg["validation"]["start_date_validation"]
    end_date_validation = cfg["validation"]["end_date_validation"]

    validation_coefs_path = get_cfg_path(
        cfg["validation"],
        "validation_coefs_path"
    )

    validation_coefs = pd.read_csv(validation_coefs_path)

    # keep only parameter rows
    validation_coefs = validation_coefs.loc[
        validation_coefs["section"].eq("parameters"),
        ["name", "value", "low", "up"]
    ].copy()

    # rename columns to match coefs dataframe
    validation_coefs = validation_coefs.rename(columns={
        "name": "parameter",
        "value": "default",
        "low": "low_range",
        "up": "up_range"
    })

    # add Description and priority from coefs dataframe
    validation_coefs = validation_coefs.merge(
        coefs[["parameter", "Description", "priority"]],
        on="parameter",
        how="left"
    )

    # same column order as coefs
    validation_coefs = validation_coefs[
        ["parameter", "default", "low_range", "up_range", "Description", "priority"]
    ]

    # saving directory for validation results
    validation_model_sed_final_pkl = cfg["validation"].get(
        "validation_model_sed_final_pkl"
    )

    validation_file_name = cfg["validation"].get(
        "validation_file_name"
    )

else:
    start_date_validation = None
    end_date_validation = None
    validation_coefs = None
    validation_model_sed_final_pkl = None
    validation_file_name = None

# change date to validation period if model_mode="validation"
if model_mode == "validation":
    start_date = cfg["validation"]["start_date_validation"]
    end_date = cfg["validation"]["end_date_validation"]


if model_mode == "validation":
    final_file_name3 = validation_file_name
    final_model_sed_final_pkl = validation_model_sed_final_pkl
else:
    final_file_name3 = output_optimisation_file_name3
    final_model_sed_final_pkl = output_model_sed_final_pkl
#%% read observed Flow [m3/s] SSC [mg/L] 

df_Q_SSC_obs=pd.read_csv(observed_path)

"""
df_Q_SSC_obs[discharge] -- unit: m3/s
df_Q_SSC_obs[SSC] -- unit: g/m3

"""
df_Q_SSC_obs2=df_Q_SSC_obs.copy()
df_Q_SSC_obs2["time"] = pd.to_datetime(df_Q_SSC_obs2["time"])

df_Q_SSC_obs2 = df_Q_SSC_obs2[
    (df_Q_SSC_obs2["time"] >= start_date) &
    (df_Q_SSC_obs2["time"] <= end_date)]

# -----------------------------
# 3) Resample to hourlyn(mean is standard for SSC + discharge)
dfobs_hourly = (
    df_Q_SSC_obs2
    .set_index("time")
    .resample("h")
    .mean(numeric_only=True)
    .reset_index())

# 4) Match df_runoff time format
# df_runoff uses "time" column
dfobs_hourly = dfobs_hourly.rename(columns={"datetime": "time"})

df_SSC_obs=dfobs_hourly.drop(columns=['segId','discharge'])

# change SSC to kg/m3 if mg/L (commmonly SSC is reported in mg/L - or g/m3)
df_SSC_obs['SSC']=df_SSC_obs['SSC']/1000



#%% read Hydrology model (here SUMMA) settings

setting_path=os.path.join(catchment_path,setting_directory) # '/Users/armanhaddadchi/Documents/CONFLUENCE/domain_CAN_07BJ001_macro/settings/SUMMA/modelDecisions.txt'

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
    
from utils import read_tbl_like_text

params=read_tbl_like_text (setting_path)

# for outputcontrol.txt
# averageRoutedRunoff=params.loc[params['name']=='averageRoutedRunoff','levels'].iloc[0]

# for modelDecisions.txt
soilCatTbl=params.loc[params['name']=='soilCatTbl','value'].iloc[0]
vegeParTbl=params.loc[params['name']=='vegeParTbl','value'].iloc[0]

#% read catchment shapefile from confluence

cat_gdf=gpd.read_file(os.path.join(catchment_path,cat_shp_path))

#% read land cover data 

# from SUMMA
# landcover_Box=rasterio.open(os.path.join(catchment_path,landcover_path))

#%% read land cover data from 2020-Canada Land Cover
landcover_Box=rasterio.open(landcover_path)

#% intersect land cover map 

from utils import raster_cut

landcover_array, landcover_shape, nodata,landcover_transform, landcover_crs, landcover_profile = raster_cut(cat_gdf, landcover_Box)

#% categorise land cover classes into erodibility keys

landcover_classes=pd.read_csv(landclass_directory)

# depending on land cover map input different land cover classes can be read: 
    # -LandCover_Canada2020
    # -MODIS_IGBP
    # -MODIS_LAI_FPAR ...
which_landcover=vegeParTbl

# parse data to read numbers from csv code
def parse_codes(s):
    if s is None or str(s).strip() == "":
        return []
    return [int(v) for v in re.split(r"\s*,\s*", str(s))]

val_to_class = {}

for _, row in landcover_classes.iterrows():
    target_class = int(row["Landcover_Erodibility_Classes"])
    codes = parse_codes(row[which_landcover])
    for c in codes:
        val_to_class[c] = target_class

#% make new land cover map based on erodibility (as an array)

# create look up table
max_code = int(np.nanmax(landcover_array))
lut = np.zeros(max_code + 1, dtype=np.uint8)  # default = 0

for code, tgt in val_to_class.items():
    if 0 <= code <= max_code:
        lut[code] = tgt

# make new land cover class based on erodibility
landcover_erod = np.full(landcover_shape, np.nan, dtype=np.float32)

landcover_int = landcover_array.astype(int)
valid = (~np.isnan(landcover_array) &
         (landcover_int >= 0) &
         (landcover_int <= max_code)
         )

landcover_erod[valid] = lut[landcover_int[valid]]

# Preserve nodata as 0 (or change if desired)
if nodata is not None:
    landcover_erod[landcover_array == nodata] = np.nan

#%% DEM file from Hydrology model attributes

demBox=rasterio.open(os.path.join(catchment_path,dem_path))

#% intersect DEM map 
# landcover_array, landcover_shape, nodata,landcover_transform, landcover_crs, landcover_profile = raster_cut(cat_gdf, landcover_Box)
dem_array, dem_shape, nodata,dem_transform, dem_crs, dem_profile=raster_cut(cat_gdf, demBox)


#% generate slope layer from DEM
 
if demBox.crs.is_geographic and 0.0002<demBox.transform.a<0.0003:
    dx=dy=30  
elif demBox.crs.is_geographic and 0.0007<demBox.transform.a<0.0009:
    dx=dy=90  
elif demBox.crs.is_geographic and 0.002<demBox.transform.a<0.003:
    dx=dy=250
elif demBox.crs.is_geographic and 0.004 <demBox.transform.a<0.005:
    dx=dy=500
elif demBox.crs.is_geographic and 0.007 <demBox.transform.a<0.009:
    dx=dy=1000
else:  #elif demBox.crs.is_projected:
    dx = demBox.transform.a
    dy = abs(demBox.transform.e)
    
# work on a float copy
dem_array = dem_array.astype(np.float32)

# handle nodata
if nodata is not None:
    dem_array = np.where(dem_array == nodata, np.nan, dem_array)

# gradients (dz/dy, dz/dx)
dz_dy, dz_dx = np.gradient(dem_array, dy, dx)

# slope in degrees
slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))).astype(np.float32)

# slope in percent
slope_perc=(100*np.sqrt(dz_dx**2 + dz_dy**2)).astype(np.float32)

#%% read surficial geology map

geol=rasterio.open(geol_path)

geol_array, geol_shape, nodata,geol_transform, geol_crs, geol_profile = raster_cut(cat_gdf, geol)

#% categorise geology classes into erodibility keys

geol_classes=pd.read_csv(geolclass_directory)

# --- build mapping dictionary ---
value_to_rank = dict(
    zip(geol_classes["RasterValue"], geol_classes["Erodibility Rank"])
)

# --- initialize output as NaN ---
geol_erod = np.full(geol_array.shape, np.nan, dtype=np.float32)

# --- apply mapping ---
for val, rank in value_to_rank.items():
    geol_erod[geol_array == val] = rank

#%% read rainfall data (for consistency use similar to those used in Hydrology model)

from utils import forcingnc_to_dataframe
from utils import forcingnc_to_dict_by_hru

rain = forcingnc_to_dataframe(
    directory=os.path.join(catchment_path,rainfall_directory),
    var_name="precipitation_flux",  # "pptrate" "precipitation_flux"
    time_name="time",
    hru_name="hruId",
    start_date=start_date,
    end_date=end_date,
    save_csv=False,
    output_csv=None
)

rain_dict=forcingnc_to_dict_by_hru(
    os.path.join(catchment_path,rainfall_directory),
    var_name="precipitation_flux",# "pptrate" "precipitation_flux"
    time_name="time",
    hru_name="hruId",
    start_date=start_date,
    end_date=end_date
)

dt = (
    rain[rain['hruId'] == rain['hruId'].iloc[0]]['time']
    .diff()
    .dropna()
    .iloc[0]
    .total_seconds()
)


#%% read soil texture data

# Number of size fraction classes
# number_fractions=3
number_fractions = cfg.get("erosion3_optimization", {}).get("number_fractions", 3)

sand=rasterio.open(sand_path)
silt=rasterio.open(silt_path)

sand_array, sand_shape, nodata, sand_transform, sand_crs, sand_profile=raster_cut(cat_gdf, sand)
silt_array, silt_shape, nodata, silt_transform, silt_crs, silt_profile=raster_cut(cat_gdf, silt)


#%% map land cover, geology, soil texture and DEM
# from mapMaker import make_map_classes
# make_map_classes (landcover_array,np.nan , 'Original landcover classes')
# make_map_classes (landcover_erod,np.nan , 'Erodibility landcover classes')
# from mapMaker import make_map_continuous

# make_map_continuous (dem_array, "Digital Elevation Model (DEM)","Elevation","terrain")
# make_map_continuous (slope_perc, "Slope from dem","Percent", "Spectral_r")

# make_map_classes (geol_array,np.nan , 'original geology classes')
# make_map_classes (geol_erod,np.nan , 'Erodibility geology classes')
# make_map_continuous (sand_array,'Sand', 'Top soil proportion (%)', "Spectral_r")
# make_map_continuous (silt_array,'Silt', 'Top soil proportion (%)', "Spectral_r")

#%% read catchment hru

cat_hru=gpd.read_file(os.path.join(catchment_path,cat_hru_path))

#%% Generate a grid with user-defined size

from utils import make_catchment_grid

# grid= make_catchment_grid (cat_gdf, cell_size, clip_to_catchment=False)

grid = make_catchment_grid(
    cat_gdf=cat_gdf,
    cell_size=cell_size,
    clip_to_catchment=False,
    cat_hru=cat_hru,
    hru_id_col="HRU_ID",
    assign_hru_method="largest_overlap"
)
print(grid.head(10))


#% overlay grid for land cover and geology

from utils import extract_class_from_grid
landcover_erod_dominant, landcover_erod_fractions = extract_class_from_grid (grid,
                                                                             landcover_erod,
                                                                             landcover_shape,
                                                                             landcover_transform,
                                                                             landcover_crs,
                                                                             grid_id_col="grid_id",
                                                                             fill_grid_id=np.nan, 
                                                                             feature_name='landcover',
                                                                             return_long_fractions=False)

geol_erod_dominant, geol_erod_fractions = extract_class_from_grid (grid,
                                                                   geol_erod,
                                                                   geol_shape,
                                                                   geol_transform,
                                                                   geol_crs,
                                                                   grid_id_col="grid_id",
                                                                   fill_grid_id=np.nan, 
                                                                   feature_name='geol',
                                                                   return_long_fractions=False)

#% overlay grid for slope

from utils import extract_stats_from_grid

slope_stats= extract_stats_from_grid(
    grid,
    map_array=slope_perc,
    map_shape=dem_shape,
    map_transform=dem_transform,
    map_crs=dem_crs,
    feature_name="slope",
    grid_id_col="grid_id",
    fill_grid_id=np.nan
)

#% overlay grid for fractions
from utils import extract_stats_from_hru

sand_hru_stat = extract_stats_from_hru(
    cat_hru=cat_hru,
    map_array=sand_array,
    map_shape=sand_shape,
    map_transform=sand_transform,
    map_crs=sand_crs,
    feature_name="sand",
    hru_id_col="HRU_ID"
)

silt_hru_stat = extract_stats_from_hru(
    cat_hru=cat_hru,
    map_array=sand_array,
    map_shape=sand_shape,
    map_transform=sand_transform,
    map_crs=sand_crs,
    feature_name="silt",
    hru_id_col="HRU_ID"
)

#%% join layers back with grid info

# input of model including all na  
model_input_all = (
    grid
    .merge(slope_stats, on="grid_id", how="left")
    .merge(landcover_erod_dominant, on="grid_id", how="left")
    .merge(geol_erod_dominant, on="grid_id", how="left")
)

model_input = (
    grid
    .merge(slope_stats, on="grid_id", how="left")
    .merge(landcover_erod_dominant, on="grid_id", how="left")
    .merge(geol_erod_dominant, on="grid_id", how="left")
    .dropna(subset=[
    "median_slope",
    "dominant_class_landcover",
    "dominant_class_geol"
    ])
)

# print (model_input.head(10))

# Optional: also join fractions (wide tables)
model_input_wide = (
    model_input
    .merge(landcover_erod_fractions, on="grid_id", how="left")
    .merge(geol_erod_fractions, on="grid_id", how="left")
)

# pd.set_option("display.max_columns", None)
# pd.set_option("display.width", None)
# print(model_input.head(5))

#%% Extract runoff data from Hydrology model (here SUMMA)
from utils import extract_runoff_nc
from utils import extract_runoff_nc_to_dict_by_hru


df_runoff = extract_runoff_nc(
    nc_file=os.path.join(catchment_path,SUMMA_path),
    var_name="averageRoutedRunoff",
    time_name="time",
    hru_name="hruId",
    start_date=start_date,
    end_date=end_date,
    save_csv=False
)

df_runoff_dict = extract_runoff_nc_to_dict_by_hru(
    nc_file=os.path.join(catchment_path,SUMMA_path),
    var_name="averageRoutedRunoff",
    time_name="time",
    hru_name="hruId",
    start_date=start_date,
    end_date=end_date,
    save_csv=False
)

# -------------------------
# Extract SWE data from SUMMA
# -------------------------
if cold_region:
    try:
        # cold_region=True → read actual SWE from SUMMA
        df_swe = extract_runoff_nc(
            nc_file=os.path.join(catchment_path, SUMMA_path),
            var_name="scalarSWE",
            time_name="time",
            hru_name="hruId",
            start_date=start_date,
            end_date=end_date,
            save_csv=False
        )
    except Exception as e:
        df_swe = df_runoff[["time", "hruId"]].copy()
        df_swe["scalarSWE"] = 0.0
else:
    # cold_region=False → create same structure as runoff, but SWE = 0
    df_swe = df_runoff[["time", "hruId"]].copy()
    df_swe["scalarSWE"] = 0.0

#%% extract flow data from hydrology routing model (here mizuRoute)
# nc = Dataset(os.path.join(catchment_path,mizu_path))

# # Dimensions (gives #timesteps, #layers, etc.)
# print("Dimensions:")
# for d in nc.dimensions:
#     print(f"  {d}: {len(nc.dimensions[d])}")

# # Variables
# print("\nVariables:")
# print(list(nc.variables.keys()))

# # Timeframe (common SUMMA: 'time')
# if "time" in nc.variables:
#     tvar = nc.variables["time"]
#     tvals = tvar[:]
#     units = getattr(tvar, "units", None)
#     cal = getattr(tvar, "calendar", "standard")

#     print("\nTime:")
#     print("  n_timesteps:", len(tvals))
#     if units is not None:
#         dt = num2date(tvals, units=units, calendar=cal)
#         print("  start:", dt[0])
#         print("  end  :", dt[-1])
#     else:
#         print("  time units not found; showing raw:", tvals[:5])

# nc.close()

#%% extract rivernetwork hydrological data
from utils import extract_flow_variable_nc
from utils import extract_flow_variable_nc_to_dict_by_segid
# width
df_width=extract_flow_variable_nc(
    os.path.join(catchment_path,mizu_path),
    'channel_width',
    time_name="time",
    seg_name="segId",
    start_date=start_date,
    end_date=end_date,
    save_csv=False,
    output_csv=None
)
df_width_dict=extract_flow_variable_nc_to_dict_by_segid(
    os.path.join(catchment_path,mizu_path),
    'channel_width',
    time_name="time",
    seg_name="segId",
    start_date=start_date,
    end_date=end_date,
    save_csv=False,
    output_dir=None
)
# water depth
df_h=extract_flow_variable_nc(
    os.path.join(catchment_path,mizu_path),
    'channel_depth',
    time_name="time",
    seg_name="segId",
    start_date=start_date,
    end_date=end_date,
    save_csv=False,
    output_csv=None
)
df_h_dict=extract_flow_variable_nc_to_dict_by_segid(
    os.path.join(catchment_path,mizu_path),
    'channel_depth',
    time_name="time",
    seg_name="segId",
    start_date=start_date,
    end_date=end_date,
    save_csv=False,
    output_dir=None
)
# unit flow
df_q=extract_flow_variable_nc(
    os.path.join(catchment_path,mizu_path),
    'unit_flow',
    time_name="time",
    seg_name="segId",
    start_date=start_date,
    end_date=end_date,
    save_csv=False,
    output_csv=None
)
df_q_dict=extract_flow_variable_nc_to_dict_by_segid(
    os.path.join(catchment_path,mizu_path),
    'unit_flow',
    time_name="time",
    seg_name="segId",
    start_date=start_date,
    end_date=end_date,
    save_csv=False,
    output_dir=None
)
# discharge
df_Q=extract_flow_variable_nc(
    os.path.join(catchment_path,mizu_path),
    'discharge',
    time_name="time",
    seg_name="segId",
    start_date=start_date,
    end_date=end_date,
    save_csv=False,
    output_csv=None
)
df_Q_dict=extract_flow_variable_nc_to_dict_by_segid(
    os.path.join(catchment_path,mizu_path),
    'discharge',
    time_name="time",
    seg_name="segId",
    start_date=start_date,
    end_date=end_date,
    save_csv=False,
    output_dir=None
)

#%% Erosion model coeffs for a and b

def read_all_coefs(
    coefs,
    name_col="parameter",
    value_col="default",
    low_col="low_range",
    up_col="up_range",
    priority_col="priority"
):
    """
    Return dictionary of parameters:
        {
            parameter_name: {
                "value": v,
                "low": l,
                "up": u,
                "priority": p
            }
        }

    Notes
    -----
    - priority is optional
    - if missing or blank, it is stored as None
    """

    params = {}

    for _, row in coefs.iterrows():
        param_name = str(row[name_col]).strip()

        priority_val = None
        if priority_col in coefs.columns:
            raw_priority = row[priority_col]
            if pd.notna(raw_priority) and str(raw_priority).strip() != "":
                priority_val = int(raw_priority)

        params[param_name] = {
            "value": row[value_col],
            "low": row[low_col],
            "up": row[up_col],
            "priority": priority_val
        }

    return params
if model_mode == "validation":
    param_dict = read_all_coefs(validation_coefs)
else:
    param_dict = read_all_coefs(coefs)

# print(param_dict["al0"]["value"])
# print(param_dict["al0"]["low"])
# print(param_dict["al0"]["up"])

#%% re-write depth and unitflow, flow and  dataframe to be in a format of 

from utils import long_to_time_index_matrix


h = long_to_time_index_matrix(
    df_h,
    id_col="segId",
    value_col="channel_depth"
)

q=long_to_time_index_matrix(
    df_q,
    id_col="segId",
    value_col="unit_flow"
)
Q=long_to_time_index_matrix(
    df_Q,
    id_col="segId",
    value_col="discharge"
)
width=long_to_time_index_matrix(
    df_width,
    id_col="segId",
    value_col="channel_width"
)

df_runoff["time"] = pd.to_datetime(df_runoff["time"]).dt.round("h")

# keep the river-routing inputs on the same rounded hourly period used by runoff
routing_common_times = (
    pd.DatetimeIndex(h.index)
    .intersection(pd.DatetimeIndex(q.index))
    .intersection(pd.DatetimeIndex(Q.index))
    .intersection(pd.DatetimeIndex(width.index))
    .intersection(pd.DatetimeIndex(df_runoff["time"].unique()))
    .sort_values()
)

h = h.loc[routing_common_times].copy()
q = q.loc[routing_common_times].copy()
Q = Q.loc[routing_common_times].copy()
width = width.loc[routing_common_times].copy()

#%% ======================
#   Shared routing inputs
#   ======================

# read shapefile of river network
river_gdf = gpd.read_file(os.path.join(catchment_path, river_shp_file))

# read reach bed calibration and sediment size classes
# reach_bed_calib = pd.read_csv(os.path.join(main_dir_routing, reach_bed_file))
sediment_size = pd.read_csv(os.path.join(main_dir_routing, sediment_size_file))

# ensure TempSedRout functions can be imported inside optimisation_updated.py
if main_dir_routing not in sys.path:
    sys.path.append(main_dir_routing)

#%% for validation or calibration

if model_mode == "validation":

    from optimisation_updated import build_model_sed_from_params
    from optimisation_updated import add_time_columns_to_model
    from optimisation_updated import calculate_grid_ssc
    # from optimisation_updated import save_validation_results3_full
    from optimisation_updated import save_validation_results3_basic

    from utils import compute_hru_ssc_from_grids_pergridrunoff
    from utils import route_ssc_hru_gamma
    from utils import create_ssc_hru_fraction_dict
    from utils import fill_missing_hru

    model_sed = build_model_sed_from_params(param_dict, model_input)

    # add time step columns to model_sed
    # round all time columns
    for df in [df_runoff, df_swe, df_h, df_Q, df_q, df_width]:
        df["time"] = pd.to_datetime(df["time"]).dt.round("h")

    # find common times between df_runoff and df_h
    common_times = (
        pd.Index(df_runoff["time"].unique())
        .intersection(df_h["time"].unique())
        .sort_values()
    )

    # filter all dataframes
    df_runoff = df_runoff[df_runoff["time"].isin(common_times)].copy()
    df_h = df_h[df_h["time"].isin(common_times)].copy()
    df_Q = df_Q[df_Q["time"].isin(common_times)].copy()
    df_q = df_q[df_q["time"].isin(common_times)].copy()
    df_width = df_width[df_width["time"].isin(common_times)].copy()

    model_sed, time_cols = add_time_columns_to_model(model_sed, common_times)

    model_sed = calculate_grid_ssc(
        model_sed,
        df_runoff,
        rain,
        time_cols,
        df_swe=df_swe,
        cold_region=cold_region,
        zero_landcover_class0=True
    )

    param_dict_final = param_dict
    model_sed_final1b = model_sed

    # 1) identify timestep columns
    time_cols = [c for c in model_sed_final1b.columns if c.startswith("t_")]

    SSC_hru_final1b = compute_hru_ssc_from_grids_pergridrunoff(
        model_sed=model_sed_final1b,
        df_runoff=df_runoff,
        grid_hru_col="HRU_ID",
        runoff_hru_col="hruId",
        runoff_col="averageRoutedRunoff",
        return_wide=True
    )

    a_rout = param_dict_final["a_rout"]["value"]
    mt_rout = param_dict_final["mt_rout"]["value"]
    K_rout = param_dict_final["K_rout"]["value"]

    SSC_hru_routed_final1b = route_ssc_hru_gamma(
        SSC_hru=SSC_hru_final1b,
        a=a_rout,
        mt=mt_rout,
        K=K_rout,
        hru_col="HRU_ID"
    )

    SSC_hru = SSC_hru_routed_final1b

    SSC_hru_frac = create_ssc_hru_fraction_dict(
        SSC_hru=SSC_hru,
        sand_hru_stat=sand_hru_stat,
        silt_hru_stat=silt_hru_stat,
        hru_col="HRU_ID",
        sand_col="mean_sand",
        silt_col="mean_silt",
        number_fractions=number_fractions
    )

    SSC_hru_frac = {
        i: df.set_index("HRU_ID")
        for i, df in SSC_hru_frac.items()
    }

    # make common times between SSC_hru and mizurout input
    h_times = pd.DatetimeIndex(h.index).sort_values()
    common_times = h_times.copy()

    for frac, df in SSC_hru_frac.items():
        ssc_times = pd.to_datetime(
            df.columns,
            format="t_%Y%m%d_%H%M%S"
        ).sort_values()

        common_times = common_times.intersection(ssc_times)

    h = h.loc[common_times].copy()
    q = q.loc[common_times].copy()
    Q = Q.loc[common_times].copy()
    width = width.loc[common_times].copy()

    SSC_hru_frac_common = {}

    for frac, df in SSC_hru_frac.items():
        col_time_map = {
            pd.to_datetime(col, format="t_%Y%m%d_%H%M%S"): col
            for col in df.columns
        }

        keep_cols = [col_time_map[t] for t in common_times if t in col_time_map]
        SSC_hru_frac_common[frac] = df.loc[:, keep_cols].copy()

    SSC_hru_frac = SSC_hru_frac_common

    # match SSC_hru_frac with river_gdf IDs
    # add zero rows for all time steps for missing HRU_ID
    SSC_hru_frac = fill_missing_hru(
        SSC_hru_frac,
        river_gdf,
        id_col="LINKNO"
    )

    # #save basic validation/calibration-style outputs
    validation_outputs3 = save_validation_results3_basic(
        param_dict=param_dict,
        model_input=model_input,
        df_runoff=df_runoff,
        rain=rain,
        df_SSC_obs=df_SSC_obs,
        sand_hru_stat=sand_hru_stat,
        silt_hru_stat=silt_hru_stat,
        river_gdf=river_gdf,
        sediment_size=sediment_size,
        toml_file=routing_constants_toml,
        h=h,
        q=q,
        Q=Q,
        width=width,
        output_dir=output_dir,
        file_name=final_file_name3,
        obs_time_col="time",
        obs_value_col="SSC",
        zero_landcover_class0=True,
        number_fractions=number_fractions,
        df_swe=df_swe,
        cold_region=cold_region,
        use_storage=use_storage,
        river_storage=river_storage,
        storage_data_type=storage_data_type
    )

    # save full validation/calibration-style outputs
    # validation_outputs3 = save_validation_results3_full(
    #     param_dict=param_dict,
    #     model_input=model_input,
    #     df_runoff=df_runoff,
    #     rain=rain,
    #     cat_hru=cat_hru,
    #     df_SSC_obs=df_SSC_obs,
    #     sand_hru_stat=sand_hru_stat,
    #     silt_hru_stat=silt_hru_stat,
    #     river_gdf=river_gdf,
    #     sediment_size=sediment_size,
    #     toml_file=routing_constants_toml,
    #     h=h,
    #     q=q,
    #     Q=Q,
    #     width=width,
    #     output_dir=output_dir,
    #     file_name=final_file_name3,
    #     model_sed_pkl_name=final_model_sed_final_pkl,
    #     obs_time_col="time",
    #     obs_value_col="SSC",
    #     zero_landcover_class0=True,
    #     number_fractions=number_fractions,
    #     df_swe=df_swe,
    #     cold_region=cold_region,
    #     use_storage=use_storage,
    #     river_storage=river_storage,
    #     storage_data_type=storage_data_type,
    #     objective="log_rmse"
    # )

    print("Saved validation full results using file name:", final_file_name3)


else:

    #% ======================
    #   Optimisation 3 - Combined ErosionModel + TempSedRout optimisation
    #   This optimises Erosion parameters and TempSedRout parameters together
    #   comparing final routed outlet SSC vs. df_SSC_obs
    #   ======================

    '''
    Recommended staged calibration strategy
    ---------------------------------------
    Although optimise3_deap can optimise all selected parameters together,
    a stable practical strategy is:

    Step 1: Optimise main erosion parameters first.

        # Base coefficients
        "abase", "bbase"

        # Slope coefficients
        "as", "bs"

        # Rainfall / erosion / snow
        "crain", "ceros", "ksnow"

    Step 2: Keep the main erosion parameters fixed at their calibrated values,
    then optimise main TempSedRout routing parameters.

        # Dispersion coefficients
        "dispers1_TempSedRout",
        "dispers2_TempSedRout",
        "dispers3_TempSedRout"

        # Sediment size distribution
        "median_diam_TempSedRout",
        "SF_TempSedRout",
        "interp_TempSedRout"

        # Deposition coefficients
        "Fd1_TempSedRout",
        "Fd2_TempSedRout",
        "Fd3_TempSedRout"

        # Stream power / entrainment coefficients
        "cr1_TempSedRout",
        "cr2_TempSedRout",
        "cr3_TempSedRout"

    Step 3: Keep the main erosion and routing parameters fixed, then optimise
    landcover and geology multipliers.

        # Landcover coefficients for a
        "al0", "al1", "al2", "al3", "al4", "al5"

        # Landcover coefficients for b
        "bl0", "bl1", "bl2", "bl3", "bl4", "bl5"

        # Geology coefficients for a
        "ag1", "ag2", "ag3", "ag4", "ag5", "ag6", "ag7",
        "ag8", "ag9", "ag10", "ag11", "ag12", "ag13"

        # Geology coefficients for b
        "bg1", "bg2", "bg3", "bg4", "bg5", "bg6", "bg7",
        "bg8", "bg9", "bg10", "bg11", "bg12", "bg13"

    Step 4: Optimise HRU gamma-routing parameters.

        "a_rout", "mt_rout", "K_rout"

    Step 5: If use_storage=True, keep the other calibrated parameters fixed
    and optimise storage parameters.

        "fl_storage", "fh_storage", "fw_storage", "fa_storage"
    '''

    from optimisation_updated import optimise3_deap
    from optimisation_updated import prepare_obs_sim_series3
    from optimisation_updated import objective_from_series
    from optimisation_updated import save_optimisation_results3_full

    # -----------------------------------------------------
    # Read Optimisation 3 settings from TOML
    # -----------------------------------------------------
    erosion3_cfg = cfg.get("erosion3_optimization", None)

    if erosion3_cfg is None:
        objective3 = "log_rmse"
        cold_region3 = cold_region
        zero_landcover_class0 = True
        optimize_hill_routing_params = False

        optimise_only_erosion = cfg_list_or_none(
            cfg["erosion1b_optimization"],
            "optimize_only"
        )

        optimise_only_routing = cfg_list_or_none(
            cfg["erosion2_optimization"],
            "optimize_only"
        )

        number_fractions = 3

        n_generations3 = cfg["erosion2_optimization"].get("n_generations", 10)
        population_size3 = cfg["erosion2_optimization"].get("population_size", 10)
        cxpb3 = cfg["erosion2_optimization"].get("cxpb", 0.6)
        mutpb3 = cfg["erosion2_optimization"].get("mutpb", 0.3)
        eta3 = cfg["erosion2_optimization"].get("eta", 20.0)
        seed3 = cfg["erosion2_optimization"].get("seed", 42)
        checkpoint_path3 = "optimise3_deap_checkpoint.pkl"
        early_stop_rounds3 = cfg["erosion2_optimization"].get("early_stop_rounds", 5)
        early_stop_tol3 = cfg["erosion2_optimization"].get("early_stop_tol", 1e-4)

    else:
        objective3 = erosion3_cfg.get("objective", "log_rmse")
        cold_region3 = erosion3_cfg.get("cold_region", cold_region)
        zero_landcover_class0 = erosion3_cfg.get("zero_landcover_class0", True)
        optimize_hill_routing_params = erosion3_cfg.get(
            "optimize_hill_routing_params",
            True
        )

        optimise_only_erosion = cfg_list_or_none(
            erosion3_cfg,
            "optimise_only_erosion"
        )

        optimise_only_routing = cfg_list_or_none(
            erosion3_cfg,
            "optimise_only_routing"
        )

        number_fractions = erosion3_cfg.get("number_fractions", 3)

        n_generations3 = erosion3_cfg.get("n_generations", 10)
        population_size3 = erosion3_cfg.get("population_size", 10)
        cxpb3 = erosion3_cfg.get("cxpb", 0.6)
        mutpb3 = erosion3_cfg.get("mutpb", 0.3)
        eta3 = erosion3_cfg.get("eta", 20.0)
        seed3 = erosion3_cfg.get("seed", 42)
        checkpoint_path3 = erosion3_cfg.get(
            "checkpoint_path",
            "optimise3_deap_checkpoint.pkl"
        )
        early_stop_rounds3 = erosion3_cfg.get("early_stop_rounds", 5)
        early_stop_tol3 = erosion3_cfg.get("early_stop_tol", 1e-4)

    # -----------------------------------------------------
    # Run Optimisation 3
    # -----------------------------------------------------
    optimised_param_dict3, best_score3, pop3, logbook3, hof3 = optimise3_deap(
        param_dict=param_dict,
        model_input=model_input,
        df_runoff=df_runoff,
        rain=rain,
        df_swe=df_swe,
        sand_hru_stat=sand_hru_stat,
        silt_hru_stat=silt_hru_stat,
        river_gdf=river_gdf,
        sediment_size=sediment_size,
        toml_file=routing_constants_toml,
        h=h,
        q=q,
        Q=Q,
        width=width,
        df_SSC_obs=df_SSC_obs,
        use_storage=use_storage,
        river_storage=river_storage,
        storage_data_type=storage_data_type,
        obs_time_col="time",
        obs_value_col="SSC",
        objective=objective3,
        cold_region=cold_region3,
        zero_landcover_class0=zero_landcover_class0,
        optimize_hill_routing_params=optimize_hill_routing_params,
        optimise_only_erosion=optimise_only_erosion,
        optimise_only_routing=optimise_only_routing,
        number_fractions=number_fractions,
        n_generations=n_generations3,
        population_size=population_size3,
        cxpb=cxpb3,
        mutpb=mutpb3,
        eta=eta3,
        seed=seed3,
        checkpoint_path=checkpoint_path3,
        early_stop_rounds=early_stop_rounds3,
        early_stop_tol=early_stop_tol3
    )

    # -----------------------------------------------------
    # Final objective check
    # -----------------------------------------------------
    obs3, sim3 = prepare_obs_sim_series3(
        param_dict=optimised_param_dict3,
        model_input=model_input,
        df_runoff=df_runoff,
        rain=rain,
        df_swe=df_swe,
        sand_hru_stat=sand_hru_stat,
        silt_hru_stat=silt_hru_stat,
        river_gdf=river_gdf,
        sediment_size=sediment_size,
        toml_file=routing_constants_toml,
        h=h,
        q=q,
        Q=Q,
        width=width,
        df_SSC_obs=df_SSC_obs,
        obs_time_col="time",
        obs_value_col="SSC",
        cold_region=cold_region3,
        zero_landcover_class0=zero_landcover_class0,
        number_fractions=number_fractions,
        use_storage=use_storage,
        river_storage=river_storage,
        storage_data_type=storage_data_type
    )

    final_score3 = objective_from_series(
        obs3,
        sim3,
        objective=objective3
    )

    print("Optimisation 3 best score:", best_score3)
    print("Final combined optimisation objective:", final_score3)

    # -----------------------------------------------------
    # Save Optimisation 3 full results
    # -----------------------------------------------------
    save_optimisation_results3_full(
        optimised_param_dict=optimised_param_dict3,
        best_score=best_score3,
        pop=pop3,
        logbook=logbook3,
        hof=hof3,

        # Erosion model inputs
        model_input=model_input,
        df_runoff=df_runoff,
        rain=rain,
        cat_hru=cat_hru,
        df_SSC_obs=df_SSC_obs,
        sand_hru_stat=sand_hru_stat,
        silt_hru_stat=silt_hru_stat,

        # TempSedRout inputs
        river_gdf=river_gdf,
        sediment_size=sediment_size,
        toml_file=routing_constants_toml,
        h=h,
        q=q,
        Q=Q,
        width=width,

        output_dir=output_dir,
        file_name=final_file_name3,
        obs_time_col="time",
        obs_value_col="SSC",
        zero_landcover_class0=zero_landcover_class0,
        number_fractions=number_fractions,
        df_swe=df_swe,
        cold_region=cold_region3,
        use_storage=use_storage,
        river_storage=river_storage,
        storage_data_type=storage_data_type
    )

    print("Saved Optimisation full results using file name:", final_file_name3)