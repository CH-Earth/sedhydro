#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SedHydro Multiprocessing code
---------
A Multi-Fraction Erosion and Sediment Transport 
Framework with Hydrology-Model-Agnostic Coupling.

Developed by the CAPE Team
University of Calgary

Repository:
https://github.com/<organization>/SedHydro

"""

# =========================================================
# 1) IMPORTS Packages
# =========================================================
import os
import sys
import re
import platform
import tomllib

import multiprocessing as mp

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio


# =========================================================
# 2) CONFIG FILE
# =========================================================
toml_file = 'directory_settings_Hay.toml'

# if len(sys.argv) > 1:
#     toml_file = sys.argv[1]
# else:
#     toml_file = "directory_settings_Hay.toml"

# =========================================================
# 3) SMALL SAFE HELPERS
# =========================================================
def parse_codes(s):
    if s is None or str(s).strip() == "":
        return []
    return [int(v) for v in re.split(r"\s*,\s*", str(s))]


def read_all_coefs(
    coefs,
    name_col="parameter",
    value_col="default",
    low_col="low_range",
    up_col="up_range",
    priority_col="priority"
):
    params = {}
    has_priority = priority_col in coefs.columns

    for _, row in coefs.iterrows():
        param_name = str(row[name_col]).strip()

        priority_val = None
        if has_priority:
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


def clean_geometries(gdf):
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf


def build_landcover_erodibility(landcover_array, landcover_classes, which_landcover):
    val_to_class = {}

    for _, row in landcover_classes.iterrows():
        target_class = int(row["Landcover_Erodibility_Classes"])
        codes = parse_codes(row[which_landcover])
        for c in codes:
            val_to_class[c] = target_class

    max_code = int(np.nanmax(landcover_array))
    lut = np.zeros(max_code + 1, dtype=np.uint8)

    for code, tgt in val_to_class.items():
        if 0 <= code <= max_code:
            lut[code] = tgt

    landcover_erod = np.full(landcover_array.shape, np.nan, dtype=np.float32)

    valid = np.isfinite(landcover_array)
    landcover_int = np.zeros_like(landcover_array, dtype=np.int32)
    landcover_int[valid] = landcover_array[valid].astype(np.int32)

    valid = valid & (landcover_int >= 0) & (landcover_int <= max_code)
    landcover_erod[valid] = lut[landcover_int[valid]]

    return landcover_erod


def build_geology_erodibility(geol_array, geol_classes):
    value_to_rank = dict(
        zip(geol_classes["RasterValue"], geol_classes["Erodibility Rank"])
    )

    geol_erod = np.full(geol_array.shape, np.nan, dtype=np.float32)

    for val, rank in value_to_rank.items():
        geol_erod[geol_array == val] = rank

    return geol_erod


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


def get_cfg_path(cfg_section, key_base):
    suffix = get_os_suffix()
    key = f"{key_base}{suffix}"

    if key in cfg_section:
        return os.path.normpath(cfg_section[key])

    if key_base in cfg_section:
        return os.path.normpath(cfg_section[key_base])

    raise KeyError(f"Missing key '{key}' or fallback key '{key_base}' in TOML.")


def safe_chdir(path_to_use):
    if os.path.isdir(path_to_use):
        os.chdir(path_to_use)
    else:
        print(f"Warning: working directory not found, skipping os.chdir(): {path_to_use}")


def cfg_list_or_none(section, key):
    values = section.get(key, None)
    if values is None:
        return None
    return list(values)


# =========================================================
# 4) MAIN WORKFLOW
# =========================================================
def main():
    # -----------------------------------------------------
    # 4.1) READ CONFIG
    # -----------------------------------------------------
    # read toml file
    cfg = load_directory_settings()
    
    # 4.1a) Settings
    start_date = cfg["settings"]["start_date"]
    end_date = cfg["settings"]["end_date"]
    cell_size = cfg["settings"]["cell_size"]
    cold_region = cfg["settings"].get("cold_region", True)
    
    # 4.1b) DIRECTORY
    # working directory
    working_directory = get_cfg_path(cfg["common"], "working_directory")
    safe_chdir(working_directory)
    
    # ========== directory for specific catchment======
    # main directory for hydrological model run
    main_directory = get_cfg_path(cfg["common"], "main_directory")
    
    # folder name for running catchment --> 
    catchment_folder = cfg["common"]["catchment_folder"]
    catchment_path = os.path.join(main_directory, catchment_folder)
    # main directory for routing hydrological model
    main_dir_routing = get_cfg_path(cfg["common"], "main_dir_routing")
    # read hydrological model setting: SUMMA
    setting_directory = cfg["catchment"]["setting_directory"]
    
    # catchment shapefile directory
    cat_shp_path = cfg["catchment"]["cat_shp_path"]
    
    # catchment hru directory
    cat_hru_path = cfg["catchment"]["cat_hru_path"]
    dem_path = cfg["catchment"]["dem_path"]
    rainfall_directory = cfg["catchment"]["rainfall_directory"]
    
    # runoff netcdf file directory from SUMMA
    SUMMA_path = cfg["catchment"]["SUMMA_path"]
    # river hydrological variables directory from mizuRoute
    mizu_path = cfg["catchment"]["mizu_path"]
    # shapefile of river network directory
    river_shp_file = cfg["catchment"]["river_shp_file"]
    
    # ============= independent directory=======
    # csv file of land cover classes
    landclass_directory = get_cfg_path(cfg["reference"], "landclass_directory")
    # surficial geology map directory
    geol_path = get_cfg_path(cfg["reference"], "geol_path")
    # csv file of geology classes
    geolclass_directory = get_cfg_path(cfg["reference"], "geolclass_directory")
    # from landcover 2020 class
    landcover_path = get_cfg_path(cfg["reference"], "landcover_path")
    # soil texture data
    sand_path = get_cfg_path(cfg["reference"], "sand_path")
    silt_path = get_cfg_path(cfg["reference"], "silt_path")
    # riverbed and sediment size class files
    # reach_bed_file = cfg["reference"]["reach_bed_file"]
    sediment_size_file = cfg["reference"]["sediment_size_file"]
    routing_constants_toml = get_cfg_path(cfg["reference"], "routing_constants_toml")
    # initial model coefficients for param_dict
    coefs_path = get_cfg_path(cfg["common"], "coefs_path")
    observed_path = get_cfg_path(cfg["observed"], "observed_path")
    
    #============= output saving directory======
    # model output saving directory
    output_dir = get_cfg_path(cfg["output"], "output_dir")
    # filename to save model output for Erosion3 model parameters[in csv and pkl]
    output_optimisation_file_name3 = cfg["output"].get(
        "output_optimisation_file_name3",
        "optimised3_combined_run1"
    )
    # storage settings
    use_storage = cfg.get("storage", {}).get("use_storage", False)
    storage_data_type = cfg.get("storage", {}).get("storage_data_type", "length")

    river_storage = None
    if use_storage:
        storage_file_dir = get_cfg_path(cfg["storage"], "storage_file_dir")
        river_storage = pd.read_csv(storage_file_dir)

    os.makedirs(output_dir, exist_ok=True)

    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)

    if working_directory not in sys.path:
        sys.path.insert(0, working_directory)

    if main_dir_routing not in sys.path:
        sys.path.insert(0, main_dir_routing)

    # -----------------------------------------------------
    # 4.2) IMPORT PROJECT FUNCTIONS INSIDE main()
    # -----------------------------------------------------
    from utils import (
        read_tbl_like_text,
        raster_cut,
        forcingnc_to_dataframe,
        extract_runoff_nc,
        extract_flow_variable_nc,
        make_catchment_grid,
        extract_class_from_grid,
        extract_stats_from_grid,
        extract_stats_from_hru,
        long_to_time_index_matrix,
    )

    from optimisation_updated import (
        optimise3_deap_mp,
        prepare_obs_sim_series3,
        objective_from_series,
    
        build_model_sed_from_params,
        align_forcing_data,
        add_time_columns_to_model,
        calculate_grid_ssc,
    
        compute_hru_ssc_from_grids_pergridrunoff,
        route_ssc_hru_gamma,
        create_ssc_hru_fraction_dict,
    
        run_final_tempsedrout,
        prepare_obs_sim_series_tempsedrout,
        save_optimisation_results3_full,
    )

    # -----------------------------------------------------
    # 4.3) READ OBSERVED SSC
    # -----------------------------------------------------
    df_Q_SSC_obs = pd.read_csv(observed_path)
    df_Q_SSC_obs["time"] = pd.to_datetime(df_Q_SSC_obs["time"])

    df_Q_SSC_obs = df_Q_SSC_obs[
        (df_Q_SSC_obs["time"] >= start_date) &
        (df_Q_SSC_obs["time"] <= end_date)
    ]

    dfobs_hourly = (
        df_Q_SSC_obs
        .set_index("time")
        .resample("h")
        .mean(numeric_only=True)
        .reset_index()
    )

    df_SSC_obs = dfobs_hourly.drop(columns=["segId", "discharge"], errors="ignore")
    df_SSC_obs["SSC"] = df_SSC_obs["SSC"] / 1000.0

    # -----------------------------------------------------
    # 4.4) READ Hydrology model settings (here SUMMA))
    # -----------------------------------------------------
    setting_path = os.path.join(catchment_path, setting_directory)
    params = read_tbl_like_text(setting_path)

    vegeParTbl = params.loc[
        params["name"] == "vegeParTbl",
        "value"
    ].iloc[0]

    # -----------------------------------------------------
    # 4.5) READ VECTOR DATA
    # -----------------------------------------------------
    cat_gdf = gpd.read_file(os.path.join(catchment_path, cat_shp_path))
    cat_gdf = clean_geometries(cat_gdf)

    cat_hru = gpd.read_file(os.path.join(catchment_path, cat_hru_path))
    cat_hru = clean_geometries(cat_hru)

    river_gdf = gpd.read_file(os.path.join(catchment_path, river_shp_file))
    river_gdf = clean_geometries(river_gdf)

    # -----------------------------------------------------
    # 4.6) READ RASTERS
    # -----------------------------------------------------
    landcover_Box = rasterio.open(landcover_path)
    demBox = rasterio.open(os.path.join(catchment_path, dem_path))
    geol = rasterio.open(geol_path)
    sand = rasterio.open(sand_path)
    silt = rasterio.open(silt_path)

    landcover_array, landcover_shape, nodata, landcover_transform, landcover_crs, landcover_profile = raster_cut(cat_gdf, landcover_Box)
    dem_array, dem_shape, nodata, dem_transform, dem_crs, dem_profile = raster_cut(cat_gdf, demBox)
    geol_array, geol_shape, nodata, geol_transform, geol_crs, geol_profile = raster_cut(cat_gdf, geol)
    sand_array, sand_shape, nodata, sand_transform, sand_crs, sand_profile = raster_cut(cat_gdf, sand)
    silt_array, silt_shape, nodata, silt_transform, silt_crs, silt_profile = raster_cut(cat_gdf, silt)

    # -----------------------------------------------------
    # 4.7) BUILD DERIVED RASTERS
    # -----------------------------------------------------
    landcover_classes = pd.read_csv(landclass_directory)

    landcover_erod = build_landcover_erodibility(
        landcover_array=landcover_array,
        landcover_classes=landcover_classes,
        which_landcover=vegeParTbl
    )

    geol_classes = pd.read_csv(geolclass_directory)
    geol_erod = build_geology_erodibility(geol_array, geol_classes)

    if demBox.crs.is_geographic and 0.0002 < demBox.transform.a < 0.0003:
        dx = dy = 30
    elif demBox.crs.is_geographic and 0.0007 < demBox.transform.a < 0.0009:
        dx = dy = 90
    elif demBox.crs.is_geographic and 0.002 < demBox.transform.a < 0.003:
        dx = dy = 250
    elif demBox.crs.is_geographic and 0.004 < demBox.transform.a < 0.005:
        dx = dy = 500
    elif demBox.crs.is_geographic and 0.007 < demBox.transform.a < 0.009:
        dx = dy = 1000
    else:
        dx = demBox.transform.a
        dy = abs(demBox.transform.e)

    dem_array = dem_array.astype(np.float32)

    if nodata is not None:
        dem_array = np.where(dem_array == nodata, np.nan, dem_array)

    dz_dy, dz_dx = np.gradient(dem_array, dy, dx)
    slope_perc = (100 * np.sqrt(dz_dx**2 + dz_dy**2)).astype(np.float32)

    # -----------------------------------------------------
    # 4.8) READ FORCING DATA
    # -----------------------------------------------------
    rain = forcingnc_to_dataframe(
        directory=os.path.join(catchment_path, rainfall_directory),
        var_name="precipitation_flux", # "pptrate" "precipitation_flux"
        time_name="time",
        hru_name="hruId",
        start_date=start_date,
        end_date=end_date,
        save_csv=False,
        output_csv=None
    )

    dt = (
        rain[rain["hruId"] == rain["hruId"].iloc[0]]["time"]
        .diff()
        .dropna()
        .iloc[0]
        .total_seconds()
    )

    df_runoff = extract_runoff_nc(
        nc_file=os.path.join(catchment_path, SUMMA_path),
        var_name="averageRoutedRunoff",
        time_name="time",
        hru_name="hruId",
        start_date=start_date,
        end_date=end_date,
        save_csv=False
    )

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
        except Exception:
            df_swe = df_runoff[["time", "hruId"]].copy()
            df_swe["scalarSWE"] = 0.0
    else:
        # cold_region=False → create same structure as runoff, but SWE = 0
        df_swe = df_runoff[["time", "hruId"]].copy()
        df_swe["scalarSWE"] = 0.0

    # -----------------------------------------------------
    # 4.9) READ Hydrology routing model variable (here MizuRoute)
    # -----------------------------------------------------
    df_width = extract_flow_variable_nc(
        os.path.join(catchment_path, mizu_path),
        "channel_width",
        time_name="time",
        seg_name="segId",
        start_date=start_date,
        end_date=end_date,
        save_csv=False,
        output_csv=None
    )

    df_h = extract_flow_variable_nc(
        os.path.join(catchment_path, mizu_path),
        "channel_depth",
        time_name="time",
        seg_name="segId",
        start_date=start_date,
        end_date=end_date,
        save_csv=False,
        output_csv=None
    )

    df_q = extract_flow_variable_nc(
        os.path.join(catchment_path, mizu_path),
        "unit_flow",
        time_name="time",
        seg_name="segId",
        start_date=start_date,
        end_date=end_date,
        save_csv=False,
        output_csv=None
    )

    df_Q = extract_flow_variable_nc(
        os.path.join(catchment_path, mizu_path),
        "discharge",
        time_name="time",
        seg_name="segId",
        start_date=start_date,
        end_date=end_date,
        save_csv=False,
        output_csv=None
    )

    h = long_to_time_index_matrix(
        df_h,
        id_col="segId",
        value_col="channel_depth"
    )

    q = long_to_time_index_matrix(
        df_q,
        id_col="segId",
        value_col="unit_flow"
    )

    Q = long_to_time_index_matrix(
        df_Q,
        id_col="segId",
        value_col="discharge"
    )

    width = long_to_time_index_matrix(
        df_width,
        id_col="segId",
        value_col="channel_width"
    )

    # -----------------------------------------------------
    # 4.10) READ PARAMETER TABLE
    # -----------------------------------------------------
    coefs = pd.read_csv(coefs_path)
    param_dict = read_all_coefs(coefs)

    # -----------------------------------------------------
    # 4.11) READ TempSedRout tables
    # -----------------------------------------------------
    sediment_size = pd.read_csv(os.path.join(main_dir_routing, sediment_size_file))

    # -----------------------------------------------------
    # 4.12) BUILD GRID
    # -----------------------------------------------------
    grid = make_catchment_grid(
        cat_gdf=cat_gdf,
        cell_size=cell_size,
        clip_to_catchment=False,
        cat_hru=cat_hru,
        hru_id_col="HRU_ID",
        assign_hru_method="largest_overlap"
    )

    grid = clean_geometries(grid)

    # -----------------------------------------------------
    # 4.13) OVERLAY GRID WITH MAPS
    # -----------------------------------------------------
    landcover_erod_dominant, landcover_erod_fractions = extract_class_from_grid(
        grid,
        landcover_erod,
        landcover_shape,
        landcover_transform,
        landcover_crs,
        grid_id_col="grid_id",
        fill_grid_id=np.nan,
        feature_name="landcover",
        return_long_fractions=False
    )

    geol_erod_dominant, geol_erod_fractions = extract_class_from_grid(
        grid,
        geol_erod,
        geol_shape,
        geol_transform,
        geol_crs,
        grid_id_col="grid_id",
        fill_grid_id=np.nan,
        feature_name="geol",
        return_long_fractions=False
    )

    slope_stats = extract_stats_from_grid(
        grid,
        map_array=slope_perc,
        map_shape=dem_shape,
        map_transform=dem_transform,
        map_crs=dem_crs,
        feature_name="slope",
        grid_id_col="grid_id",
        fill_grid_id=np.nan
    )

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
        map_array=silt_array,
        map_shape=silt_shape,
        map_transform=silt_transform,
        map_crs=silt_crs,
        feature_name="silt",
        hru_id_col="HRU_ID"
    )

    # -----------------------------------------------------
    # 4.14) BUILD MODEL INPUT
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # 4.15) OPTIMISATION SETTINGS
    # -----------------------------------------------------
    erosion3_cfg = cfg.get("erosion3_optimization", None)

    if erosion3_cfg is None:
        objective = "log_rmse"
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

        n_generations = cfg["erosion2_optimization"]["n_generations"]
        population_size = cfg["erosion2_optimization"]["population_size"]
        cxpb = cfg["erosion2_optimization"]["cxpb"]
        mutpb = cfg["erosion2_optimization"]["mutpb"]
        eta = cfg["erosion2_optimization"]["eta"]
        seed = cfg["erosion2_optimization"]["seed"]
        checkpoint_path = "optimise3_deap_mp_checkpoint.pkl"
        early_stop = cfg["erosion2_optimization"]["early_stop"]
        early_stop_rounds = cfg["erosion2_optimization"]["early_stop_rounds"]
        early_stop_tol = cfg["erosion2_optimization"]["early_stop_tol"]
        n_cores = cfg["erosion2_optimization"]["n_cores"]
        chunksize = cfg["erosion2_optimization"]["chunksize"]

    else:
        objective = erosion3_cfg.get("objective", "log_rmse")
        zero_landcover_class0 = erosion3_cfg.get("zero_landcover_class0", True)
        optimize_hill_routing_params = erosion3_cfg.get("optimize_hill_routing_params", True)

        optimise_only_erosion = cfg_list_or_none(
            erosion3_cfg,
            "optimise_only_erosion"
        )

        optimise_only_routing = cfg_list_or_none(
            erosion3_cfg,
            "optimise_only_routing"
        )

        n_generations = erosion3_cfg.get("n_generations", 30)
        population_size = erosion3_cfg.get("population_size", 40)
        cxpb = erosion3_cfg.get("cxpb", 0.6)
        mutpb = erosion3_cfg.get("mutpb", 0.3)
        eta = erosion3_cfg.get("eta", 20.0)
        seed = erosion3_cfg.get("seed", 42)
        checkpoint_path = erosion3_cfg.get(
            "checkpoint_path",
            "optimise3_deap_mp_checkpoint.pkl"
        )
        early_stop = erosion3_cfg.get("early_stop", False)
        early_stop_rounds = erosion3_cfg.get("early_stop_rounds", 10)
        early_stop_tol = erosion3_cfg.get("early_stop_tol", 1e-4)
        n_cores = erosion3_cfg.get("n_cores", None)
        chunksize = erosion3_cfg.get("chunksize", 1)
        
        slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
        if slurm_cpus is not None:
            n_cores = int(slurm_cpus)
    # If routing list is empty in TOML, optimise all eligible TempSedRout parameters.

    # -----------------------------------------------------
    # 4.16) RUN MULTIPROCESS OPTIMISATION 3
    # -----------------------------------------------------
    (
        optimised_param_dict3,
        best_score3,
        pop3,
        logbook3,
        hof3,
        generation_history3,
        population_history3
    ) = optimise3_deap_mp(
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
        objective=objective,
        cold_region=cold_region,
        zero_landcover_class0=zero_landcover_class0,
        optimize_hill_routing_params=optimize_hill_routing_params,
        optimise_only_erosion=optimise_only_erosion,
        optimise_only_routing=optimise_only_routing,
        number_fractions=3,
        n_generations=n_generations,
        population_size=population_size,
        cxpb=cxpb,
        mutpb=mutpb,
        eta=eta,
        seed=seed,
        checkpoint_path=checkpoint_path,
        early_stop=early_stop,
        early_stop_rounds=early_stop_rounds,
        early_stop_tol=early_stop_tol,
        n_cores=n_cores,
        chunksize=chunksize
    )


    # -----------------------------------------------------
    # 4.17) FINAL OBJECTIVE CHECK
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
        cold_region=cold_region,
        zero_landcover_class0=zero_landcover_class0,
        number_fractions=3,
        use_storage=use_storage,
        river_storage=river_storage,
        storage_data_type=storage_data_type
    )

    final_score3 = objective_from_series(obs3, sim3, objective=objective)

    print("Optimisation 3 best score:", best_score3)
    print("Final combined optimisation objective:", final_score3)

    # -----------------------------------------------------
    # 4.18) SAVE basic or full model results
    # -----------------------------------------------------
    # save_optimisation_results3_basic(
    #     optimised_param_dict=optimised_param_dict3,
    #     best_score=best_score3,
    #     pop=pop3,
    #     logbook=logbook3,
    #     hof=hof3,
    #     output_dir=output_dir,
    #     file_name=output_optimisation_file_name3,
    #     generation_history=generation_history3,
    #     population_history=population_history3
    # )

    save_optimisation_results3_full(
        optimised_param_dict=optimised_param_dict3,
        best_score=best_score3,
        pop=pop3,
        logbook=logbook3,
        hof=hof3,
    
        model_input=model_input,
        df_runoff=df_runoff,
        rain=rain,
        cat_hru=cat_hru,
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
        file_name=output_optimisation_file_name3,
        obs_time_col="time",
        obs_value_col="SSC",
        zero_landcover_class0=zero_landcover_class0,
        number_fractions=3,
        df_swe=df_swe,
        cold_region=cold_region,
        use_storage=use_storage,
        river_storage=river_storage,
        storage_data_type=storage_data_type,
        generation_history=generation_history3,
        population_history=population_history3
    )

    print("\nFinished ErosionModel3 multiprocessing successfully.")


# =========================================================
# 5) ENTRY POINT ONLY
# =========================================================
if __name__ == "__main__":
    mp.freeze_support()
    main()