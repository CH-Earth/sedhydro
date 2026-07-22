#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HydroSed Optimization Module

This module contains the parameter optimization framework used for
automatic calibration of HydroSed model parameters. The optimization
is implemented using the DEAP evolutionary algorithm library and
supports both single-core and multiprocessing execution.

Main capabilities
-----------------
- Evolutionary optimization of erosion and routing parameters
- DEAP-based genetic algorithm calibration
- Checkpointing and restart functionality
- Parallel model evaluation using multiprocessing
- Calibration against observed suspended sediment concentration (SSC)
- Support for multiple objective functions

Developed by the CAPE Team
University of Calgary

"""

import re
import os
import sys
import copy
import random
import numpy as np
import pandas as pd
import time
import multiprocessing as mp
import platform
from deap import base, creator, tools
import pickle
from scipy.stats import gamma




def compute_catchment_ssc(SSC_hru, df_runoff, cat_hru,
                          hru_col="HRU_ID",
                          runoff_hru_col="hruId",
                          runoff_col="averageRoutedRunoff",
                          area_col="HRU_area",
                          return_wide=False):
    """
    Compute catchment SSC as a runoff-area weighted average over HRUs.

    Parameters
    ----------
    SSC_hru : pd.DataFrame
        Wide dataframe with one row per HRU and time columns like:
        HRU_ID, t_20240115_010000, t_20240115_020000, ...

    df_runoff : pd.DataFrame
        Long dataframe with columns:
        time, hruId, averageRoutedRunoff

    cat_hru : pd.DataFrame or GeoDataFrame
        Dataframe with HRU area information.

    hru_col : str, default="HRU_ID"
        HRU column name in SSC_hru and cat_hru.

    runoff_hru_col : str, default="hruId"
        HRU column name in df_runoff.

    runoff_col : str, default="averageRoutedRunoff"
        Runoff column name in df_runoff.

    area_col : str, default="HRU_area"
        Area column name in cat_hru.

    return_wide : bool, default=False
        If True, return one-row wide dataframe with columns like t_YYYYMMDD_HHMMSS.
        If False, return long dataframe with columns: time, SSC_catchment.

    Returns
    -------
    pd.DataFrame
        Catchment SSC time series. Same unit as input SSC_hru
        Note!!!: units of Area and runoff are not important as they cancel in divisions.
    """
    # time columns in SSC_hru
    time_cols = [c for c in SSC_hru.columns if c != hru_col]
    # SSC wide -> long
    ssc_long = SSC_hru.melt(
        id_vars=hru_col,
        value_vars=time_cols,
        var_name="time_col",
        value_name="SSC"
    )

    ssc_long["time"] = pd.to_datetime(
        ssc_long["time_col"].str.replace("t_", "", regex=False),
        format="%Y%m%d_%H%M%S"
    )

    ssc_long = ssc_long[[hru_col, "time", "SSC"]]

    # runoff table
    runoff_long = df_runoff[["time", runoff_hru_col, runoff_col]].rename(
        columns={runoff_hru_col: hru_col, runoff_col: "runoff"}
    ).copy()

    runoff_long["time"] = pd.to_datetime(runoff_long["time"])

    # area table
    hru_area = cat_hru[[hru_col, area_col]].drop_duplicates(subset=hru_col).copy()

    # merge
    df = (
        ssc_long
        .merge(runoff_long, on=[hru_col, "time"], how="inner")
        .merge(hru_area, on=hru_col, how="inner")
    )

    # weights = runoff * area
    df["weight"] = df["runoff"] * df[area_col]

    # weighted SSC by time
    SSC_catchment = (
        df.groupby("time")
        .apply(
            lambda g: 0.0
            if g["weight"].sum() == 0
            else (g["SSC"] * g["weight"]).sum() / g["weight"].sum(),
            include_groups=False
        )
        .reset_index(name="SSC_catchment")
    )

    if return_wide:
        SSC_catchment_wide = SSC_catchment.set_index("time").T
        SSC_catchment_wide.columns = [
            f"t_{c.strftime('%Y%m%d_%H%M%S')}" for c in SSC_catchment_wide.columns
        ]
        SSC_catchment_wide = SSC_catchment_wide.reset_index(drop=True)
        return SSC_catchment_wide

    return SSC_catchment


def routing_gamma(qinst, a, mt, K, dt):
    """
    Route a time series using gamma routing convolution.

    Parameters
    ----------
    qinst : array-like
        Input time series.

    a : float
        Shape parameter in gamma distribution.

    mt : float
        Mean travel time / delay.

    K : int
        Routing length in number of time steps.

    dt : float
        Time step size in hours.

    Returns
    -------
    np.ndarray
        Routed time series.
    """
    a = float(a)
    mt = float(mt)
    dt = float(dt)
    K = int(round(K))
    # gamma scale parameter
    theta = mt / a
    
    n = len(qinst)

    t = np.arange(0, K + 1) * dt
    F = gamma.cdf(t, a, scale=theta)
    w = F[1:] - F[:-1]
    w = w / w.sum()

    qrouted = np.zeros(n)

    for i in range(n):
        s = 0.0
        for k in range(K):
            if i - k >= 0:
                s += w[k] * qinst[i - k]
        qrouted[i] = s

    return qrouted


def build_model_sed_from_params(param_dict, model_input):
    """
    Build erosion model coefficients from calibrated parameter values.

    This function creates the model_sed DataFrame used by SedHydro by
    translating parameter values into spatially distributed erosion
    coefficients for each grid cell or HRU. Land cover, geology, and slope
    information are combined to calculate the erosion parameters a and b.
    Additional rainfall, erosion, and snow attenuation coefficients are also
    added when available.

    Parameters
    ----------
    param_dict : dict
        Dictionary containing model parameter values and metadata. Expected
        keys include land cover coefficients (al* and bl*), geology
        coefficients (ag* and bg*), slope exponents (as, bs), base
        coefficients (abase, bbase), and erosion-related parameters
        such as crain, ceros, and optionally ksnow.

    model_input : pandas.DataFrame
        Input spatial table containing dominant land cover class, dominant
        geology class, median slope, and other grid/HRU attributes required
        for coefficient calculation.

    Returns
    -------
    model_sed : pandas.DataFrame
        Copy of model_input with added coefficient columns and final erosion
        parameters a and b used in SedHydro sediment generation calculations.
    """

    model_sed = model_input.copy()

    # -------------------------
    # a coefficients
    # -------------------------
    landcover_coef_map_a = {
        int(re.findall(r"\d+", k)[0]): v["value"]
        for k, v in param_dict.items()
        if k.startswith("al") and re.search(r"\d+", k)
    }

    geol_coef_map_a = {
        int(re.findall(r"\d+", k)[0]): v["value"]
        for k, v in param_dict.items()
        if k.startswith("ag") and re.search(r"\d+", k)
    }

    model_sed["dominant_class_landcover_a"] = (
        model_sed["dominant_class_landcover"]
        .round()
        .astype("Int64")
        .map(landcover_coef_map_a)
    )

    model_sed["dominant_class_geol_a"] = (
        model_sed["dominant_class_geol"]
        .round()
        .astype("Int64")
        .map(geol_coef_map_a)
    )

    model_sed["slope_a"] = (
        model_sed["median_slope"] ** param_dict["as"]["value"]
    )

    model_sed["a"] = (
        param_dict["abase"]["value"]
        * model_sed["dominant_class_landcover_a"]
        * model_sed["dominant_class_geol_a"]
        * model_sed["slope_a"]
    )

    # -------------------------
    # b coefficients
    # -------------------------
    landcover_coef_map_b = {
        int(re.findall(r"\d+", k)[0]): v["value"]
        for k, v in param_dict.items()
        if k.startswith("bl") and re.search(r"\d+", k)
    }

    geol_coef_map_b = {
        int(re.findall(r"\d+", k)[0]): v["value"]
        for k, v in param_dict.items()
        if k.startswith("bg") and re.search(r"\d+", k)
    }

    model_sed["dominant_class_landcover_b"] = (
        model_sed["dominant_class_landcover"]
        .round()
        .astype("Int64")
        .map(landcover_coef_map_b)
    )

    model_sed["dominant_class_geol_b"] = (
        model_sed["dominant_class_geol"]
        .round()
        .astype("Int64")
        .map(geol_coef_map_b)
    )

    model_sed["slope_b"] = (
        model_sed["median_slope"] ** param_dict["bs"]["value"]
    )

    model_sed["b"] = (
        param_dict["bbase"]["value"]
        * model_sed["dominant_class_landcover_b"]
        * model_sed["dominant_class_geol_b"]
        * model_sed["slope_b"]
    )

    # -------------------------
    # rain coefficients
    # -------------------------
    model_sed["crain"] = param_dict["crain"]["value"]
    model_sed["ceros"] = param_dict["ceros"]["value"]
    # snow attenuation coefficient
    if "ksnow" in param_dict:
        model_sed["ksnow"] = param_dict["ksnow"]["value"]

    return model_sed


def align_forcing_data(df_runoff, rain):
    """
    Align runoff and rainfall data to common hourly timestamps.

    Parameters
    ----------
    df_runoff : pandas.DataFrame
        Runoff time series with a "time" column.
    rain : pandas.DataFrame
        Rainfall time series with a "time" column.

    Returns
    -------
    df_runoff : pandas.DataFrame
        Runoff data filtered to common timestamps.
    rain : pandas.DataFrame
        Rainfall data filtered to common timestamps.
    common_times : pandas.DatetimeIndex
        Shared hourly timestamps.
    """

    df_runoff = df_runoff.copy()
    rain = rain.copy()

    df_runoff["time"] = pd.to_datetime(df_runoff["time"]).dt.round("h")
    rain["time"] = pd.to_datetime(rain["time"]).dt.round("h")

    common_times = (
        pd.Index(df_runoff["time"].unique())
        .intersection(rain["time"].unique())
        .sort_values()
    )

    df_runoff = df_runoff[df_runoff["time"].isin(common_times)].copy()
    rain = rain[rain["time"].isin(common_times)].copy()

    return df_runoff, rain, common_times


def add_time_columns_to_model(model_sed, common_times):
    """
    Add zero-valued time columns to model_sed.
    """

    time_cols = pd.Series(common_times).dt.strftime("t_%Y%m%d_%H%M%S").tolist()
    zeros = pd.DataFrame(0.0, index=model_sed.index, columns=time_cols)

    insert_pos = model_sed.columns.get_loc("ceros") + 1

    model_sed = pd.concat(
        [model_sed.iloc[:, :insert_pos], zeros, model_sed.iloc[:, insert_pos:]],
        axis=1
    )

    return model_sed, time_cols

# snow attenuation factor
def snow_attenuation_factor(swe_array, ksnow):
    """
    Snow attenuation factor for cold-region erosion.

    swe_array : np.ndarray
        SWE in mm, or kg m-2. Numerically, 1 kg m-2 = 1 mm water equivalent.

    ksnow : float or np.ndarray
        Snow attenuation coefficient [mm-1].

    Returns
    -------
    np.ndarray
        Attenuation multiplier between 0 and 1.
    """

    swe_array = np.nan_to_num(swe_array, nan=0.0)
    swe_array = np.clip(swe_array, 0.0, None)

    return np.exp(-ksnow * swe_array)


def calculate_grid_ssc(
    model_sed,
    df_runoff,
    rain,
    time_cols,
    df_swe=None,
    cold_region=True,
    zero_landcover_class0=False
):
    """
    Calculate suspended sediment concentration (SSC) for each grid cell
    and timestep using a log-transformed runoff-based erosion formulation,
    with optional cold-region snow attenuation.

    Parameters
    ----------
    model_sed : pd.DataFrame
        Sediment model dataframe containing one row per grid/HRU and the
        coefficient columns required for SSC calculation.

        Required columns:
        - HRU_ID
        - a
        - b
        - ceros
        - crain
        - dominant_class_landcover

        Additional required column when `cold_region=True`:
        - ksnow

        The dataframe is updated in place for all columns listed in
        `time_cols`.

    df_runoff : pd.DataFrame
        Long-format routed runoff dataframe with columns:
        - time
        - hruId
        - averageRoutedRunoff

        Runoff values are rounded to hourly resolution, pivoted to wide
        format, aligned to `model_sed["HRU_ID"]`, and clipped to be
        non-negative.

    rain : pd.DataFrame
        Long-format rainfall dataframe with columns:
        - time
        - hruId
        - pptrate

        Rainfall values are rounded to hourly resolution, pivoted to wide
        format, aligned to `model_sed["HRU_ID"]`, and clipped to be
        non-negative.

    time_cols : list[str]
        Output timestep column names in the format used by `model_sed`,
        typically:
        - "t_YYYYMMDD_HHMMSS"

        These columns define both the expected output timesteps and the
        alignment target for pivoted runoff, rainfall, and SWE inputs.

    df_swe : pd.DataFrame or None, default=None
        Long-format snow water equivalent (SWE) dataframe with columns:
        - time
        - hruId
        - scalarSWE

        Required only when `cold_region=True`.

        SWE values are assumed to be:
        - kg m-2
        which are numerically equivalent to:
        - mm water equivalent

        SWE values are rounded to hourly resolution, pivoted to wide
        format, aligned to `model_sed["HRU_ID"]`, and clipped to be
        non-negative.

    cold_region : bool, default=True
        Controls whether snow attenuation is applied.

        - True:
            Apply snow attenuation using:
                exp(-ksnow * SWE)

        - False:
            Use the standard erosion formulation without snow attenuation.

    zero_landcover_class0 : bool, default=False
        If True, rows with:
            dominant_class_landcover == 0

        are assigned:
            SSC = 0

        for all timesteps after SSC calculation.

        If False, SSC is computed for all rows using the same equation.

    Returns
    -------
    pd.DataFrame
        The input `model_sed` dataframe with `time_cols`
        populated by the computed SSC values for each grid
        and timestep.

    Notes
    -----
    Time preprocessing
    ------------------
    - Time values in runoff, rainfall, and SWE dataframes are rounded
      to hourly resolution before reshaping.
    - Missing aligned values are filled with 0.
    - Runoff, rainfall, and SWE are clipped to be non-negative.

    Numerical stability
    -------------------
    Small positive offsets are added before applying `log10`
    to avoid undefined values:

    - erunoff = 0.1 × minimum positive routed runoff
    - erain   = 10 × minimum positive rainfall
    - epsC    = 1e-12 for coefficient and back-transformation stability

    Base erosion equation
    ---------------------
    SSC is computed in log space using:

        log10(C + epsC)
            = log10(a)
            + b * log10(Q + erunoff)

    where:
    - C = suspended sediment concentration
    - Q = routed runoff

    Snow attenuation
    ----------------
    When `cold_region=True`, SSC is attenuated using a
    snow attenuation factor:

        f_snow = exp(-ksnow * SWE)

    where:
    - SWE    = snow water equivalent [mm]
    - ksnow  = snow attenuation coefficient [mm-1]

    Final SSC becomes:

        SSC = SSC_base × f_snow

    Interpretation:
    - Larger SWE -> stronger attenuation -> lower SSC
    - Larger ksnow -> stronger attenuation sensitivity

    Final processing
    ----------------
    - SSC values are clipped to be non-negative.
    - Optional zeroing is applied for landcover class 0.
    - The function updates `model_sed` in place and also returns it
      for convenience.
    """
    # -------------------------
    # 1) runoff -> wide
    # -------------------------
    df_runoff2 = df_runoff.copy()
    df_runoff2["time"] = pd.to_datetime(df_runoff2["time"]).dt.round("h")

    erunoff = 0.1 * df_runoff2.loc[
        df_runoff2["averageRoutedRunoff"] > 0, "averageRoutedRunoff"
    ].min()

    runoff_wide = df_runoff2.pivot_table(
        index="hruId",
        columns="time",
        values="averageRoutedRunoff",
        aggfunc="first"
    )

    runoff_wide.columns = [
        f"t_{t.strftime('%Y%m%d_%H%M%S')}" for t in runoff_wide.columns
    ]

    runoff_wide = runoff_wide.reindex(columns=time_cols)

    runoff_aligned = runoff_wide.reindex(model_sed["HRU_ID"]).reset_index(drop=True)
    runoff_np = np.nan_to_num(runoff_aligned.to_numpy(), nan=0.0)
    runoff_np = np.clip(runoff_np, 0.0, None)

    # -------------------------
    # 2) rain -> wide
    # -------------------------
    rain2 = rain.copy()
    rain2["time"] = pd.to_datetime(rain2["time"]).dt.round("h")

    erain = 10.0 * rain2.loc[
        rain2["precipitation_flux"] > 0, "precipitation_flux" #"pptrate  "precipitation_flux"
    ].min()

    rain_wide = rain2.pivot_table(
        index="hruId",
        columns="time",
        values="precipitation_flux", #"pptrate" "precipitation_flux"
        aggfunc="first"
    )

    rain_wide.columns = [
        f"t_{t.strftime('%Y%m%d_%H%M%S')}" for t in rain_wide.columns
    ]

    rain_wide = rain_wide.reindex(columns=time_cols)

    rain_aligned = rain_wide.reindex(model_sed["HRU_ID"]).reset_index(drop=True)
    rain_np = np.nan_to_num(rain_aligned.to_numpy(), nan=0.0)
    rain_np = np.clip(rain_np, 0.0, None)

    # -------------------------
    # CHANGE: 2b) SWE -> wide
    # -------------------------
    if cold_region:
        if df_swe is None:
            raise ValueError("cold_region=True requires df_swe.")

        if "ksnow" not in model_sed.columns:
            raise ValueError("cold_region=True requires model_sed['ksnow'].")

        df_swe2 = df_swe.copy()
        df_swe2["time"] = pd.to_datetime(df_swe2["time"]).dt.round("h")

        swe_wide = df_swe2.pivot_table(
            index="hruId",
            columns="time",
            values="scalarSWE",
            aggfunc="first"
        )

        swe_wide.columns = [
            f"t_{t.strftime('%Y%m%d_%H%M%S')}" for t in swe_wide.columns
        ]

        swe_wide = swe_wide.reindex(columns=time_cols)

        swe_aligned = swe_wide.reindex(model_sed["HRU_ID"]).reset_index(drop=True)
        swe_np = np.nan_to_num(swe_aligned.to_numpy(), nan=0.0)
        swe_np = np.clip(swe_np, 0.0, None)

    # -------------------------
    # 3) base SSC calculation
    # -------------------------
    epsC = 1e-12

    a_np = np.clip(model_sed["a"].to_numpy()[:, None], epsC, None)
    b_np = model_sed["b"].to_numpy()[:, None]
    ceros_np = np.clip(model_sed["ceros"].to_numpy()[:, None], epsC, None)
    crain_np = model_sed["crain"].to_numpy()[:, None]

    # same as your current no-rain form
    logC = (
        np.log10(a_np)
        + b_np * np.log10(runoff_np + erunoff)
    )
    
    #!!! use log form with rain
    # logC = (
    #     np.log10(a_np)
    #     + b_np * np.log10(runoff_np + erunoff)     
    #     + crain_np * np.log10(rain_np + erain)
    # )
    
    # -------------------------
    # CHANGE: optional cold-region attenuation
    # -------------------------
    if cold_region:
        ksnow_np = model_sed["ksnow"].to_numpy()[:, None]

        snow_att_np = snow_attenuation_factor(
            swe_array=swe_np,
            ksnow=ksnow_np
        )

        ssc_calc = ((10 ** logC) - epsC) * snow_att_np

    else:
        ssc_calc = (10 ** logC) - epsC

    ssc_calc = np.clip(ssc_calc, 0.0, None)

    # -------------------------
    # 4) optional zeroing for landcover class 0
    # -------------------------
    if zero_landcover_class0:
        landcover_zero_mask = (
            model_sed["dominant_class_landcover"]
            .fillna(-1)
            .eq(0)
            .to_numpy()[:, None]
        )
        ssc_calc = np.where(landcover_zero_mask, 0.0, ssc_calc)

    model_sed.loc[:, time_cols] = ssc_calc
    return model_sed



def aggregate_to_hru(model_sed, time_cols):
    """
    Aggregate grid SSC to HRU SSC.
    """

    SSC_hru = (
        model_sed
        .groupby("HRU_ID")[time_cols]
        .mean()
        .reset_index()
    )

    return SSC_hru


def prepare_obs_sim_series(
    param_dict=None,
    individual=None,
    param_names=None,
    base_param_dict=None,
    model_input=None,
    df_runoff=None,
    rain=None,
    df_swe=None,
    cat_hru=None,
    df_SSC_obs=None,
    obs_time_col="time",
    obs_value_col="SSC",
    cold_region=True,
    zero_landcover_class0=False,
):
    """
    Prepare aligned observed and simulated SSC series.

    Parameters
    ----------
    param_dict : dict, optional
        Parameter dictionary for a final SedHydro model run.
    individual : list, optional
        Candidate parameter values used during optimisation.
    param_names : list, optional
        Names corresponding to values in individual.
    base_param_dict : dict, optional
        Base parameter dictionary updated with individual values.
    model_input : pandas.DataFrame
        Spatial model input table.
    df_runoff : pandas.DataFrame
        Runoff time series.
    rain : pandas.DataFrame
        Rainfall time series.
    df_swe : pandas.DataFrame, optional
        Snow water equivalent time series.
    cat_hru : pandas.DataFrame
        HRU attribute table used for catchment aggregation.
    df_SSC_obs : pandas.DataFrame
        Observed SSC time series.
    obs_time_col : str, optional
        Observed time column name.
    obs_value_col : str, optional
        Observed SSC column name.
    cold_region : bool, optional
        If True, apply snow attenuation.
    zero_landcover_class0 : bool, optional
        If True, force landcover class 0 to zero SSC.

    Returns
    -------
    tuple or pandas.DataFrame
        If individual is provided, returns observed and simulated arrays:
        (obs, sim). If param_dict is provided, returns a DataFrame with
        aligned time, SSC_obs, and SSC_sim.
    """

    if param_dict is None:
        if individual is None or param_names is None or base_param_dict is None:
            raise ValueError(
                "Provide either param_dict OR (individual, param_names, base_param_dict)."
            )

        param_dict = copy.deepcopy(base_param_dict)
        for name, value in zip(param_names, individual):
            param_dict[name]["value"] = float(value)

        return_arrays = True
    else:
        param_dict = copy.deepcopy(param_dict)
        return_arrays = False
        
    param_dict = apply_order_constraints_to_param_dict(param_dict)
    # -------------------------
    # build model
    # -------------------------
    model_sed = build_model_sed_from_params(param_dict, model_input)
    
    # -------------------------
    # align forcing data
    # -------------------------
    df_runoff_a, rain_a, common_times = align_forcing_data(
        df_runoff,
        rain
    )

    # -------------------------
    # add time columns
    # -------------------------
    model_sed, time_cols = add_time_columns_to_model(model_sed, common_times)
    
    # -------------------------
    # calculate grid SSC
    # -------------------------
    model_sed = calculate_grid_ssc(
        model_sed,
        df_runoff_a,
        rain_a,
        time_cols,
        df_swe=df_swe,
        cold_region=cold_region,
        zero_landcover_class0=zero_landcover_class0
    )

    # -------------------------
    # aggregate to HRU
    # -------------------------
    SSC_hru = aggregate_to_hru(model_sed, time_cols)

    # -------------------------
    # gamma routing
    # -------------------------
    a_rout = param_dict["a_rout"]["value"]
    mt_rout = param_dict["mt_rout"]["value"]
    K_rout = param_dict["K_rout"]["value"]

    SSC_hru_routed = route_ssc_hru_gamma(
        SSC_hru=SSC_hru,
        a=a_rout,
        mt=mt_rout,
        K=K_rout,
        hru_col="HRU_ID"
    )

    # -------------------------
    # aggregate to catchment
    # -------------------------
    SSC_catchment = compute_catchment_ssc(SSC_hru_routed, df_runoff_a, cat_hru)

    if not isinstance(SSC_catchment, pd.DataFrame):
        return (None, None) if return_arrays else None

    sim_df = SSC_catchment.copy()

    if "time" not in sim_df.columns:
        possible_time_cols = [c for c in sim_df.columns if "time" in c.lower()]
        if possible_time_cols:
            sim_df = sim_df.rename(columns={possible_time_cols[0]: "time"})
        else:
            return (None, None) if return_arrays else None

    if "SSC" not in sim_df.columns:
        possible_ssc_cols = [c for c in sim_df.columns if "ssc" in c.lower()]
        if possible_ssc_cols:
            sim_df = sim_df.rename(columns={possible_ssc_cols[0]: "SSC"})
        else:
            return (None, None) if return_arrays else None

    sim_df["time"] = pd.to_datetime(sim_df["time"]).dt.round("h")

    obs_df = df_SSC_obs.copy()
    obs_df[obs_time_col] = pd.to_datetime(obs_df[obs_time_col]).dt.round("h")

    obs_df = (
        obs_df.groupby(obs_time_col, as_index=False)[obs_value_col]
        .mean()
        .rename(columns={obs_time_col: "time", obs_value_col: "SSC_obs"})
    )

    sim_df = (
        sim_df.groupby("time", as_index=False)["SSC"]
        .mean()
        .rename(columns={"SSC": "SSC_sim"})
    )

    merged = pd.merge(obs_df, sim_df, on="time", how="inner")

    if merged.empty:
        return (None, None) if return_arrays else merged

    obs = merged["SSC_obs"].to_numpy(dtype=float)
    sim = merged["SSC_sim"].to_numpy(dtype=float)

    valid = np.isfinite(obs) & np.isfinite(sim)

    if valid.sum() == 0:
        return (None, None) if return_arrays else merged.iloc[0:0].copy()

    if return_arrays:
        return obs[valid], sim[valid]

    return merged.loc[valid].reset_index(drop=True)


def rmse_from_series(obs, sim):
    """
    Compute RMSE ignoring NaNs.
    """
    valid = np.isfinite(obs) & np.isfinite(sim)
    if valid.sum() == 0:
        return 1e12
    return np.sqrt(np.mean((obs[valid] - sim[valid]) ** 2))


def log_rmse_from_series(obs, sim, eps=1e-12):
    """
    Compute log-RMSE ignoring NaNs.
    using log10
    """
    valid = np.isfinite(obs) & np.isfinite(sim)
    if valid.sum() == 0:
        return 1e12

    obs_v = np.log10(np.clip(obs[valid], eps, None))
    sim_v = np.log10(np.clip(sim[valid], eps, None))

    return np.sqrt(np.mean((obs_v - sim_v) ** 2))


def mse_from_series(obs, sim):
    """
    Compute MSE ignoring NaNs.
    """
    valid = np.isfinite(obs) & np.isfinite(sim)
    if valid.sum() == 0:
        return 1e12
    return np.mean((obs[valid] - sim[valid]) ** 2)


def kge_from_series(obs, sim):
    """
    Compute Kling-Gupta Efficiency (KGE).
    Higher is better, ideal value is 1.
    """
    valid = np.isfinite(obs) & np.isfinite(sim)
    if valid.sum() < 2:
        return -1e12

    obs_v = obs[valid]
    sim_v = sim[valid]

    obs_mean = np.mean(obs_v)
    sim_mean = np.mean(sim_v)
    obs_std = np.std(obs_v, ddof=0)
    sim_std = np.std(sim_v, ddof=0)

    if obs_mean == 0 or obs_std == 0:
        return -1e12

    r = np.corrcoef(obs_v, sim_v)[0, 1]
    if not np.isfinite(r):
        return -1e12

    alpha = sim_std / obs_std
    beta = sim_mean / obs_mean

    kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return kge if np.isfinite(kge) else -1e12


def nkge_from_series(obs, sim):
    """
    Compute normalized KGE (nKGE).
    Transforms KGE to a bounded score in [0, 1] approximately,
    where higher is better.
    """
    kge = kge_from_series(obs, sim)
    if not np.isfinite(kge):
        return -1e12
    nkge = 1.0 / (2.0 - kge)
    return nkge if np.isfinite(nkge) else -1e12


def nsh_from_series(obs, sim):
    """
    Compute Nash-Sutcliffe Efficiency (NSE/NSH).
    Higher is better, ideal value is 1.
    """
    valid = np.isfinite(obs) & np.isfinite(sim)
    if valid.sum() == 0:
        return -1e12

    obs_v = obs[valid]
    sim_v = sim[valid]

    denom = np.sum((obs_v - np.mean(obs_v)) ** 2)
    if denom == 0:
        return -1e12

    nsh = 1.0 - (np.sum((sim_v - obs_v) ** 2) / denom)
    return nsh if np.isfinite(nsh) else -1e12


def objective_from_series(obs, sim, objective="rmse"):
    """
    Compute objective value from observed and simulated series.

    For rmse and mse:
        lower is better

    For kge, nkge, nsh:
        higher is better internally, but returned as negative so DEAP can minimize.
    """
    objective = str(objective).lower()

    if objective == "rmse":
        return rmse_from_series(obs, sim)

    if objective == "mse":
        return mse_from_series(obs, sim)

    if objective == "log_rmse":
        return log_rmse_from_series(obs, sim)

    if objective == "kge":
        val = kge_from_series(obs, sim)
        return -val if np.isfinite(val) else 1e12

    if objective == "nkge":
        val = nkge_from_series(obs, sim)
        return -val if np.isfinite(val) else 1e12

    if objective in ["nsh", "nse"]:
        val = nsh_from_series(obs, sim)
        return -val if np.isfinite(val) else 1e12

    raise ValueError(
        "Unknown objective. Choose from: 'rmse', 'log_rmse', 'mse', 'kge', 'nkge', 'nsh'."
    )


def evaluate_param_set(
    individual,
    param_names,
    base_param_dict,
    model_input,
    df_runoff,
    rain,
    df_swe,
    cat_hru,
    df_SSC_obs,
    obs_time_col="time",
    obs_value_col="SSC",
    objective="log_rmse",
    cold_region=True,
    zero_landcover_class0=False,
):
    """
    Evaluate a candidate parameter set for DEAP by computing simulated and
    observed SSC series and returning the selected objective value in
    minimization form.
    
    Parameters
    ----------
    individual : list[float]
        Candidate parameter values proposed by DEAP.
    
    param_names : list[str]
        Names of the parameters corresponding to the values in `individual`.
        Values in `individual` are mapped onto these parameter names in order.
    
    base_param_dict : dict
        Baseline parameter dictionary with structure:
        {
            "param_name": {"value": float, "low": float, "up": float},
            ...
        }
        A copy is updated internally using the values from `individual`
        before model evaluation.
    
    model_input : pd.DataFrame or GeoDataFrame
        Grid-based input data used to run the sediment model.
    
    df_runoff : pd.DataFrame
        Long-format dataframe of routed runoff with columns:
        - time
        - hruId
        - averageRoutedRunoff
    
    rain : pd.DataFrame
        Long-format dataframe of rainfall with columns:
        - time
        - hruId
        - pptrate
    
    cat_hru : pd.DataFrame or GeoDataFrame
        HRU-level data including area information used for
        catchment-scale SSC aggregation.
    
    df_SSC_obs : pd.DataFrame
        Observed suspended sediment concentration (SSC) time series.
    
    obs_time_col : str, default="time"
        Column name representing timestamps in `df_SSC_obs`.
    
    obs_value_col : str, default="SSC"
        Column name representing observed SSC values in `df_SSC_obs`.
    
    objective : str, default="log_rmse"
        Objective function used to score the simulated series against the
        observed series. Supported options depend on `objective_from_series`
        and are expected to include:
        - "rmse"
        - "log_rmse"
        - "mse"
        - "kge"
        - "nkge"
        - "nsh"
        - "nse"
    
        Note:
        Metrics that are naturally maximized (for example KGE and NSE
        variants) are returned in transformed form so DEAP can minimize them.
    
    zero_landcover_class0 : bool, default=False
        If True, grids with `dominant_class_landcover == 0` are forced to
        have zero SSC across all timesteps during simulation.
    
    Returns
    -------
    tuple
        One-element tuple containing the objective value for DEAP
        minimization.
    
        - Lower values indicate better performance.
        - If evaluation fails, series preparation returns no valid overlap,
          or the objective is non-finite, a large penalty value `(1e12,)`
          is returned.
    
    Notes
    -----
    - This function is designed for direct use as the DEAP evaluation
      function.
    - Observed and simulated series are prepared using
      `prepare_obs_sim_series`.
    - Objective values are computed using `objective_from_series`.
    - Any exception during evaluation is caught and penalized so that
      optimization can continue robustly.
    """

    try:
        obs, sim = prepare_obs_sim_series(
            individual=individual,
            param_names=param_names,
            base_param_dict=base_param_dict,
            model_input=model_input,
            df_runoff=df_runoff,
            rain=rain,
            df_swe=df_swe,
            cat_hru=cat_hru,
            df_SSC_obs=df_SSC_obs,
            obs_time_col=obs_time_col,
            obs_value_col=obs_value_col,
            cold_region=cold_region,
            zero_landcover_class0=zero_landcover_class0,
        )

        if obs is None or sim is None:
            return (1e12,)

        score = objective_from_series(obs, sim, objective=objective)

        if not np.isfinite(score):
            score = 1e12

        return (score,)

    except Exception as e:
        print(f"Evaluation failed: {e}")
        return (1e12,)

def save_optimisation_checkpoint(
    output_path,
    generation,
    param_dict,
    param_names,
    population,
    logbook,
    hof,
    objective,
    zero_landcover_class0,
    optimize_hill_routing_params,
    optimize_only=None,
    generation_history=None,
    population_history=None
):
    """
    Save a SedHydro optimisation checkpoint.

    Parameters
    ----------
    output_path : str
        Path to the checkpoint file.
    generation : int
        Completed generation number.
    param_dict : dict
        Base parameter dictionary.
    param_names : list
        Names of optimised parameters.
    population : list
        Current optimisation population.
    logbook : deap.tools.Logbook
        Optimisation statistics.
    hof : deap.tools.HallOfFame
        Best solutions found so far.
    objective : str
        Objective function name.
    zero_landcover_class0 : bool
        Option to force landcover class 0 erosion to zero.
    optimize_hill_routing_params : bool
        If True, optimise hillslope routing parameters.
    optimize_only : str, optional
        Restrict optimisation to a subset of parameters.
    generation_history : pandas.DataFrame, optional
        Summary statistics by generation.
    population_history : pandas.DataFrame, optional
        Parameter values and fitness for all evaluated models.

    Returns
    -------
    None
        Writes the checkpoint file to disk.
    """

    best_ind = hof[0] if len(hof) > 0 else None

    best_score = None
    if best_ind is not None and best_ind.fitness.valid:
        best_score = best_ind.fitness.values[0]

    optimized_param_dict = copy.deepcopy(param_dict)
    if best_ind is not None:
        for name, value in zip(param_names, best_ind):
            optimized_param_dict[name]["value"] = float(value)

    optimized_param_dict = apply_order_constraints_to_param_dict(optimized_param_dict)

    checkpoint = {
        "generation": generation,
        "optimized_param_dict": optimized_param_dict,
        "best_score": best_score,
        "pop": population,
        "logbook": logbook,
        "hof": hof,
        "objective": objective,
        "zero_landcover_class0": zero_landcover_class0,
        "optimize_hill_routing_params": optimize_hill_routing_params,
        "optimize_only": optimize_only,
        "generation_history": generation_history,
        "population_history": population_history,
    }

    with open(output_path, "wb") as f:
        pickle.dump(checkpoint, f)



def optimise1a_deap(
    param_dict,
    model_input,
    df_runoff,
    rain,
    df_swe=None,          
    cat_hru=None,
    df_SSC_obs=None,
    obs_time_col="time",
    obs_value_col="SSC",
    objective="log_rmse",
    cold_region=True,
    zero_landcover_class0=False,
    optimize_hill_routing_params=True,
    n_generations=30,
    population_size=40,
    cxpb=0.6,
    mutpb=0.3,
    eta=20.0,
    seed=42,
    checkpoint_path="optimise1_deap_checkpoint.pkl",
    early_stop_rounds=None,   
    early_stop_tol=1e-4       
):
    """
    Calibrate sediment model parameters using a DEAP-based evolutionary
    algorithm by minimizing an objective function comparing simulated and
    observed catchment SSC.
    
    Parameters
    ----------
    param_dict : dict
        Dictionary of model parameters with structure:
        {
            "param_name": {"value": float, "low": float, "up": float},
            ...
        }
        Only parameters with (up > low) are included in the optimization.
        Parameters with fixed bounds are kept constant.
    
    model_input : pd.DataFrame or GeoDataFrame
        Grid-based input data used to initialize the sediment model.
        Must include:
        - HRU_ID
        - dominant_class_landcover
        - dominant_class_geol
        - median_slope
    
    df_runoff : pd.DataFrame
        Long-format dataframe of routed runoff with columns:
        - time
        - hruId
        - averageRoutedRunoff
    
    rain : pd.DataFrame
        Long-format dataframe of rainfall with columns:
        - time
        - hruId
        - pptrate
    
    cat_hru : pd.DataFrame or GeoDataFrame
        HRU-level data including area information used for
        catchment aggregation.
    
    df_SSC_obs : pd.DataFrame
        Observed suspended sediment concentration (SSC) time series.
    
    obs_time_col : str, default="time"
        Column name representing timestamps in df_SSC_obs.
    
    obs_value_col : str, default="SSC"
        Column name representing observed SSC values.
    
    objective : str, default="log_rmse"
        Objective function to optimize. Supported options:
        - "rmse"     : Root Mean Square Error
        - "log_rmse" : RMSE computed in log-space
        - "mse"      : Mean Square Error
        - "kge"      : Kling-Gupta Efficiency
        - "nkge"     : Normalized Kling-Gupta Efficiency
        - "nsh"/"nse": Nash-Sutcliffe Efficiency
    
        Note:
        Metrics that are typically maximized (KGE/NSE variants) are internally
        transformed to a minimization problem.
    
    zero_landcover_class0 : bool, default=False
        If True, grids with dominant_class_landcover == 0 are forced to have
        zero SSC across all timesteps.
    
    optimize_hill_routing_params : bool, default=True
        If False, routing parameters ("a_rout", "mt_rout") are excluded
        from optimization even if bounds are provided.
    
    n_generations : int, default=30
        Number of generations for the evolutionary algorithm.
    
    population_size : int, default=40
        Number of individuals in the population.
    
    cxpb : float, default=0.6
        Probability of crossover between individuals.
    
    mutpb : float, default=0.3
        Probability of mutation for each individual.
    
    eta : float, default=20.0
        Distribution index controlling mutation spread
        (higher values → smaller mutations).
    
    seed : int, default=42
        Random seed for reproducibility.
    
    checkpoint_path : str, default="optimise1_deap_checkpoint.pkl"
        File path used to save the last fully completed generation.
        
    Early stopping
        
        If early_stop_rounds is not None, optimisation stops when the best
        objective value has not improved by more than early_stop_tol for
        early_stop_rounds consecutive generations.
    Returns
    -------
    optimized_param_dict : dict
        Copy of param_dict with updated "value" fields for optimized parameters.
    
    best_score : float
        Best objective value achieved.
        - For rmse, log_rmse, mse: lower is better.
        - For kge/nkge/nse/nsh: returned value is minimized form
          (actual metric = -best_score).
    
    pop : list
        Final population of individuals after evolution.
    
    logbook : deap.tools.Logbook
        Recorded statistics (min, mean, std) for each generation.
    
    hof : deap.tools.HallOfFame
        Hall of Fame object containing the best individual found.
    
    Notes
    -----
    - SSC is computed at the grid level, aggregated to HRU (mean),
      then to catchment scale using `compute_catchment_ssc`.
    - Model–observation comparison is performed on overlapping timestamps only.
    - Time alignment is enforced at hourly resolution.
    - Parameter bounds are strictly enforced after mutation.
    - Generation-wise progress (objective + parameter groups) is printed.
    - If the run is interrupted, the last fully completed generation is kept
      in `checkpoint_path`.
    """

    random.seed(seed)
    np.random.seed(seed)

    objective = str(objective).lower()
    allowed_objectives = {"rmse", "log_rmse", "mse", "kge", "nkge", "nsh", "nse"}
    if objective not in allowed_objectives:
        raise ValueError(
            "objective must be one of: 'rmse', 'log_rmse', 'mse', 'kge', 'nkge', 'nsh', 'nse'"
        )

    if early_stop_rounds is not None:
        early_stop_rounds = int(early_stop_rounds)
        if early_stop_rounds <= 0:
            raise ValueError("early_stop_rounds must be a positive integer or None.")
        early_stop_tol = float(early_stop_tol)
        if early_stop_tol < 0:
            raise ValueError("early_stop_tol must be >= 0.")

    param_names = [
        k for k, v in param_dict.items()
        if float(v["up"]) > float(v["low"])
    ]

    if not optimize_hill_routing_params:
        param_names = [k for k in param_names if k not in ["a_rout", "mt_rout"]]

    bounds = [(float(param_dict[k]["low"]), float(param_dict[k]["up"])) for k in param_names]

    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    def init_individual():
        vals = []
        for low, up in bounds:
            if low == up:
                vals.append(low)
            else:
                vals.append(random.uniform(low, up))
        return creator.Individual(vals)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate",
        evaluate_param_set,
        param_names=param_names,
        base_param_dict=param_dict,
        model_input=model_input,
        df_runoff=df_runoff,
        rain=rain,
        df_swe=df_swe,                 # snow
        cat_hru=cat_hru,
        df_SSC_obs=df_SSC_obs,
        obs_time_col=obs_time_col,
        obs_value_col=obs_value_col,
        objective=objective,
        cold_region=cold_region,       # snow
        zero_landcover_class0=zero_landcover_class0,
    )

    toolbox.register("mate", tools.cxBlend, alpha=0.2)
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        eta=eta,
        low=[b[0] for b in bounds],
        up=[b[1] for b in bounds],
        indpb=0.2
    )
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("mean", np.mean)
    stats.register("std", np.std)
    
    # for timer/core info: added
    generation_start_time = time.perf_counter()
    total_start_time = time.perf_counter()
    n_cores_used = 1
    core_type = platform.processor()
    if not core_type:
        core_type = platform.machine()
    total_logical_cores = mp.cpu_count()
    # for timer/core info: finished
    
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)
    best_score_so_far = hof[0].fitness.values[0]
    stagnant_generations = 0

    def print_generation_status(gen, population):
        best_ind_gen = tools.selBest(population, 1)[0]
        best_score_gen = best_ind_gen.fitness.values[0]

        best_param_dict_gen = copy.deepcopy(param_dict)
        for name, value in zip(param_names, best_ind_gen):
            best_param_dict_gen[name]["value"] = float(value)

        a_params = {
            k: v["value"]
            for k, v in best_param_dict_gen.items()
            if k.startswith("a") or k in ["as", "abase"]
        }
        b_params = {
            k: v["value"]
            for k, v in best_param_dict_gen.items()
            if k.startswith("b") or k in ["bs", "bbase"]
        }
        routing_params = {
            k: v["value"]
            for k, v in best_param_dict_gen.items()
            if k in ["a_rout", "mt_rout"]
        }
        snow_params = {
            k: v["value"]
            for k, v in best_param_dict_gen.items()
            if k in ["ksnow"]
        }
        # for timer/core info: added
        generation_elapsed_min = (time.perf_counter() - generation_start_time) / 60.0
        total_elapsed_min = (time.perf_counter() - total_start_time) / 60.0
        # for timer/core info: finished
        
        print(f"\nGeneration {gen}/{n_generations}")
        print(f"Objective ({objective}) = {best_score_gen}")
        print(f"Elapsed this generation = {generation_elapsed_min:.2f} min")
        print(f"Elapsed total = {total_elapsed_min:.2f} min")
        print(f"Cores used = {n_cores_used} / {total_logical_cores}")
        print(f"Core/CPU type = {core_type}")
        print(f"a params = {a_params}")
        print(f"b params = {b_params}")
        print(f"routing params = {routing_params}")
        print(f"snow params = {snow_params}")
        
        # for early stop: message
        if early_stop_rounds is not None:
            print(
                f"Early stopping monitor = {stagnant_generations}/{early_stop_rounds} "
                f"(tol={early_stop_tol})"
            )

    print_generation_status(0, pop)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    record = stats.compile(pop) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    print(logbook.stream)
    
    # checkpoint save after fully completed generation 0: added
    save_optimisation_checkpoint(
        output_path=checkpoint_path,
        generation=0,
        param_dict=param_dict,
        param_names=param_names,
        population=pop,
        logbook=logbook,
        hof=hof,
        objective=objective,
        zero_landcover_class0=zero_landcover_class0,
        optimize_hill_routing_params=optimize_hill_routing_params
    )
    # checkpoint save after fully completed generation 0: added
    
    # interrupt-safe evolution loop with checkpointing: added
    try:
        for gen in range(1, n_generations + 1):
            generation_start_time = time.perf_counter()

            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            for ind in offspring:
                for j, (low, up) in enumerate(bounds):
                    if ind[j] < low:
                        ind[j] = low
                    elif ind[j] > up:
                        ind[j] = up

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring
            hof.update(pop)

            current_best_score = hof[0].fitness.values[0]
            improvement = best_score_so_far - current_best_score

            if improvement > early_stop_tol:
                best_score_so_far = current_best_score
                stagnant_generations = 0
            else:
                stagnant_generations += 1

            record = stats.compile(pop) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(logbook.stream)

            print_generation_status(gen, pop)
            
            # checkpoint save after each fully completed generation
            save_optimisation_checkpoint(
                output_path=checkpoint_path,
                generation=gen,
                param_dict=param_dict,
                param_names=param_names,
                population=pop,
                logbook=logbook,
                hof=hof,
                objective=objective,
                zero_landcover_class0=zero_landcover_class0,
                optimize_hill_routing_params=optimize_hill_routing_params
            )

            if early_stop_rounds is not None and stagnant_generations >= early_stop_rounds:
                print(
                    f"\nEarly stopping triggered at generation {gen}: "
                    f"best objective did not improve by more than {early_stop_tol} "
                    f"for {early_stop_rounds} consecutive generations."
                )
                break

    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        print(f"Last fully completed generation saved at: {checkpoint_path}")
        raise

    best_ind = hof[0]
    best_score = hof[0].fitness.values[0]

    optimized_param_dict = copy.deepcopy(param_dict)
    for name, value in zip(param_names, best_ind):
        optimized_param_dict[name]["value"] = float(value)

    return optimized_param_dict, best_score, pop, logbook, hof

def save_optimisation_results1a(
    optimised_param_dict,
    best_score,
    pop,
    logbook,
    hof,
    model_input,
    df_runoff,
    rain,
    df_swe=None,                 # 
    cat_hru=None,                # 
    df_SSC_obs=None,             # 
    output_dir=None,
    file_name="optimised_parameters",
    obs_time_col="time",
    obs_value_col="SSC",
    cold_region=True,            # 
    zero_landcover_class0=False
):
    """
    Save optimisation results (parameters, population, logbook, hof, and obs/sim)
    to both CSV and PKL files.
    """

    import os
    import pickle
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    csv_file = os.path.join(output_dir, f"{file_name}.csv")
    pkl_file = os.path.join(output_dir, f"{file_name}.pkl")

    # -------------------------
    # 1) observed vs simulated
    # -------------------------
    obs_sim_df = prepare_obs_sim_series(
        param_dict=optimised_param_dict,
        model_input=model_input,
        df_runoff=df_runoff,
        rain=rain,
        df_swe=df_swe,                  # 
        cat_hru=cat_hru,
        df_SSC_obs=df_SSC_obs,
        obs_time_col=obs_time_col,
        obs_value_col=obs_value_col,
        cold_region=cold_region,        # 
        zero_landcover_class0=zero_landcover_class0
    )
    # -------------------------
    # 2) parameters
    # -------------------------
    param_rows = []
    for k, v in optimised_param_dict.items():
        param_rows.append({
            "section": "parameters",
            "name": k,
            "value": v.get("value"),
            "low": v.get("low"),
            "up": v.get("up")
        })

    df_params = pd.DataFrame(param_rows)

    # -------------------------
    # 3) best score
    # -------------------------
    df_score = pd.DataFrame([{
        "section": "best_score",
        "name": "objective_value",
        "value": best_score
    }])

    # -------------------------
    # 4) best individual
    # -------------------------
    best_ind = hof[0]
    df_best_ind = pd.DataFrame([
        {
            "section": "best_individual",
            "name": f"param_{i}",
            "value": val
        }
        for i, val in enumerate(best_ind)
    ])

    # -------------------------
    # 5) population
    # -------------------------
    pop_rows = []
    for i, ind in enumerate(pop):
        pop_rows.append({
            "section": "population",
            "name": f"ind_{i}",
            "value": ind.fitness.values[0] if len(ind.fitness.values) > 0 else None
        })

    df_pop = pd.DataFrame(pop_rows)

    # -------------------------
    # 6) logbook
    # -------------------------
    log_rows = []
    for record in logbook:
        log_rows.append({
            "section": "logbook",
            "generation": record.get("gen"),
            "nevals": record.get("nevals"),
            "min": record.get("min"),
            "mean": record.get("mean"),
            "std": record.get("std")
        })

    df_log = pd.DataFrame(log_rows)

    # -------------------------
    # 7) obs vs sim
    # -------------------------
    if obs_sim_df is None:
        df_obs_sim_out = pd.DataFrame([{
            "section": "obs_sim",
            "name": "no_data"
        }])
    else:
        df_obs_sim_out = obs_sim_df.copy()
        df_obs_sim_out["section"] = "obs_sim"

    # -------------------------
    # 8) combine
    # -------------------------
    df_all = pd.concat(
        [df_params, df_score, df_best_ind, df_pop, df_log, df_obs_sim_out],
        ignore_index=True,
        sort=False
    )

    # -------------------------
    # 9) save CSV
    # -------------------------
    df_all.to_csv(csv_file, index=False)

    # -------------------------
    # 10) save PKL
    # -------------------------
    with open(pkl_file, "wb") as f:
        pickle.dump(
            {
                "optimised_param_dict": optimised_param_dict,
                "best_score": best_score,
                "pop": pop,
                "logbook": logbook,
                "hof": hof,
                "obs_sim_df": obs_sim_df
            },
            f
        )

    print(f"Saved CSV: {csv_file}")
    print(f"Saved PKL: {pkl_file}")


# Optimisation 1b
def get_priority_groups_from_param_dict(param_dict, prefix):
    """
    Build ordered priority groups for a parameter family.

    Example
    -------
    prefix="al" returns something like:
        [["al5", "al4", "al3", "al2"], ["al1"], ["al0"]]

    using the 'priority' values stored in param_dict.

    Rules
    -----
    - Higher priority number means higher rank
    - Same priority means equality allowed (>=)
    - Between priority groups, strict decrease is attempted (>)
    - Names inside each group are sorted by numeric suffix descending
    """

    family = []
    for name, meta in param_dict.items():
        if name.startswith(prefix):
            pr = meta.get("priority", None)
            if pr is not None:
                family.append((name, int(pr)))

    if len(family) == 0:
        return []

    grouped = {}
    for name, pr in family:
        grouped.setdefault(pr, []).append(name)

    def suffix_num(x):
        m = re.search(r"(\d+)$", x)
        return int(m.group(1)) if m else -999999

    groups = []
    for pr in sorted(grouped.keys(), reverse=True):
        names = sorted(grouped[pr], key=suffix_num, reverse=True)
        groups.append(names)

    return groups


def _repair_descending_group_values(param_dict, groups, eps=1e-12):
    """
    Enforce descending order on grouped parameters.

    groups example:
        [["al5", "al4", "al3", "al2"], ["al1"], ["al0"]]

    Meaning:
        al5 >= al4 >= al3 >= al2 > al1 > al0

    Notes
    -----
    - Within each group: non-increasing (>=)
    - Between consecutive groups: strict decrease (>) if feasible
    - If strict decrease is impossible due to bounds, fallback to equality
    """

    for group in groups:
        for name in group:
            low = float(param_dict[name]["low"])
            up = float(param_dict[name]["up"])
            val = float(param_dict[name]["value"])
            param_dict[name]["value"] = min(max(val, low), up)

    prev_group_last_val = None

    for g_idx, group in enumerate(groups):
        current_vals = []

        for i, name in enumerate(group):
            low = float(param_dict[name]["low"])
            up = float(param_dict[name]["up"])
            val = float(param_dict[name]["value"])

            upper_cap = up

            # same group -> allow equality, enforce descending
            if i > 0:
                upper_cap = min(upper_cap, current_vals[-1])

            # between groups -> strict decrease if feasible
            if g_idx > 0 and i == 0:
                strict_cap = prev_group_last_val - eps
                if strict_cap >= low:
                    upper_cap = min(upper_cap, strict_cap)
                else:
                    upper_cap = min(upper_cap, prev_group_last_val)

            val = min(max(val, low), upper_cap)
            param_dict[name]["value"] = float(val)
            current_vals.append(float(val))

        prev_group_last_val = current_vals[-1]

    return param_dict

def apply_order_constraints_to_param_dict(param_dict, eps=1e-12):
    """
    Apply ordering constraints using user-defined 'priority' values
    read from the CSV.

    Supported parameter families
    ----------------------------
    - al
    - ag
    - bl
    - bg

    Notes
    -----
    - Same priority group: >=
    - Lower priority group must be smaller than higher group if feasible
    - If strict inequality is impossible because of bounds, equality is used
    """

    for prefix in ["al", "ag", "bl", "bg"]:
        groups = get_priority_groups_from_param_dict(param_dict, prefix)
        if len(groups) > 0:
            param_dict = _repair_descending_group_values(
                param_dict=param_dict,
                groups=groups,
                eps=eps
            )

    return param_dict


def repair_individual_with_order_constraints(individual, param_names, base_param_dict, eps=1e-12):
    """
    Repair one DEAP individual so mapped parameter values satisfy the
    priority-based ordering constraints read from CSV.
    """

    temp_param_dict = copy.deepcopy(base_param_dict)

    for name, value in zip(param_names, individual):
        temp_param_dict[name]["value"] = float(value)

    temp_param_dict = apply_order_constraints_to_param_dict(
        temp_param_dict,
        eps=eps
    )

    for j, name in enumerate(param_names):
        individual[j] = float(temp_param_dict[name]["value"])

    return individual





#% Sum SSC for whole catchment from grids

def compute_catchment_ssc_from_grids_pergridrunoff(
    model_sed,
    df_runoff,
    grid_hru_col="HRU_ID",
    runoff_hru_col="hruId",
    runoff_col="averageRoutedRunoff",
    return_wide=False
):
    """
    Compute catchment SSC directly from grid SSC values, after converting
    HRU runoff to per-grid runoff within each HRU.

    Parameters
    ----------
    model_sed : pd.DataFrame
        Grid-based dataframe with one row per grid and time columns like:
        ..., HRU_ID, t_20240115_010000, t_20240115_020000, ...

        Each row is one grid cell.

    df_runoff : pd.DataFrame
        Long dataframe with columns:
        time, hruId, averageRoutedRunoff

        Runoff is assumed to be available at HRU level.

    grid_hru_col : str, default="HRU_ID"
        HRU column name in model_sed.

    runoff_hru_col : str, default="hruId"
        HRU column name in df_runoff.

    runoff_col : str, default="averageRoutedRunoff"
        Runoff column name in df_runoff.

    return_wide : bool, default=False
        If True, return one-row wide dataframe with columns like
        t_YYYYMMDD_HHMMSS.
        If False, return long dataframe with columns:
        time, SSC_catchment.

    Returns
    -------
    pd.DataFrame
        Catchment SSC time series computed from all grids.

    Notes
    -----
    This function first computes the number of grids in each HRU and then
    converts HRU runoff to per-grid runoff as:

        runoff_grid = runoff_hru / n_grids_in_hru

    Catchment SSC is then computed as:

        SSC_catchment(t) =
            sum[ SSC_grid(t) * runoff_grid(t) ] / sum[ runoff_grid(t) ]

    This avoids overweighting HRUs that contain more grids.
    """

    # 1) identify timestep columns in model_sed
    time_cols = [c for c in model_sed.columns if c.startswith("t_")]

    # 2) count number of grids per HRU
    hru_grid_counts = (
        model_sed.groupby(grid_hru_col)
        .size()
        .reset_index(name="n_grids")
    )

    # 3) grid SSC wide -> long
    ssc_long = model_sed.melt(
        id_vars=[grid_hru_col],
        value_vars=time_cols,
        var_name="time_col",
        value_name="SSC"
    )

    ssc_long["time"] = pd.to_datetime(
        ssc_long["time_col"].str.replace("t_", "", regex=False),
        format="%Y%m%d_%H%M%S"
    )

    ssc_long = ssc_long[[grid_hru_col, "time", "SSC"]]

    # 4) runoff long table
    runoff_long = df_runoff[["time", runoff_hru_col, runoff_col]].rename(
        columns={runoff_hru_col: grid_hru_col, runoff_col: "runoff_hru"}
    ).copy()

    runoff_long["time"] = pd.to_datetime(runoff_long["time"])

    # 5) attach number of grids and compute per-grid runoff
    runoff_long = runoff_long.merge(hru_grid_counts, on=grid_hru_col, how="inner")
    runoff_long["runoff_grid"] = runoff_long["runoff_hru"] / runoff_long["n_grids"]

    # 6) merge SSC and per-grid runoff
    df = ssc_long.merge(
        runoff_long[[grid_hru_col, "time", "runoff_grid"]],
        on=[grid_hru_col, "time"],
        how="inner"
    )

    # 7) runoff-weighted catchment SSC
    SSC_catchment = (
        df.groupby("time")
        .apply(
            lambda g: 0.0
            if g["runoff_grid"].sum() == 0
            else (g["SSC"] * g["runoff_grid"]).sum() / g["runoff_grid"].sum(),
            include_groups=False
        )
        .reset_index(name="SSC_catchment")
    )

    if return_wide:
        SSC_catchment_wide = SSC_catchment.set_index("time").T
        SSC_catchment_wide.columns = [
            f"t_{c.strftime('%Y%m%d_%H%M%S')}" for c in SSC_catchment_wide.columns
        ]
        SSC_catchment_wide = SSC_catchment_wide.reset_index(drop=True)
        return SSC_catchment_wide

    return SSC_catchment



def prepare_obs_sim_series1b(
    param_dict=None,
    individual=None,
    param_names=None,
    base_param_dict=None,
    model_input=None,
    df_runoff=None,
    rain=None,
    df_swe=None,                 
    df_SSC_obs=None,
    obs_time_col="time",
    obs_value_col="SSC",
    cold_region=True,            
    zero_landcover_class0=True,
):
    """
    Prepare observed and simulated SSC series for SedHydro optimisation 1b.

    Parameters
    ----------
    param_dict : dict, optional
        Parameter dictionary for a final model run.
    individual : list, optional
        Candidate parameter values used during optimisation.
    param_names : list, optional
        Names corresponding to individual values.
    base_param_dict : dict, optional
        Base parameter dictionary updated with individual values.
    model_input : pandas.DataFrame
        Spatial model input table.
    df_runoff : pandas.DataFrame
        Runoff time series.
    rain : pandas.DataFrame
        Rainfall time series.
    df_swe : pandas.DataFrame, optional
        Snow water equivalent time series.
    df_SSC_obs : pandas.DataFrame
        Observed SSC time series.
    obs_time_col : str, optional
        Observed time column name.
    obs_value_col : str, optional
        Observed SSC column name.
    cold_region : bool, optional
        If True, apply snow attenuation.
    zero_landcover_class0 : bool, optional
        If True, force landcover class 0 to zero SSC.

    Returns
    -------
    tuple or pandas.DataFrame
        If individual is provided, returns observed and simulated arrays:
        (obs, sim). If param_dict is provided, returns a DataFrame with
        aligned time, SSC_obs, and SSC_sim.

    Notes
    -----
    Optimisation 1b computes SSC at the grid level, integrates it directly
    to catchment scale using per-grid runoff, and then applies gamma routing
    to the catchment SSC series.
    """

    from utils import compute_catchment_ssc_from_grids_pergridrunoff
    from utils import routing_gamma

    if param_dict is None:
        if individual is None or param_names is None or base_param_dict is None:
            raise ValueError(
                "Provide either param_dict OR (individual, param_names, base_param_dict)."
            )

        param_dict = copy.deepcopy(base_param_dict)
        for name, value in zip(param_names, individual):
            param_dict[name]["value"] = float(value)

        return_arrays = True
    else:
        param_dict = copy.deepcopy(param_dict)
        return_arrays = False

    # -------------------------
    # apply priority-based order constraints
    # -------------------------
    param_dict = apply_order_constraints_to_param_dict(param_dict)

    # -------------------------
    # build model
    # -------------------------
    model_sed = build_model_sed_from_params(param_dict, model_input)

    # -------------------------
    # align forcing data
    # -------------------------
    df_runoff_a, rain_a, common_times = align_forcing_data(df_runoff, rain)

    # -------------------------
    # add time columns
    # -------------------------
    model_sed, time_cols = add_time_columns_to_model(model_sed, common_times)

    # -------------------------
    # calculate grid SSC
    # -------------------------
    model_sed = calculate_grid_ssc(
        model_sed,
        df_runoff_a,
        rain_a,
        time_cols,
        df_swe=df_swe,                  # CHANGE
        cold_region=cold_region,        # CHANGE
        zero_landcover_class0=zero_landcover_class0
    )
    # -------------------------
    # integrate to catchment from grids
    # -------------------------
    SSC_catchment_fromgrid = compute_catchment_ssc_from_grids_pergridrunoff(
        model_sed=model_sed,
        df_runoff=df_runoff_a,
        grid_hru_col="HRU_ID",
        runoff_hru_col="hruId",
        runoff_col="averageRoutedRunoff",
        return_wide=False
    )

    if not isinstance(SSC_catchment_fromgrid, pd.DataFrame):
        return (None, None) if return_arrays else None

    sim_df = SSC_catchment_fromgrid.copy()

    # -------------------------
    # route catchment SSC
    # -------------------------
    a_rout = float(param_dict["a_rout"]["value"])
    mt_rout = float(param_dict["mt_rout"]["value"])
    K_rout = int(round(param_dict["K_rout"]["value"]))

    if "index" in sim_df.columns:
        times = pd.to_datetime(
            sim_df["index"].astype(str).str.replace("t_", "", regex=False),
            format="%Y%m%d_%H%M%S"
        )
    elif "time" in sim_df.columns:
        times = pd.to_datetime(sim_df["time"])
    else:
        possible_time_cols = [c for c in sim_df.columns if "time" in c.lower() or c.lower() == "index"]
        if len(possible_time_cols) == 0:
            return (None, None) if return_arrays else None
        col = possible_time_cols[0]
        if col == "index":
            times = pd.to_datetime(
                sim_df[col].astype(str).str.replace("t_", "", regex=False),
                format="%Y%m%d_%H%M%S"
            )
        else:
            times = pd.to_datetime(sim_df[col])

    if len(times) < 2:
        return (None, None) if return_arrays else None

    dt_hours = (times.iloc[1] - times.iloc[0]).total_seconds() / 3600.0

    if "SSC_catchment" not in sim_df.columns:
        possible_ssc_cols = [c for c in sim_df.columns if "ssc" in c.lower()]
        if len(possible_ssc_cols) == 0:
            return (None, None) if return_arrays else None
        sim_df = sim_df.rename(columns={possible_ssc_cols[0]: "SSC_catchment"})

    ssc_in = sim_df["SSC_catchment"].to_numpy(dtype=float)

    ssc_routed = routing_gamma(
        qinst=ssc_in,
        a=a_rout,
        mt=mt_rout,
        K=K_rout,
        dt=dt_hours
    )

    sim_df = sim_df.copy()
    sim_df["time"] = times
    sim_df["SSC_sim"] = ssc_routed
    sim_df["time"] = pd.to_datetime(sim_df["time"]).dt.round("h")

    # -------------------------
    # prepare observed
    # -------------------------
    obs_df = df_SSC_obs.copy()
    obs_df[obs_time_col] = pd.to_datetime(obs_df[obs_time_col]).dt.round("h")

    obs_df = (
        obs_df.groupby(obs_time_col, as_index=False)[obs_value_col]
        .mean()
        .rename(columns={obs_time_col: "time", obs_value_col: "SSC_obs"})
    )

    sim_df = (
        sim_df.groupby("time", as_index=False)["SSC_sim"]
        .mean()
    )

    merged = pd.merge(obs_df, sim_df, on="time", how="inner")

    if merged.empty:
        return (None, None) if return_arrays else merged

    obs = merged["SSC_obs"].to_numpy(dtype=float)
    sim = merged["SSC_sim"].to_numpy(dtype=float)

    valid = np.isfinite(obs) & np.isfinite(sim)

    if valid.sum() == 0:
        return (None, None) if return_arrays else merged.iloc[0:0].copy()

    if return_arrays:
        return obs[valid], sim[valid]

    return merged.loc[valid].reset_index(drop=True)


def evaluate_param_set1b(
    individual,
    param_names,
    base_param_dict,
    model_input,
    df_runoff,
    rain,
    df_swe,                       # 
    df_SSC_obs,
    obs_time_col="time",
    obs_value_col="SSC",
    objective="log_rmse",
    cold_region=True,             # 
    zero_landcover_class0=False,
):
    """
    Evaluate one SedHydro optimisation 1b parameter set.

    Parameters
    ----------
    individual : list
        Candidate parameter values.
    param_names : list
        Names corresponding to individual values.
    base_param_dict : dict
        Base parameter dictionary.
    model_input : pandas.DataFrame
        Spatial model input table.
    df_runoff : pandas.DataFrame
        Runoff time series.
    rain : pandas.DataFrame
        Rainfall time series.
    df_swe : pandas.DataFrame
        Snow water equivalent time series.
    df_SSC_obs : pandas.DataFrame
        Observed SSC time series.
    obs_time_col : str, optional
        Observed time column name.
    obs_value_col : str, optional
        Observed SSC column name.
    objective : str, optional
        Objective function name.
    cold_region : bool, optional
        If True, apply snow attenuation.
    zero_landcover_class0 : bool, optional
        If True, force landcover class 0 to zero SSC.

    Returns
    -------
    tuple
        One-element objective value tuple for DEAP minimisation.
    """

    try:
        individual = repair_individual_with_order_constraints(
            individual=individual,
            param_names=param_names,
            base_param_dict=base_param_dict
        )

        obs, sim = prepare_obs_sim_series1b(
            individual=individual,
            param_names=param_names,
            base_param_dict=base_param_dict,
            model_input=model_input,
            df_runoff=df_runoff,
            rain=rain,
            df_swe=df_swe,                  # CHANGE
            df_SSC_obs=df_SSC_obs,
            obs_time_col=obs_time_col,
            obs_value_col=obs_value_col,
            cold_region=cold_region,        # CHANGE
            zero_landcover_class0=zero_landcover_class0,
        )
        if obs is None or sim is None:
            return (1e12,)

        score = objective_from_series(obs, sim, objective=objective)

        if not np.isfinite(score):
            score = 1e12

        return (score,)

    except Exception as e:
        print(f"Evaluation1b failed: {e}")
        return (1e12,)


def optimise1b_deap(
    param_dict,
    model_input,
    df_runoff,
    rain,
    df_swe=None,
    df_SSC_obs=None,
    obs_time_col="time",
    obs_value_col="SSC",
    objective="log_rmse",
    cold_region=True,
    zero_landcover_class0=False,
    optimize_hill_routing_params=True,
    optimize_only=None,
    n_generations=30,
    population_size=40,
    cxpb=0.6,
    mutpb=0.3,
    eta=20.0,
    seed=42,
    checkpoint_path="optimise1b_deap_checkpoint.pkl",
    early_stop_rounds=None,
    early_stop_tol=1e-4,
):
    """
    Calibrate SedHydro optimisation 1b parameters using DEAP.

    Parameters
    ----------
    param_dict : dict
        Model parameter dictionary.
    model_input : pandas.DataFrame
        Spatial model input table.
    df_runoff : pandas.DataFrame
        Runoff time series.
    rain : pandas.DataFrame
        Rainfall time series.
    df_swe : pandas.DataFrame, optional
        Snow water equivalent time series.
    df_SSC_obs : pandas.DataFrame
        Observed SSC time series.
    obs_time_col : str, optional
        Observed time column name.
    obs_value_col : str, optional
        Observed SSC column name.
    objective : str, optional
        Objective function name.
    cold_region : bool, optional
        If True, apply snow attenuation.
    zero_landcover_class0 : bool, optional
        If True, force landcover class 0 to zero SSC.
    optimize_hill_routing_params : bool, optional
        If True, include hillslope routing parameters.
    optimize_only : list, optional
        Parameter names to optimise. If None, all eligible parameters are used.
    n_generations : int, optional
        Number of DEAP generations.
    population_size : int, optional
        Number of individuals.
    cxpb : float, optional
        Crossover probability.
    mutpb : float, optional
        Mutation probability.
    eta : float, optional
        Mutation distribution index.
    seed : int, optional
        Random seed.
    checkpoint_path : str, optional
        Checkpoint output path.
    early_stop_rounds : int, optional
        Stop after this many generations without improvement.
    early_stop_tol : float, optional
        Minimum improvement threshold for early stopping.

    Returns
    -------
    optimized_param_dict : dict
        Parameter dictionary with optimised values.
    best_score : float
        Best objective value.
    pop : list
        Final DEAP population.
    logbook : deap.tools.Logbook
        Optimisation statistics.
    hof : deap.tools.HallOfFame
        Best individual found.
    """

    random.seed(seed)
    np.random.seed(seed)

    objective = str(objective).lower()
    allowed_objectives = {"rmse", "log_rmse", "mse", "kge", "nkge", "nsh", "nse"}
    if objective not in allowed_objectives:
        raise ValueError(
            "objective must be one of: 'rmse', 'log_rmse', 'mse', 'kge', 'nkge', 'nsh', 'nse'"
        )

    if early_stop_rounds is not None:
        early_stop_rounds = int(early_stop_rounds)
        if early_stop_rounds <= 0:
            raise ValueError("early_stop_rounds must be a positive integer or None.")
        early_stop_tol = float(early_stop_tol)
        if early_stop_tol < 0:
            raise ValueError("early_stop_tol must be >= 0.")

    param_names = [
        k for k, v in param_dict.items()
        if float(v["up"]) > float(v["low"])
    ]

    if not optimize_hill_routing_params:
        param_names = [k for k in param_names if k not in ["a_rout", "mt_rout"]]

    if optimize_only is not None:
        optimize_only = list(optimize_only)

        unknown_params = [k for k in optimize_only if k not in param_dict]
        if len(unknown_params) > 0:
            raise ValueError(
                f"These parameters in optimize_only are not in param_dict: {unknown_params}"
            )

        param_names = [k for k in param_names if k in optimize_only]

    bounds = [(float(param_dict[k]["low"]), float(param_dict[k]["up"])) for k in param_names]

    if len(param_names) == 0:
        raise ValueError(
            "No parameters selected for optimisation. Check optimize_only, bounds, and optimize_hill_routing_params."
        )

    if "FitnessMin1b" not in creator.__dict__:
        creator.create("FitnessMin1b", base.Fitness, weights=(-1.0,))
    if "Individual1b" not in creator.__dict__:
        creator.create("Individual1b", list, fitness=creator.FitnessMin1b)

    toolbox = base.Toolbox()

    def init_individual():
        vals = []
        for low, up in bounds:
            if low == up:
                vals.append(low)
            else:
                vals.append(random.uniform(low, up))
        ind = creator.Individual1b(vals)
        ind = repair_individual_with_order_constraints(
            individual=ind,
            param_names=param_names,
            base_param_dict=param_dict
        )
        return ind

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate",
        evaluate_param_set1b,
        param_names=param_names,
        base_param_dict=param_dict,
        model_input=model_input,
        df_runoff=df_runoff,
        rain=rain,
        df_swe=df_swe,                 
        df_SSC_obs=df_SSC_obs,
        obs_time_col=obs_time_col,
        obs_value_col=obs_value_col,
        objective=objective,
        cold_region=cold_region,       
        zero_landcover_class0=zero_landcover_class0,
    )
    toolbox.register("mate", tools.cxBlend, alpha=0.2)
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        eta=eta,
        low=[b[0] for b in bounds],
        up=[b[1] for b in bounds],
        indpb=0.2
    )
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("mean", np.mean)
    stats.register("std", np.std)

    generation_start_time = time.perf_counter()
    total_start_time = time.perf_counter()
    n_cores_used = 1
    core_type = platform.processor()
    if not core_type:
        core_type = platform.machine()
    total_logical_cores = mp.cpu_count()

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)
    best_score_so_far = hof[0].fitness.values[0]
    stagnant_generations = 0

    def print_generation_status(gen, population):
        best_ind_gen = tools.selBest(population, 1)[0]
        best_score_gen = best_ind_gen.fitness.values[0]

        best_param_dict_gen = copy.deepcopy(param_dict)
        for name, value in zip(param_names, best_ind_gen):
            best_param_dict_gen[name]["value"] = float(value)

        best_param_dict_gen = apply_order_constraints_to_param_dict(best_param_dict_gen)

        a_params = {
            k: v["value"]
            for k, v in best_param_dict_gen.items()
            if k.startswith("a") or k in ["as", "abase"]
        }
        b_params = {
            k: v["value"]
            for k, v in best_param_dict_gen.items()
            if k.startswith("b") or k in ["bs", "bbase"]
        }
        routing_params = {
            k: v["value"]
            for k, v in best_param_dict_gen.items()
            if k in ["a_rout", "mt_rout"]
        }
        routing_params["K_rout"] = param_dict["K_rout"]["value"]
        snow_params = {
            k: v["value"]
            for k, v in best_param_dict_gen.items()
            if k in ["ksnow"]
        }
        generation_elapsed_min = (time.perf_counter() - generation_start_time) / 60.0
        total_elapsed_min = (time.perf_counter() - total_start_time) / 60.0

        print(f"\nGeneration {gen}/{n_generations}")
        print(f"Objective ({objective}) = {best_score_gen}")
        print(f"Elapsed this generation = {generation_elapsed_min:.2f} min")
        print(f"Elapsed total = {total_elapsed_min:.2f} min")
        print(f"Cores used = {n_cores_used} / {total_logical_cores}")
        print(f"Core/CPU type = {core_type}")
        print(f"a params = {a_params}")
        print(f"b params = {b_params}")
        print(f"routing params = {routing_params}")
        print(f"snow params = {snow_params}")  
        if early_stop_rounds is not None:
            print(
                f"Early stopping monitor = {stagnant_generations}/{early_stop_rounds} "
                f"(tol={early_stop_tol})"
            )

    print_generation_status(0, pop)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    record = stats.compile(pop) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    print(logbook.stream)

    save_optimisation_checkpoint(
        output_path=checkpoint_path,
        generation=0,
        param_dict=param_dict,
        param_names=param_names,
        population=pop,
        logbook=logbook,
        hof=hof,
        objective=objective,
        zero_landcover_class0=zero_landcover_class0,
        optimize_hill_routing_params=optimize_hill_routing_params,
        optimize_only=optimize_only
    )

    try:
        for gen in range(1, n_generations + 1):
            generation_start_time = time.perf_counter()

            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            for ind in offspring:
                for j, (low, up) in enumerate(bounds):
                    if ind[j] < low:
                        ind[j] = low
                    elif ind[j] > up:
                        ind[j] = up

                repair_individual_with_order_constraints(
                    individual=ind,
                    param_names=param_names,
                    base_param_dict=param_dict
                )

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring
            hof.update(pop)

            current_best_score = hof[0].fitness.values[0]
            improvement = best_score_so_far - current_best_score

            if improvement > early_stop_tol:
                best_score_so_far = current_best_score
                stagnant_generations = 0
            else:
                stagnant_generations += 1

            record = stats.compile(pop) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(logbook.stream)

            print_generation_status(gen, pop)

            save_optimisation_checkpoint(
                output_path=checkpoint_path,
                generation=gen,
                param_dict=param_dict,
                param_names=param_names,
                population=pop,
                logbook=logbook,
                hof=hof,
                objective=objective,
                zero_landcover_class0=zero_landcover_class0,
                optimize_hill_routing_params=optimize_hill_routing_params,
                optimize_only=optimize_only
            )

            if early_stop_rounds is not None and stagnant_generations >= early_stop_rounds:
                print(
                    f"\nEarly stopping triggered at generation {gen}: "
                    f"best objective did not improve by more than {early_stop_tol} "
                    f"for {early_stop_rounds} consecutive generations."
                )
                break

    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        print(f"Last fully completed generation saved at: {checkpoint_path}")
        raise

    best_ind = hof[0]
    best_score = hof[0].fitness.values[0]

    optimized_param_dict = copy.deepcopy(param_dict)
    for name, value in zip(param_names, best_ind):
        optimized_param_dict[name]["value"] = float(value)

    optimized_param_dict = apply_order_constraints_to_param_dict(optimized_param_dict)

    return optimized_param_dict, best_score, pop, logbook, hof

# functions for save_optimisation_results1b


# required function from utils for: save_optimisation_results1b




#% Sum SSC for each HRU from grids [this codes is also copied in optimisation_updated.py]
def compute_hru_ssc_from_grids_pergridrunoff(
    model_sed,
    df_runoff,
    grid_hru_col="HRU_ID",
    runoff_hru_col="hruId",
    runoff_col="averageRoutedRunoff",
    return_wide=True
):
    """
    Compute HRU SSC from grid SSC values and return a dataframe like:

        HRU_ID   t_20200420_000000   t_20200420_010000   ...

    Parameters
    ----------
    model_sed : pd.DataFrame
        Grid-based dataframe with one row per grid and time columns like:
        ..., HRU_ID, t_20240115_010000, t_20240115_020000, ...

    df_runoff : pd.DataFrame
        Long dataframe with columns:
        time, hruId, averageRoutedRunoff

    grid_hru_col : str, default="HRU_ID"
        HRU column name in model_sed.

    runoff_hru_col : str, default="hruId"
        HRU column name in df_runoff.

    runoff_col : str, default="averageRoutedRunoff"
        Runoff column name in df_runoff.

    return_wide : bool, default=True
        If True, return wide dataframe:
            HRU_ID, t_YYYYMMDD_HHMMSS, ...
        If False, return long dataframe:
            HRU_ID, time, SSC_hru

    Returns
    -------
    pd.DataFrame
        HRU SSC time series.

    Notes
    -----
    Because runoff is only available at HRU level, converting to per-grid runoff gives:

        runoff_grid = runoff_hru / n_grids_in_hru

    Since all grids inside the same HRU get the same runoff_grid at a given time,
    the runoff-weighted SSC within each HRU becomes the arithmetic mean of SSC
    across grids in that HRU.
    """

    # 1) identify timestep columns
    time_cols = [c for c in model_sed.columns if c.startswith("t_")]

    # 2) count grids per HRU
    hru_grid_counts = (
        model_sed.groupby(grid_hru_col)
        .size()
        .reset_index(name="n_grids")
    )

    # 3) grid SSC wide -> long
    ssc_long = model_sed.melt(
        id_vars=[grid_hru_col],
        value_vars=time_cols,
        var_name="time_col",
        value_name="SSC"
    )

    ssc_long["time"] = pd.to_datetime(
        ssc_long["time_col"].str.replace("t_", "", regex=False),
        format="%Y%m%d_%H%M%S"
    )

    ssc_long = ssc_long[[grid_hru_col, "time", "SSC"]]

    # 4) runoff long table
    runoff_long = df_runoff[["time", runoff_hru_col, runoff_col]].rename(
        columns={runoff_hru_col: grid_hru_col, runoff_col: "runoff_hru"}
    ).copy()

    runoff_long["time"] = pd.to_datetime(runoff_long["time"])

    # 5) attach number of grids and compute per-grid runoff
    runoff_long = runoff_long.merge(hru_grid_counts, on=grid_hru_col, how="inner")
    runoff_long["runoff_grid"] = runoff_long["runoff_hru"] / runoff_long["n_grids"]

    # 6) merge SSC and per-grid runoff
    df = ssc_long.merge(
        runoff_long[[grid_hru_col, "time", "runoff_grid"]],
        on=[grid_hru_col, "time"],
        how="inner"
    )

    # 7) runoff-weighted HRU SSC
    SSC_hru_long = (
        df.groupby([grid_hru_col, "time"])
        .apply(
            lambda g: 0.0
            if g["runoff_grid"].sum() == 0
            else (g["SSC"] * g["runoff_grid"]).sum() / g["runoff_grid"].sum(),
            include_groups=False
        )
        .reset_index(name="SSC_hru")
    )

    if return_wide:
        SSC_hru = (
            SSC_hru_long
            .pivot(index=grid_hru_col, columns="time", values="SSC_hru")
            .reset_index()
        )

        SSC_hru.columns.name = None
        SSC_hru = SSC_hru.rename_axis(None, axis=1)

        # rename datetime columns to t_YYYYMMDD_HHMMSS
        SSC_hru.columns = [
            grid_hru_col if c == grid_hru_col else f"t_{c.strftime('%Y%m%d_%H%M%S')}"
            for c in SSC_hru.columns
        ]

        return SSC_hru

    return SSC_hru_long

#%
def route_ssc_hru_gamma(SSC_hru, a, mt, K, hru_col="HRU_ID"):
    """
    Route SSC time series for each HRU using gamma routing.

    Parameters
    ----------
    SSC_hru : pd.DataFrame
        Wide dataframe with one row per HRU and timestep columns like:
        HRU_ID, t_20200101_010000, t_20200101_020000, ...

    a : float
        Shape parameter of gamma distribution.

    mt : float
        Mean travel time / delay of the gamma routing function.
        Must use the same time unit as dt (e.g., hours if dt is hours).

    K : int
        Routing length in number of time steps used in convolution.

    hru_col : str, default="HRU_ID"
        Name of HRU identifier column.

    Returns
    -------
    SSC_hru_routed : pd.DataFrame
        Same-style dataframe as SSC_hru, with routed SSC values.

    Notes
    -----
    The routing is applied independently to each HRU row:

        SSC_routed_j(t) = sum_k [ w(k) * SSC_j(t-k) ]

    where w(k) are discrete gamma-distribution routing weights.

    The timestep dt is inferred from the first two timestep columns:
        dt = time[1] - time[0]

    dt is converted to hours before passing to routing_gamma, so mt should
    also be provided in hours.
    """

    # 1) identify timestep columns
    time_cols = [c for c in SSC_hru.columns if c.startswith("t_")]
    if len(time_cols) < 2:
        raise ValueError("SSC_hru must contain at least two timestep columns.")

    # 2) infer dt from first two timestep columns
    times = pd.to_datetime(
        [c.replace("t_", "") for c in time_cols],
        format="%Y%m%d_%H%M%S"
    )
    dt_hours = (times[1] - times[0]).total_seconds() / 3600.0

    # 3) copy structure
    SSC_hru_routed = SSC_hru[[hru_col] + time_cols].copy()

    # 4) route each HRU row
    ssc_values = SSC_hru[time_cols].to_numpy(dtype=float)
    routed_values = np.zeros_like(ssc_values, dtype=float)

    for i in range(ssc_values.shape[0]):
        routed_values[i, :] = routing_gamma(
            qinst=ssc_values[i, :],
            a=a,
            mt=mt,
            K=K,
            dt=dt_hours
        )

    # 5) put back into dataframe
    SSC_hru_routed.loc[:, time_cols] = routed_values

    return SSC_hru_routed

#%
def create_ssc_hru_fraction_dict(
    SSC_hru,
    sand_hru_stat,
    silt_hru_stat,
    hru_col="HRU_ID",
    sand_col="mean_sand",
    silt_col="mean_silt",
    number_fractions=3,
    clip_fractions=True
):
    """
    Create a dict of SSC DataFrames by sediment fraction at HRU level.

    For the 3-fraction case:
        clay = 100 - (silt + sand)

    Fractions are converted to decimals and multiplied by SSC_hru
    for each HRU and each timestep.

    Parameters
    ----------
    SSC_hru : pandas.DataFrame
        HRU-level SSC table with one HRU ID column and timestep columns.
    sand_hru_stat : pandas.DataFrame
        Per-HRU sand statistics table containing hru_col and sand_col.
    silt_hru_stat : pandas.DataFrame
        Per-HRU silt statistics table containing hru_col and silt_col.
    hru_col : str, default "HRU_ID"
        HRU ID column name.
    sand_col : str, default "mean_sand"
        Column in sand_hru_stat holding sand percentage.
    silt_col : str, default "mean_silt"
        Column in silt_hru_stat holding silt percentage.
    number_fractions : int, default 3
        Number of fractions to create. Currently implemented for 3.
    clip_fractions : bool, default True
        If True, clip clay/silt/sand fractions to [0, 100].

    Returns
    -------
    SSC_hru_frac : dict[int, pandas.DataFrame]
        Dict of fraction-specific SSC DataFrames.
        For 3 fractions:
            0 = clay
            1 = silt
            2 = sand
        For each time step starting with t_
    """
    import numpy as np
    import pandas as pd

    if number_fractions != 3:
        raise NotImplementedError(
            "This function currently supports number_fractions=3 only "
            "(clay, silt, sand)."
        )

    if hru_col not in SSC_hru.columns:
        raise KeyError(f"'{hru_col}' not found in SSC_hru columns.")

    if hru_col not in sand_hru_stat.columns:
        raise KeyError(f"'{hru_col}' not found in sand_hru_stat columns.")

    if hru_col not in silt_hru_stat.columns:
        raise KeyError(f"'{hru_col}' not found in silt_hru_stat columns.")

    if sand_col not in sand_hru_stat.columns:
        raise KeyError(f"'{sand_col}' not found in sand_hru_stat columns.")

    if silt_col not in silt_hru_stat.columns:
        raise KeyError(f"'{silt_col}' not found in silt_hru_stat columns.")

    # Identify timestep columns
    time_cols = [c for c in SSC_hru.columns if c != hru_col]

    # Build one HRU fraction table aligned to SSC_hru
    frac_df = (
        SSC_hru[[hru_col]]
        .merge(
            sand_hru_stat[[hru_col, sand_col]].rename(columns={sand_col: "sand_pct"}),
            on=hru_col,
            how="left"
        )
        .merge(
            silt_hru_stat[[hru_col, silt_col]].rename(columns={silt_col: "silt_pct"}),
            on=hru_col,
            how="left"
        )
    )

    # Missing fractions -> 0
    frac_df["sand_pct"] = frac_df["sand_pct"].fillna(0.0)
    frac_df["silt_pct"] = frac_df["silt_pct"].fillna(0.0)

    # Clay from remainder
    frac_df["clay_pct"] = 100.0 - (frac_df["silt_pct"] + frac_df["sand_pct"])

    if clip_fractions:
        frac_df["sand_pct"] = frac_df["sand_pct"].clip(0, 100)
        frac_df["silt_pct"] = frac_df["silt_pct"].clip(0, 100)
        frac_df["clay_pct"] = frac_df["clay_pct"].clip(0, 100)

    # Convert % to fractions
    frac_df["clay_frac"] = frac_df["clay_pct"] / 100.0
    frac_df["silt_frac"] = frac_df["silt_pct"] / 100.0
    frac_df["sand_frac"] = frac_df["sand_pct"] / 100.0

    # Create output dict
    ni = number_fractions
    SSC_hru_frac = {
        i: pd.DataFrame(index=SSC_hru.index, columns=SSC_hru.columns, dtype=float)
        for i in range(ni)
    }

    # Keep HRU_ID in each dataframe
    for i in range(ni):
        SSC_hru_frac[i][hru_col] = SSC_hru[hru_col].values

    # Multiply SSC by each fraction
    SSC_hru_frac[0][time_cols] = (
        SSC_hru[time_cols].to_numpy()
        * frac_df["clay_frac"].to_numpy()[:, None]
    )

    SSC_hru_frac[1][time_cols] = (
        SSC_hru[time_cols].to_numpy()
        * frac_df["silt_frac"].to_numpy()[:, None]
    )

    SSC_hru_frac[2][time_cols] = (
        SSC_hru[time_cols].to_numpy()
        * frac_df["sand_frac"].to_numpy()[:, None]
    )

    return SSC_hru_frac

#%

def save_optimisation_results1b(
    optimised_param_dict,
    best_score,
    pop,
    logbook,
    hof,
    model_input,
    df_runoff,
    rain,
    df_swe=None,
    cat_hru=None,
    df_SSC_obs=None,
    output_dir=None,
    file_name="optimised_parameters",
    obs_time_col="time",
    obs_value_col="SSC",
    cold_region=True,
    zero_landcover_class0=False,
    sand_hru_stat=None,
    silt_hru_stat=None,
    number_fractions=3,
    generation_history=None,
    population_history=None
):
    """
    Save optimisation1b results and final model outputs.

    Saved outputs
    -------------
    Main summary
    - {file_name}.csv
    - {file_name}.pkl

    Final model outputs
    - {file_name}_model_sed_final.pkl
    - {file_name}_model_sed_final.csv
    - {file_name}_model_sed_final_raster.nc

    HRU SSC outputs
    - {file_name}_SSC_hru.pkl
    - {file_name}_SSC_hru.csv

    HRU SSC fraction outputs
    - {file_name}_SSC_hru_frac.pkl
    - {file_name}_SSC_hru_frac_frac1.csv
    - ...

    Optimisation histories
    - {file_name}_generation_history.csv
    - {file_name}_generation_history.pkl
    - {file_name}_population_history.csv
    - {file_name}_population_history.pkl
    """

    import os
    import pickle
    import numpy as np
    import pandas as pd
    from netCDF4 import Dataset

    os.makedirs(output_dir, exist_ok=True)

    csv_file = os.path.join(output_dir, f"{file_name}.csv")
    pkl_file = os.path.join(output_dir, f"{file_name}.pkl")

    model_sed_pkl_file = os.path.join(output_dir, f"{file_name}_model_sed_final.pkl")
    model_sed_csv_file = os.path.join(output_dir, f"{file_name}_model_sed_final.csv")
    model_sed_nc_file = os.path.join(output_dir, f"{file_name}_model_sed_final_raster.nc")

    ssc_hru_pkl_file = os.path.join(output_dir, f"{file_name}_SSC_hru.pkl")
    ssc_hru_csv_file = os.path.join(output_dir, f"{file_name}_SSC_hru.csv")

    ssc_hru_frac_pkl_file = os.path.join(output_dir, f"{file_name}_SSC_hru_frac.pkl")

    generation_history_csv_file = os.path.join(output_dir, f"{file_name}_generation_history.csv")
    generation_history_pkl_file = os.path.join(output_dir, f"{file_name}_generation_history.pkl")

    population_history_csv_file = os.path.join(output_dir, f"{file_name}_population_history.csv")
    population_history_pkl_file = os.path.join(output_dir, f"{file_name}_population_history.pkl")

    def save_model_sed_to_raster_netcdf(model_sed_df, time_cols_in, output_nc):
        required_cols = ["row", "col"]
        missing_cols = [c for c in required_cols if c not in model_sed_df.columns]
        if missing_cols:
            raise ValueError(
                f"model_sed_df must contain columns {required_cols}. Missing: {missing_cols}"
            )

        time_values = pd.to_datetime(
            [c.replace("t_", "") for c in time_cols_in],
            format="%Y%m%d_%H%M%S"
        )
        n_time = len(time_values)

        row_vals = model_sed_df["row"].to_numpy(dtype=int)
        col_vals = model_sed_df["col"].to_numpy(dtype=int)

        row_min = int(np.min(row_vals))
        row_max = int(np.max(row_vals))
        col_min = int(np.min(col_vals))
        col_max = int(np.max(col_vals))

        n_y = row_max - row_min + 1
        n_x = col_max - col_min + 1

        row_idx = row_vals - row_min
        col_idx = col_vals - col_min

        ssc_cube = np.full((n_time, n_y, n_x), np.nan, dtype=np.float32)
        ssc_values = model_sed_df[time_cols_in].to_numpy(dtype=np.float32)

        for i in range(len(model_sed_df)):
            ssc_cube[:, row_idx[i], col_idx[i]] = ssc_values[i, :]

        grid_id_2d = None
        if "grid_id" in model_sed_df.columns:
            grid_id_2d = np.full((n_y, n_x), -9999, dtype=np.int32)
            gid_vals = model_sed_df["grid_id"].to_numpy(dtype=np.int32)
            for i in range(len(model_sed_df)):
                grid_id_2d[row_idx[i], col_idx[i]] = gid_vals[i]

        hru_id_2d = None
        if "HRU_ID" in model_sed_df.columns:
            hru_id_2d = np.full((n_y, n_x), -9999, dtype=np.int32)
            hru_vals = model_sed_df["HRU_ID"].to_numpy(dtype=np.int32)
            for i in range(len(model_sed_df)):
                hru_id_2d[row_idx[i], col_idx[i]] = hru_vals[i]

        with Dataset(output_nc, "w", format="NETCDF4") as ds:
            ds.createDimension("time", n_time)
            ds.createDimension("y", n_y)
            ds.createDimension("x", n_x)

            time_var = ds.createVariable("time", str, ("time",))
            y_var = ds.createVariable("y", "i4", ("y",))
            x_var = ds.createVariable("x", "i4", ("x",))

            ssc_var = ds.createVariable(
                "SSC_grid",
                "f4",
                ("time", "y", "x"),
                zlib=True,
                complevel=4,
                fill_value=np.float32(np.nan)
            )

            time_var[:] = np.array(
                [t.strftime("%Y-%m-%d %H:%M:%S") for t in time_values],
                dtype=object
            )
            y_var[:] = np.arange(row_min, row_max + 1, dtype=np.int32)
            x_var[:] = np.arange(col_min, col_max + 1, dtype=np.int32)

            ssc_var[:, :, :] = ssc_cube
            ssc_var.long_name = "Grid suspended sediment concentration"
            ssc_var.units = "same_as_model_output"

            if grid_id_2d is not None:
                grid_id_var = ds.createVariable(
                    "grid_id",
                    "i4",
                    ("y", "x"),
                    zlib=True,
                    complevel=4,
                    fill_value=-9999
                )
                grid_id_var[:, :] = grid_id_2d
                grid_id_var.long_name = "Grid ID"

            if hru_id_2d is not None:
                hru_id_var = ds.createVariable(
                    "HRU_ID",
                    "i4",
                    ("y", "x"),
                    zlib=True,
                    complevel=4,
                    fill_value=-9999
                )
                hru_id_var[:, :] = hru_id_2d
                hru_id_var.long_name = "HRU ID"

            ds.description = "Final model_sed grid SSC as raster-style NetCDF"
            ds.history = "Created by save_optimisation_results1b"

    obs_sim_df = prepare_obs_sim_series1b(
        param_dict=optimised_param_dict,
        model_input=model_input,
        df_runoff=df_runoff,
        rain=rain,
        df_swe=df_swe,                  # CHANGE
        df_SSC_obs=df_SSC_obs,
        obs_time_col=obs_time_col,
        obs_value_col=obs_value_col,
        cold_region=cold_region,        # CHANGE
        zero_landcover_class0=zero_landcover_class0
    )
    model_sed_final1b = build_model_sed_from_params(
        optimised_param_dict,
        model_input
    )

    df_runoff_a, rain_a, common_times = align_forcing_data(df_runoff, rain)

    model_sed_final1b, time_cols = add_time_columns_to_model(
        model_sed_final1b,
        common_times
    )

    model_sed_final1b = calculate_grid_ssc(
        model_sed_final1b,
        df_runoff_a,
        rain_a,
        time_cols,
        zero_landcover_class0=zero_landcover_class0
    )
    model_sed_final1b = calculate_grid_ssc(
        model_sed_final1b,
        df_runoff_a,
        rain_a,
        time_cols,
        df_swe=df_swe,                  # CHANGE
        cold_region=cold_region,        # CHANGE
        zero_landcover_class0=zero_landcover_class0
    )

    model_sed_final1b.to_pickle(model_sed_pkl_file)
    model_sed_final1b.to_csv(model_sed_csv_file, index=False)
    save_model_sed_to_raster_netcdf(
        model_sed_df=model_sed_final1b,
        time_cols_in=time_cols,
        output_nc=model_sed_nc_file
    )

    from utils import compute_hru_ssc_from_grids_pergridrunoff
    from utils import route_ssc_hru_gamma
    from utils import create_ssc_hru_fraction_dict

    SSC_hru_final1b = compute_hru_ssc_from_grids_pergridrunoff(
        model_sed=model_sed_final1b,
        df_runoff=df_runoff,
        grid_hru_col="HRU_ID",
        runoff_hru_col="hruId",
        runoff_col="averageRoutedRunoff",
        return_wide=True
    )

    a_rout = optimised_param_dict["a_rout"]["value"]
    mt_rout = optimised_param_dict["mt_rout"]["value"]
    K_rout = optimised_param_dict["K_rout"]["value"]

    SSC_hru = route_ssc_hru_gamma(
        SSC_hru=SSC_hru_final1b,
        a=a_rout,
        mt=mt_rout,
        K=K_rout,
        hru_col="HRU_ID"
    )

    SSC_hru.to_pickle(ssc_hru_pkl_file)
    SSC_hru.to_csv(ssc_hru_csv_file, index=False)

    SSC_hru_frac = None
    if sand_hru_stat is not None and silt_hru_stat is not None:
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

        with open(ssc_hru_frac_pkl_file, "wb") as f:
            pickle.dump(SSC_hru_frac, f)

        for frac_id, frac_df in SSC_hru_frac.items():
            frac_csv = os.path.join(
                output_dir,
                f"{file_name}_SSC_hru_frac_frac{frac_id}.csv"
            )
            frac_df.to_csv(frac_csv)

    param_rows = []
    for k, v in optimised_param_dict.items():
        param_rows.append({
            "section": "parameters",
            "name": k,
            "value": v.get("value"),
            "low": v.get("low"),
            "up": v.get("up")
        })

    df_params = pd.DataFrame(param_rows)

    df_score = pd.DataFrame([{
        "section": "best_score",
        "name": "objective_value",
        "value": best_score
    }])

    best_ind = hof[0]
    df_best_ind = pd.DataFrame([
        {
            "section": "best_individual",
            "name": f"param_{i}",
            "value": val
        }
        for i, val in enumerate(best_ind)
    ])

    pop_rows = []
    for i, ind in enumerate(pop):
        pop_rows.append({
            "section": "population_final",
            "name": f"ind_{i}",
            "value": ind.fitness.values[0] if len(ind.fitness.values) > 0 else None
        })

    df_pop = pd.DataFrame(pop_rows)

    log_rows = []
    for record in logbook:
        log_rows.append({
            "section": "logbook",
            "generation": record.get("gen"),
            "nevals": record.get("nevals"),
            "min": record.get("min"),
            "mean": record.get("mean"),
            "std": record.get("std")
        })

    df_log = pd.DataFrame(log_rows)

    if obs_sim_df is None:
        df_obs_sim_out = pd.DataFrame([{
            "section": "obs_sim",
            "name": "no_data"
        }])
    else:
        df_obs_sim_out = obs_sim_df.copy()
        df_obs_sim_out["section"] = "obs_sim"

    df_all = pd.concat(
        [df_params, df_score, df_best_ind, df_pop, df_log, df_obs_sim_out],
        ignore_index=True,
        sort=False
    )

    df_all.to_csv(csv_file, index=False)

    generation_history_df = pd.DataFrame(generation_history) if generation_history is not None else pd.DataFrame()
    population_history_df = pd.DataFrame(population_history) if population_history is not None else pd.DataFrame()

    if not generation_history_df.empty:
        generation_history_df.to_csv(generation_history_csv_file, index=False)
        with open(generation_history_pkl_file, "wb") as f:
            pickle.dump(generation_history_df, f)

    if not population_history_df.empty:
        population_history_df.to_csv(population_history_csv_file, index=False)
        with open(population_history_pkl_file, "wb") as f:
            pickle.dump(population_history_df, f)

    with open(pkl_file, "wb") as f:
        pickle.dump(
            {
                "optimised_param_dict": optimised_param_dict,
                "best_score": best_score,
                "pop": pop,
                "logbook": logbook,
                "hof": hof,
                "obs_sim_df": obs_sim_df,
                "model_sed_final1b": model_sed_final1b,
                "SSC_hru": SSC_hru,
                "SSC_hru_frac": SSC_hru_frac,
                "generation_history": generation_history_df,
                "population_history": population_history_df,
            },
            f
        )

    print(f"Saved CSV: {csv_file}")
    print(f"Saved PKL: {pkl_file}")
    print(f"Saved final model_sed PKL: {model_sed_pkl_file}")
    print(f"Saved final model_sed CSV: {model_sed_csv_file}")
    print(f"Saved final model_sed raster NetCDF: {model_sed_nc_file}")
    print(f"Saved SSC_hru PKL: {ssc_hru_pkl_file}")
    print(f"Saved SSC_hru CSV: {ssc_hru_csv_file}")

    if not generation_history_df.empty:
        print(f"Saved generation history CSV: {generation_history_csv_file}")
        print(f"Saved generation history PKL: {generation_history_pkl_file}")

    if not population_history_df.empty:
        print(f"Saved population history CSV: {population_history_csv_file}")
        print(f"Saved population history PKL: {population_history_pkl_file}")

    if SSC_hru_frac is not None:
        print(f"Saved SSC_hru_frac PKL: {ssc_hru_frac_pkl_file}")
        for frac_id in SSC_hru_frac.keys():
            print(
                "Saved SSC_hru_frac CSV: "
                + os.path.join(output_dir, f"{file_name}_SSC_hru_frac_frac{frac_id}.csv")
            )


#%% ==================
# multiprocessing for optimisation1b
#   ==================


def optimise1b_deap_mp(
    param_dict,
    model_input,
    df_runoff,
    rain,
    df_swe=None,
    df_SSC_obs=None,
    obs_time_col="time",
    obs_value_col="SSC",
    objective="log_rmse",
    cold_region=True,
    zero_landcover_class0=True,
    optimize_hill_routing_params=True,
    optimize_only=None,
    n_generations=30,
    population_size=40,
    cxpb=0.6,
    mutpb=0.3,
    eta=20.0,
    seed=42,
    checkpoint_path="optimise1b_deap_mp_checkpoint.pkl",
    early_stop=True,
    early_stop_rounds=None,
    early_stop_tol=1e-4,
    n_cores=None,
    chunksize=1,
):
    """
    Multiprocessing version of optimise1b_deap.

    Parameters
    ----------
    param_dict : dict
        Model parameter dictionary.
    model_input : pandas.DataFrame
        Spatial model input table.
    df_runoff : pandas.DataFrame
        Runoff time series.
    rain : pandas.DataFrame
        Rainfall time series.
    df_swe : pandas.DataFrame, optional
        Snow water equivalent time series.
    df_SSC_obs : pandas.DataFrame
        Observed SSC time series.
    obs_time_col : str, optional
        Observed time column name.
    obs_value_col : str, optional
        Observed SSC column name.
    objective : str, optional
        Objective function name.
    cold_region : bool, optional
        If True, apply snow attenuation.
    zero_landcover_class0 : bool, optional
        If True, force landcover class 0 to zero SSC.
    optimize_hill_routing_params : bool, optional
        If True, include hillslope routing parameters.
    optimize_only : list, optional
        Parameter names to optimise. If None, all eligible parameters are used.
    n_generations : int, optional
        Number of DEAP generations.
    population_size : int, optional
        Number of individuals.
    cxpb : float, optional
        Crossover probability.
    mutpb : float, optional
        Mutation probability.
    eta : float, optional
        Mutation distribution index.
    seed : int, optional
        Random seed.
    checkpoint_path : str, optional
        Checkpoint output path.
    early_stop : bool, optional
        If True, enable early stopping.
    early_stop_rounds : int, optional
        Stop after this many generations without improvement.
    early_stop_tol : float, optional
        Minimum improvement threshold for early stopping.
    n_cores : int, optional
        Number of CPU cores to use.
    chunksize : int, optional
        Multiprocessing chunk size.

    Returns
    -------
    optimized_param_dict : dict
        Parameter dictionary with optimised values.
    best_score : float
        Best objective value.
    pop : list
        Final DEAP population.
    logbook : deap.tools.Logbook
        Optimisation statistics.
    hof : deap.tools.HallOfFame
        Best individual found.
    generation_history : list
        Best parameter and fitness history by generation.
    population_history : list
        Parameter and fitness history for all individuals.
    """

    random.seed(seed)
    np.random.seed(seed)

    objective = str(objective).lower()
    allowed_objectives = {"rmse", "log_rmse", "mse", "kge", "nkge", "nsh", "nse"}
    if objective not in allowed_objectives:
        raise ValueError(
            "objective must be one of: 'rmse', 'log_rmse', 'mse', 'kge', 'nkge', 'nsh', 'nse'"
        )

    early_stop = bool(early_stop)

    if early_stop:
        if early_stop_rounds is None:
            raise ValueError(
                "When early_stop=True, early_stop_rounds must be provided."
            )
        early_stop_rounds = int(early_stop_rounds)
        if early_stop_rounds <= 0:
            raise ValueError("early_stop_rounds must be a positive integer.")
        early_stop_tol = float(early_stop_tol)
        if early_stop_tol < 0:
            raise ValueError("early_stop_tol must be >= 0.")
    else:
        early_stop_rounds = None
        early_stop_tol = None

    param_names = [
        k for k, v in param_dict.items()
        if float(v["up"]) > float(v["low"])
    ]

    if not optimize_hill_routing_params:
        param_names = [k for k in param_names if k not in ["a_rout", "mt_rout"]]

    if optimize_only is not None:
        optimize_only = list(optimize_only)

        unknown_params = [k for k in optimize_only if k not in param_dict]
        if len(unknown_params) > 0:
            raise ValueError(
                f"These parameters in optimize_only are not in param_dict: {unknown_params}"
            )

        param_names = [k for k in param_names if k in optimize_only]

    bounds = [(float(param_dict[k]["low"]), float(param_dict[k]["up"])) for k in param_names]

    if len(param_names) == 0:
        raise ValueError(
            "No parameters selected for optimisation. Check optimize_only, bounds, and optimize_hill_routing_params."
        )

    if "FitnessMin1bMP" not in creator.__dict__:
        creator.create("FitnessMin1bMP", base.Fitness, weights=(-1.0,))
    if "Individual1bMP" not in creator.__dict__:
        creator.create("Individual1bMP", list, fitness=creator.FitnessMin1bMP)

    toolbox = base.Toolbox()

    def init_individual():
        vals = []
        for low, up in bounds:
            if low == up:
                vals.append(low)
            else:
                vals.append(random.uniform(low, up))
        ind = creator.Individual1bMP(vals)
        ind = repair_individual_with_order_constraints(
            individual=ind,
            param_names=param_names,
            base_param_dict=param_dict
        )
        return ind

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate",
        evaluate_param_set1b,
        param_names=param_names,
        base_param_dict=param_dict,
        model_input=model_input,
        df_runoff=df_runoff,
        rain=rain,
        df_swe=df_swe,                  # 
        df_SSC_obs=df_SSC_obs,
        obs_time_col=obs_time_col,
        obs_value_col=obs_value_col,
        objective=objective,
        cold_region=cold_region,        # 
        zero_landcover_class0=zero_landcover_class0,
    )
    toolbox.register("mate", tools.cxBlend, alpha=0.2)
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        eta=eta,
        low=[b[0] for b in bounds],
        up=[b[1] for b in bounds],
        indpb=0.2
    )
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("mean", np.mean)
    stats.register("std", np.std)

    generation_start_time = time.perf_counter()
    total_start_time = time.perf_counter()

    core_type = platform.processor()
    if not core_type:
        core_type = platform.machine()
    total_logical_cores = mp.cpu_count()

    if n_cores is None:
        n_cores_used = max(1, total_logical_cores - 1)
    else:
        n_cores_used = max(1, min(int(n_cores), total_logical_cores))

    generation_history = []
    population_history = []

    def record_population_history(generation, population):
        rows = []
        for ind_idx, ind in enumerate(population):
            row = {
                "generation": generation,
                "individual_id": ind_idx,
                "fitness": ind.fitness.values[0] if ind.fitness.valid else np.nan,
            }
            for name, value in zip(param_names, ind):
                row[name] = float(value)
            rows.append(row)
        return rows

    def record_generation_history(generation, population):
        best_ind_gen = tools.selBest(population, 1)[0]
        best_score_gen = best_ind_gen.fitness.values[0]

        row = {
            "generation": generation,
            "best_score": best_score_gen,
        }

        stat_record = stats.compile(population)
        for k, v in stat_record.items():
            row[f"fitness_{k}"] = float(v)

        for name, value in zip(param_names, best_ind_gen):
            row[name] = float(value)

        best_param_dict_gen = copy.deepcopy(param_dict)
        for name, value in zip(param_names, best_ind_gen):
            best_param_dict_gen[name]["value"] = float(value)

        best_param_dict_gen = apply_order_constraints_to_param_dict(best_param_dict_gen)

        for k in param_dict.keys():
            row[f"full_{k}"] = best_param_dict_gen[k]["value"]

        return row

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=n_cores_used)

    def parallel_map(func, iterable):
        return pool.map(func, iterable, chunksize)

    toolbox.register("map", parallel_map)

    try:
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        hof.update(pop)
        best_score_so_far = hof[0].fitness.values[0]
        stagnant_generations = 0

        generation_history.append(record_generation_history(0, pop))
        population_history.extend(record_population_history(0, pop))

        def print_generation_status(gen, population):
            best_ind_gen = tools.selBest(population, 1)[0]
            best_score_gen = best_ind_gen.fitness.values[0]

            best_param_dict_gen = copy.deepcopy(param_dict)
            for name, value in zip(param_names, best_ind_gen):
                best_param_dict_gen[name]["value"] = float(value)

            best_param_dict_gen = apply_order_constraints_to_param_dict(best_param_dict_gen)

            a_params = {
                k: v["value"]
                for k, v in best_param_dict_gen.items()
                if k.startswith("a") or k in ["as", "abase"]
            }
            b_params = {
                k: v["value"]
                for k, v in best_param_dict_gen.items()
                if k.startswith("b") or k in ["bs", "bbase"]
            }
            routing_params = {
                k: v["value"]
                for k, v in best_param_dict_gen.items()
                if k in ["a_rout", "mt_rout"]
            }
            snow_params = {
                k: v["value"]
                for k, v in best_param_dict_gen.items()
                if k in ["ksnow"]
            }
            routing_params["K_rout"] = param_dict["K_rout"]["value"]

            generation_elapsed_min = (time.perf_counter() - generation_start_time) / 60.0
            total_elapsed_min = (time.perf_counter() - total_start_time) / 60.0

            print(f"\nGeneration {gen}/{n_generations}")
            print(f"Objective ({objective}) = {best_score_gen}")
            print(f"Elapsed this generation = {generation_elapsed_min:.2f} min")
            print(f"Elapsed total = {total_elapsed_min:.2f} min")
            print(f"Cores used = {n_cores_used} / {total_logical_cores}")
            print(f"Core/CPU type = {core_type}")
            print(f"a params = {a_params}")
            print(f"b params = {b_params}")
            print(f"routing params = {routing_params}")
            print(f"snow params = {snow_params}") 
            if early_stop:
                print(
                    f"Early stopping monitor = {stagnant_generations}/{early_stop_rounds} "
                    f"(tol={early_stop_tol})"
                )

        print_generation_status(0, pop)

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

        record = stats.compile(pop) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        print(logbook.stream)

        save_optimisation_checkpoint(
            output_path=checkpoint_path,
            generation=0,
            param_dict=param_dict,
            param_names=param_names,
            population=pop,
            logbook=logbook,
            hof=hof,
            objective=objective,
            zero_landcover_class0=zero_landcover_class0,
            optimize_hill_routing_params=optimize_hill_routing_params,
            optimize_only=optimize_only,
            generation_history=generation_history,
            population_history=population_history
        )

        for gen in range(1, n_generations + 1):
            generation_start_time = time.perf_counter()

            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            for ind in offspring:
                for j, (low, up) in enumerate(bounds):
                    if ind[j] < low:
                        ind[j] = low
                    elif ind[j] > up:
                        ind[j] = up

                repair_individual_with_order_constraints(
                    individual=ind,
                    param_names=param_names,
                    base_param_dict=param_dict
                )

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring
            hof.update(pop)

            if early_stop:
                current_best_score = hof[0].fitness.values[0]
                improvement = best_score_so_far - current_best_score

                if improvement > early_stop_tol:
                    best_score_so_far = current_best_score
                    stagnant_generations = 0
                else:
                    stagnant_generations += 1

            generation_history.append(record_generation_history(gen, pop))
            population_history.extend(record_population_history(gen, pop))

            record = stats.compile(pop) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(logbook.stream)

            print_generation_status(gen, pop)

            save_optimisation_checkpoint(
                output_path=checkpoint_path,
                generation=gen,
                param_dict=param_dict,
                param_names=param_names,
                population=pop,
                logbook=logbook,
                hof=hof,
                objective=objective,
                zero_landcover_class0=zero_landcover_class0,
                optimize_hill_routing_params=optimize_hill_routing_params,
                optimize_only=optimize_only,
                generation_history=generation_history,
                population_history=population_history
            )

            if early_stop and stagnant_generations >= early_stop_rounds:
                print(
                    f"\nEarly stopping triggered at generation {gen}: "
                    f"best objective did not improve by more than {early_stop_tol} "
                    f"for {early_stop_rounds} consecutive generations."
                )
                break

    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        print(f"Last fully completed generation saved at: {checkpoint_path}")
        raise

    finally:
        pool.close()
        pool.join()

    best_ind = hof[0]
    best_score = hof[0].fitness.values[0]

    optimized_param_dict = copy.deepcopy(param_dict)
    for name, value in zip(param_names, best_ind):
        optimized_param_dict[name]["value"] = float(value)

    optimized_param_dict = apply_order_constraints_to_param_dict(optimized_param_dict)

    return (
        optimized_param_dict,
        best_score,
        pop,
        logbook,
        hof,
        generation_history,
        population_history
    )

#%% ==================
# Optimisation 2
#   ==================

#% optimisation of TempSedRout
# import sys
# main_dir_routing='/Users/armanhaddadchi/Library/CloudStorage/OneDrive-UniversityofCalgary/ErosionModel/TempSedRout'
# sys.path.append(main_dir_routing)

# =========================================================
# TempSedRout path
# =========================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPSEDROUT_DIR = os.path.join(PROJECT_DIR, "TempSedRout")

if TEMPSEDROUT_DIR not in sys.path:
    sys.path.insert(0, TEMPSEDROUT_DIR)
    
# =========================================================
# helpers
# =========================================================
def save_optimisation_checkpoint2(
    output_path,
    generation,
    param_dict,
    param_names,
    population,
    logbook,
    hof,
    objective,
    optimize_only=None,
    generation_history=None,
    population_history=None
):
    """
    Save optimisation checkpoint for optimise2_deap / optimise2_deap_mp.
    """

    best_ind = hof[0] if len(hof) > 0 else None
    best_score = None

    if best_ind is not None and best_ind.fitness.valid:
        best_score = best_ind.fitness.values[0]

    optimized_param_dict = copy.deepcopy(param_dict)
    if best_ind is not None:
        for name, value in zip(param_names, best_ind):
            optimized_param_dict[name]["value"] = float(value)

    checkpoint = {
        "generation": generation,
        "optimized_param_dict": optimized_param_dict,
        "best_score": best_score,
        "pop": population,
        "logbook": logbook,
        "hof": hof,
        "objective": objective,
        "optimize_only": optimize_only,
        "generation_history": generation_history,
        "population_history": population_history,
    }

    with open(output_path, "wb") as f:
        pickle.dump(checkpoint, f)

# =========================================================
# TempSedRout-specific preparation
# =========================================================
def find_outlet_reach_id(river_gdf):
    """
    Identify the outlet reach in a river network.

    Parameters
    ----------
    river_gdf : geopandas.GeoDataFrame
        River network containing LINKNO and DSLINKNO attributes.

    Returns
    -------
    int
        LINKNO of the outlet reach, defined as the reach whose
        downstream link is not present in the network.
    """
    link_set = set(pd.to_numeric(river_gdf["LINKNO"], errors="coerce").dropna().astype(int))
    ds_set = set(pd.to_numeric(river_gdf["DSLINKNO"], errors="coerce").dropna().astype(int))

    outlet_candidates = list(ds_set - link_set)
    if len(outlet_candidates) > 0:
        outlet_ds = outlet_candidates[0]
        reach_out_id = river_gdf.loc[river_gdf["DSLINKNO"] == outlet_ds, "LINKNO"].iloc[0]
        return int(reach_out_id)

    # fallback
    reach_out_id = river_gdf.loc[
        ~river_gdf["DSLINKNO"].isin(river_gdf["LINKNO"]),
        "LINKNO"
    ].iloc[0]
    return int(reach_out_id)


def prepare_obs_sim_series_tempsedrout(
    param_dict,
    river_gdf,
    sediment_size,
    toml_file,
    h,
    q,
    Q,
    width,
    SSC_hru_frac,
    df_SSC_obs,
    obs_time_col="time",
    obs_value_col="SSC",
    use_storage=False,
    river_storage=None,
    storage_data_type="length"
):
    from TempSedRout_function import TempSedRout
    from TempSedRout_storage_function import TempSedRout_storage
    
    """
    Run TempSedRout and prepare aligned observed/simulated SSC series.

    Parameters
    ----------
    param_dict : dict
        TempSedRout parameter dictionary.
    river_gdf : geopandas.GeoDataFrame
        River network.
    sediment_size : pandas.DataFrame
        Sediment size class table.
    toml_file : str
        TempSedRout configuration file.
    h, q, Q, width : pandas.DataFrame
        Hydraulic input time series.
    SSC_hru_frac : dict
        HRU sediment concentration by size fraction.
    df_SSC_obs : pandas.DataFrame
        Observed SSC time series.
    obs_time_col : str, optional
        Observed time column name.
    obs_value_col : str, optional
        Observed SSC column name.
    use_storage : bool, optional
        If True, use TempSedRout_storage.
    river_storage : pandas.DataFrame, optional
        Storage/depression system input table.
    storage_data_type : str, optional
        Type of storage input data.

    Returns
    -------
    obs : numpy.ndarray
        Aligned observed SSC values.
    sim : numpy.ndarray
        Aligned simulated SSC values at the outlet reach.
    """

    # copy param dict to avoid in-place overwrite
    p = copy.deepcopy(param_dict)

    dispersion_coeff1 = float(p["dispers1_TempSedRout"]["value"])
    dispersion_coeff2 = float(p["dispers2_TempSedRout"]["value"])
    dispersion_coeff3 = float(p["dispers3_TempSedRout"]["value"])
    
    median_diam = float(p["median_diam_TempSedRout"]["value"])
    SF = float(p["SF_TempSedRout"]["value"])
    interpolation_numbers = int(round(p["interp_TempSedRout"]["value"]))
    
    Fd1_coeff = float(p["Fd1_TempSedRout"]["value"])
    Fd2_coeff = float(p["Fd2_TempSedRout"]["value"])
    Fd3_coeff = float(p["Fd3_TempSedRout"]["value"])
    
    cr1_coeff = float(p["cr1_TempSedRout"]["value"])
    cr2_coeff = float(p["cr2_TempSedRout"]["value"])
    cr3_coeff = float(p["cr3_TempSedRout"]["value"])
    
    storage_coef_fl = float(p.get("fl_storage", {"value": 1.0})["value"])
    storage_coef_fh = float(p.get("fh_storage", {"value": 1.0})["value"])
    storage_coef_fw = float(p.get("fw_storage", {"value": 1.0})["value"])
    storage_coef_fa = float(p.get("fa_storage", {"value": 1.0})["value"])
    
    if use_storage:
        SSC_river_frac_out, SSC_river_tot_out = TempSedRout_storage(
            river_gdf=river_gdf,
            sediment_size=sediment_size,
            config_file=toml_file,
            h=h,
            q=q,
            Q=Q,
            width=width,
            SSC_hru_frac=SSC_hru_frac,
            dispersion_coeff1=dispersion_coeff1,
            dispersion_coeff2=dispersion_coeff2,
            dispersion_coeff3=dispersion_coeff3,
            median_diam=median_diam,
            SF=SF,
            interpolation_numbers=interpolation_numbers,
            Fd1_coeff=Fd1_coeff,
            Fd2_coeff=Fd2_coeff,
            Fd3_coeff=Fd3_coeff,
            cr1_coeff=cr1_coeff,
            cr2_coeff=cr2_coeff,
            cr3_coeff=cr3_coeff,
            river_storage=river_storage,
            storage_data_type=storage_data_type,
            storage_coef_fl=storage_coef_fl,
            storage_coef_fh=storage_coef_fh,
            storage_coef_fw=storage_coef_fw,
            storage_coef_fa=storage_coef_fa
        )
    else:
        SSC_river_frac_out, SSC_river_tot_out = TempSedRout(
            river_gdf=river_gdf,
            sediment_size=sediment_size,
            config_file=toml_file,
            h=h,
            q=q,
            Q=Q,
            width=width,
            SSC_hru_frac=SSC_hru_frac,
            dispersion_coeff1=dispersion_coeff1,
            dispersion_coeff2=dispersion_coeff2,
            dispersion_coeff3=dispersion_coeff3,
            median_diam=median_diam,
            SF=SF,
            interpolation_numbers=interpolation_numbers,
            Fd1_coeff=Fd1_coeff,
            Fd2_coeff=Fd2_coeff,
            Fd3_coeff=Fd3_coeff,
            cr1_coeff=cr1_coeff,
            cr2_coeff=cr2_coeff,
            cr3_coeff=cr3_coeff
        )

    reach_out_id = find_outlet_reach_id(river_gdf)

    if reach_out_id not in SSC_river_tot_out.columns:
        return None, None

    sim_df = SSC_river_tot_out[[reach_out_id]].copy().reset_index()
    sim_df.columns = ["time", "SSC_sim"]
    sim_df["time"] = pd.to_datetime(sim_df["time"]).dt.round("h")

    obs_df = df_SSC_obs.copy()

    if obs_time_col not in obs_df.columns:
        raise ValueError(f"obs_time_col='{obs_time_col}' not found in df_SSC_obs")

    if obs_value_col not in obs_df.columns:
        # fallback: first numeric column other than time
        numeric_cols = [c for c in obs_df.columns if c != obs_time_col and pd.api.types.is_numeric_dtype(obs_df[c])]
        if len(numeric_cols) == 0:
            raise ValueError(f"obs_value_col='{obs_value_col}' not found and no numeric fallback found in df_SSC_obs")
        obs_value_col = numeric_cols[0]

    obs_df[obs_time_col] = pd.to_datetime(obs_df[obs_time_col]).dt.round("h")

    obs_df = (
        obs_df.groupby(obs_time_col, as_index=False)[obs_value_col]
        .mean()
        .rename(columns={obs_time_col: "time", obs_value_col: "SSC_obs"})
    )

    sim_df = (
        sim_df.groupby("time", as_index=False)["SSC_sim"]
        .mean()
    )

    merged = pd.merge(obs_df, sim_df, on="time", how="inner")

    if merged.empty:
        return None, None

    obs = merged["SSC_obs"].to_numpy(dtype=float)
    sim = merged["SSC_sim"].to_numpy(dtype=float)

    valid = np.isfinite(obs) & np.isfinite(sim)
    if valid.sum() == 0:
        return None, None

    return obs[valid], sim[valid]


def evaluate_param_set_tempsedrout(
    individual,
    param_names,
    base_param_dict,
    river_gdf,
    sediment_size,
    toml_file,
    h,
    q,
    Q,
    width,
    SSC_hru_frac,
    df_SSC_obs,
    obs_time_col="time",
    obs_value_col="SSC",
    objective="log_rmse",
    use_storage=False,
    river_storage=None,
    storage_data_type="length",
):
    """
    DEAP evaluation function for TempSedRout optimisation.
    """

    try:
        param_dict_eval = copy.deepcopy(base_param_dict)
        for name, value in zip(param_names, individual):
            param_dict_eval[name]["value"] = float(value)

        obs, sim = prepare_obs_sim_series_tempsedrout(
            param_dict=param_dict_eval,
            river_gdf=river_gdf,
            sediment_size=sediment_size,
            toml_file=toml_file,
            h=h,
            q=q,
            Q=Q,
            width=width,
            SSC_hru_frac=SSC_hru_frac,
            df_SSC_obs=df_SSC_obs,
            obs_time_col=obs_time_col,
            obs_value_col=obs_value_col,
            use_storage=use_storage,
            river_storage=river_storage,
            storage_data_type=storage_data_type,
        )

        if obs is None or sim is None:
            return (1e12,)

        score = objective_from_series(obs, sim, objective=objective)

        if not np.isfinite(score):
            score = 1e12

        return (score,)

    except Exception as e:
        print(f"TempSedRout evaluation failed: {e}")
        return (1e12,)

def run_final_tempsedrout(
    param_dict,
    river_gdf,
    sediment_size,
    toml_file,
    h,
    q,
    Q,
    width,
    SSC_hru_frac,
    use_storage=False,
    river_storage=None,
    storage_data_type="length"
):
    """
    Run TempSedRout once using a supplied parameter dictionary and
    return the full routed outputs.

    Returns
    -------
    SSC_river_frac_out : dict
        Fraction-wise routed SSC outputs from TempSedRout.

    SSC_river_tot_out : pd.DataFrame
        Total routed SSC for all reaches and all times.
    """
    from TempSedRout_function import TempSedRout
    from TempSedRout_storage_function import TempSedRout_storage

    p = copy.deepcopy(param_dict)

    dispersion_coeff1 = float(p["dispers1_TempSedRout"]["value"])
    dispersion_coeff2 = float(p["dispers2_TempSedRout"]["value"])
    dispersion_coeff3 = float(p["dispers3_TempSedRout"]["value"])

    median_diam = float(p["median_diam_TempSedRout"]["value"])
    SF = float(p["SF_TempSedRout"]["value"])
    interpolation_numbers = int(round(p["interp_TempSedRout"]["value"]))

    Fd1_coeff = float(p["Fd1_TempSedRout"]["value"])
    Fd2_coeff = float(p["Fd2_TempSedRout"]["value"])
    Fd3_coeff = float(p["Fd3_TempSedRout"]["value"])

    cr1_coeff = float(p["cr1_TempSedRout"]["value"])
    cr2_coeff = float(p["cr2_TempSedRout"]["value"])
    cr3_coeff = float(p["cr3_TempSedRout"]["value"])
    
    storage_coef_fl = float(p.get("fl_storage", {"value": 1.0})["value"])
    storage_coef_fh = float(p.get("fh_storage", {"value": 1.0})["value"])
    storage_coef_fw = float(p.get("fw_storage", {"value": 1.0})["value"])
    storage_coef_fa = float(p.get("fa_storage", {"value": 1.0})["value"])
        
        
    if use_storage:
        SSC_river_frac_out, SSC_river_tot_out = TempSedRout_storage(
            river_gdf=river_gdf,
            sediment_size=sediment_size,
            config_file=toml_file,
            h=h,
            q=q,
            Q=Q,
            width=width,
            SSC_hru_frac=SSC_hru_frac,
            dispersion_coeff1=dispersion_coeff1,
            dispersion_coeff2=dispersion_coeff2,
            dispersion_coeff3=dispersion_coeff3,
            median_diam=median_diam,
            SF=SF,
            interpolation_numbers=interpolation_numbers,
            Fd1_coeff=Fd1_coeff,
            Fd2_coeff=Fd2_coeff,
            Fd3_coeff=Fd3_coeff,
            cr1_coeff=cr1_coeff,
            cr2_coeff=cr2_coeff,
            cr3_coeff=cr3_coeff,
            river_storage=river_storage,
            storage_data_type=storage_data_type,
            storage_coef_fl=storage_coef_fl,
            storage_coef_fh=storage_coef_fh,
            storage_coef_fw=storage_coef_fw,
            storage_coef_fa=storage_coef_fa
        )
    else:
        SSC_river_frac_out, SSC_river_tot_out = TempSedRout(
            river_gdf=river_gdf,
            sediment_size=sediment_size,
            config_file=toml_file,
            h=h,
            q=q,
            Q=Q,
            width=width,
            SSC_hru_frac=SSC_hru_frac,
            dispersion_coeff1=dispersion_coeff1,
            dispersion_coeff2=dispersion_coeff2,
            dispersion_coeff3=dispersion_coeff3,
            median_diam=median_diam,
            SF=SF,
            interpolation_numbers=interpolation_numbers,
            Fd1_coeff=Fd1_coeff,
            Fd2_coeff=Fd2_coeff,
            Fd3_coeff=Fd3_coeff,
            cr1_coeff=cr1_coeff,
            cr2_coeff=cr2_coeff,
            cr3_coeff=cr3_coeff
        )

    return SSC_river_frac_out, SSC_river_tot_out

# =========================================================
# DEAP optimisation
# =========================================================
def optimise2_deap(
    param_dict,
    river_gdf,
    sediment_size,
    toml_file,
    h,
    q,
    Q,
    width,
    SSC_hru_frac,
    df_SSC_obs,
    use_storage=False,
    river_storage=None,
    storage_data_type="length",
    obs_time_col="time",
    obs_value_col="SSC",
    objective="log_rmse",
    optimize_only=None,
    n_generations=30,
    population_size=40,
    cxpb=0.6,
    mutpb=0.3,
    eta=20.0,
    seed=42,
    checkpoint_path="optimise2_deap_checkpoint.pkl",
    early_stop_rounds=None,
    early_stop_tol=1e-4
):
    """
    Calibrate TempSedRout parameters using DEAP.

    Parameters
    ----------
    param_dict : dict
        Model parameter dictionary.
    river_gdf : geopandas.GeoDataFrame
        River network.
    sediment_size : pandas.DataFrame
        Sediment size class table.
    toml_file : str
        TempSedRout configuration file.
    h, q, Q, width : pandas.DataFrame
        Hydraulic input time series.
    SSC_hru_frac : dict
        HRU sediment concentration by size fraction.
    df_SSC_obs : pandas.DataFrame
        Observed SSC time series.
    use_storage : bool, optional
        If True, use TempSedRout_storage.
    river_storage : pandas.DataFrame, optional
        Storage/depression system input table.
    storage_data_type : str, optional
        Type of storage input data.
    obs_time_col : str, optional
        Observed time column name.
    obs_value_col : str, optional
        Observed SSC column name.
    objective : str, optional
        Objective function name.
    optimize_only : list, optional
        Parameter names to optimise. If None, all eligible routing parameters are used.
    n_generations : int, optional
        Number of DEAP generations.
    population_size : int, optional
        Number of individuals.
    cxpb : float, optional
        Crossover probability.
    mutpb : float, optional
        Mutation probability.
    eta : float, optional
        Mutation distribution index.
    seed : int, optional
        Random seed.
    checkpoint_path : str, optional
        Checkpoint output path.
    early_stop_rounds : int, optional
        Stop after this many generations without improvement.
    early_stop_tol : float, optional
        Minimum improvement threshold for early stopping.

    Returns
    -------
    optimized_param_dict : dict
        Parameter dictionary with optimised values.
    best_score : float
        Best objective value.
    pop : list
        Final DEAP population.
    logbook : deap.tools.Logbook
        Optimisation statistics.
    hof : deap.tools.HallOfFame
        Best individual found.
    """

    random.seed(seed)
    np.random.seed(seed)

    objective = str(objective).lower()
    allowed_objectives = {"rmse", "log_rmse", "mse", "kge", "nkge", "nsh", "nse"}
    if objective not in allowed_objectives:
        raise ValueError(
            "objective must be one of: 'rmse', 'log_rmse', 'mse', 'kge', 'nkge', 'nsh', 'nse'"
        )

    if early_stop_rounds is not None:
        early_stop_rounds = int(early_stop_rounds)
        if early_stop_rounds <= 0:
            raise ValueError("early_stop_rounds must be a positive integer or None.")
        early_stop_tol = float(early_stop_tol)
        if early_stop_tol < 0:
            raise ValueError("early_stop_tol must be >= 0.")

    # >>> optimize_only: added
    allowed_param_names = [
        "dispers1_TempSedRout",
        "dispers2_TempSedRout",
        "dispers3_TempSedRout",
        "Fd1_TempSedRout",
        "Fd2_TempSedRout",
        "Fd3_TempSedRout",
        "cr1_TempSedRout",
        "cr2_TempSedRout",
        "cr3_TempSedRout",
    ]
    
    if use_storage:
        allowed_param_names += [
            "fl_storage",
            "fh_storage",
            "fw_storage",
            "fa_storage",
        ]

    missing = [k for k in allowed_param_names if k not in param_dict]
    if missing:
        raise KeyError(f"These required parameters are missing from param_dict: {missing}")

    if optimize_only is None:
        param_names = [
            k for k in allowed_param_names
            if float(param_dict[k]["up"]) > float(param_dict[k]["low"])
        ]
        optimize_only_out = None
    else:
        optimize_only = list(optimize_only)

        invalid = [k for k in optimize_only if k not in allowed_param_names]
        if invalid:
            raise ValueError(
                f"These optimize_only parameters are not allowed: {invalid}"
            )

        param_names = [
            k for k in optimize_only
            if float(param_dict[k]["up"]) > float(param_dict[k]["low"])
        ]
        optimize_only_out = optimize_only
    # >>> optimize_only: finished

    bounds = [(float(param_dict[k]["low"]), float(param_dict[k]["up"])) for k in param_names]

    if "FitnessMin2" not in creator.__dict__:
        creator.create("FitnessMin2", base.Fitness, weights=(-1.0,))
    if "Individual2" not in creator.__dict__:
        creator.create("Individual2", list, fitness=creator.FitnessMin2)

    toolbox = base.Toolbox()

    def init_individual():
        vals = []
        for low, up in bounds:
            if low == up:
                vals.append(low)
            else:
                vals.append(random.uniform(low, up))
        return creator.Individual2(vals)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate",
        evaluate_param_set_tempsedrout,
        param_names=param_names,
        base_param_dict=param_dict,
        river_gdf=river_gdf,
        sediment_size=sediment_size,
        toml_file=toml_file,
        h=h,
        q=q,
        Q=Q,
        width=width,
        SSC_hru_frac=SSC_hru_frac,
        df_SSC_obs=df_SSC_obs,
        obs_time_col=obs_time_col,
        obs_value_col=obs_value_col,
        objective=objective,
        use_storage=use_storage,
        river_storage=river_storage,
        storage_data_type=storage_data_type,
    )

    toolbox.register("mate", tools.cxBlend, alpha=0.2)
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        eta=eta,
        low=[b[0] for b in bounds],
        up=[b[1] for b in bounds],
        indpb=0.2
    )
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("mean", np.mean)
    stats.register("std", np.std)

    generation_start_time = time.perf_counter()
    total_start_time = time.perf_counter()
    n_cores_used = 1
    core_type = platform.processor() or platform.machine()
    total_logical_cores = mp.cpu_count()

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)

    best_score_so_far = hof[0].fitness.values[0]
    stagnant_generations = 0

    def print_generation_status(gen, population):
        best_ind_gen = tools.selBest(population, 1)[0]
        best_score_gen = best_ind_gen.fitness.values[0]

        best_param_dict_gen = copy.deepcopy(param_dict)
        for name, value in zip(param_names, best_ind_gen):
            best_param_dict_gen[name]["value"] = float(value)

        generation_elapsed_min = (time.perf_counter() - generation_start_time) / 60.0
        total_elapsed_min = (time.perf_counter() - total_start_time) / 60.0

        routed_params = {k: best_param_dict_gen[k]["value"] for k in allowed_param_names}

        print(f"\nGeneration {gen}/{n_generations}")
        print(f"Objective ({objective}) = {best_score_gen}")
        print(f"Elapsed this generation = {generation_elapsed_min:.2f} min")
        print(f"Elapsed total = {total_elapsed_min:.2f} min")
        print(f"Cores used = {n_cores_used} / {total_logical_cores}")
        print(f"Core/CPU type = {core_type}")
        print(f"TempSedRout params = {routed_params}")
        print(f"use_storage = {use_storage}")
        print(f"storage_data_type = {storage_data_type}")
        print(f"Optimized only = {param_names if optimize_only_out is not None else 'all eligible parameters'}")

        if early_stop_rounds is not None:
            print(
                f"Early stopping monitor = {stagnant_generations}/{early_stop_rounds} "
                f"(tol={early_stop_tol})"
            )

    print_generation_status(0, pop)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    record = stats.compile(pop) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    print(logbook.stream)

    save_optimisation_checkpoint2(
        output_path=checkpoint_path,
        generation=0,
        param_dict=param_dict,
        param_names=param_names,
        population=pop,
        logbook=logbook,
        hof=hof,
        objective=objective,
        optimize_only=optimize_only_out
    )

    try:
        for gen in range(1, n_generations + 1):
            generation_start_time = time.perf_counter()

            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            for ind in offspring:
                for j, (low, up) in enumerate(bounds):
                    if ind[j] < low:
                        ind[j] = low
                    elif ind[j] > up:
                        ind[j] = up

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring
            hof.update(pop)

            current_best_score = hof[0].fitness.values[0]
            improvement = best_score_so_far - current_best_score  # minimization

            if early_stop_rounds is not None:
                current_best_score = hof[0].fitness.values[0]
                improvement = best_score_so_far - current_best_score
            
                if improvement > early_stop_tol:
                    best_score_so_far = current_best_score
                    stagnant_generations = 0
                else:
                    stagnant_generations += 1

            record = stats.compile(pop) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(logbook.stream)

            print_generation_status(gen, pop)

            save_optimisation_checkpoint2(
                output_path=checkpoint_path,
                generation=gen,
                param_dict=param_dict,
                param_names=param_names,
                population=pop,
                logbook=logbook,
                hof=hof,
                objective=objective,
                optimize_only=optimize_only_out
            )

            if early_stop_rounds is not None and stagnant_generations >= early_stop_rounds:
                print(
                    f"\nEarly stopping triggered at generation {gen}: "
                    f"best objective did not improve by more than {early_stop_tol} "
                    f"for {early_stop_rounds} consecutive generations."
                )
                break
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        print(f"Last fully completed generation saved at: {checkpoint_path}")
        raise

    best_ind = hof[0]
    best_score = hof[0].fitness.values[0]

    optimized_param_dict = copy.deepcopy(param_dict)
    for name, value in zip(param_names, best_ind):
        optimized_param_dict[name]["value"] = float(value)

    return optimized_param_dict, best_score, pop, logbook, hof

#
def save_optimisation_results2(
    optimised_param_dict,
    best_score,
    pop,
    logbook,
    hof,
    river_gdf,
    sediment_size,
    toml_file,
    h,
    q,
    Q,
    width,
    SSC_hru_frac,
    df_SSC_obs,
    output_dir,
    file_name="optimised2_parameters",
    obs_time_col="time",
    obs_value_col="SSC",
    use_storage=False,
    river_storage=None,
    storage_data_type="length",
    generation_history=None,
    population_history=None
):
    """
    Save SedHydro optimisation 2 results and TempSedRout outputs.

    Parameters
    ----------
    optimised_param_dict : dict
        Parameter dictionary with optimised values.
    best_score : float
        Best objective value.
    pop : list
        Final DEAP population.
    logbook : deap.tools.Logbook
        Optimisation statistics.
    hof : deap.tools.HallOfFame
        Best individual found.
    river_gdf : geopandas.GeoDataFrame
        River network.
    sediment_size : pandas.DataFrame
        Sediment size class table.
    toml_file : str
        TempSedRout configuration file.
    h, q, Q, width : pandas.DataFrame
        Hydraulic input time series.
    SSC_hru_frac : dict
        HRU sediment concentration by size fraction.
    df_SSC_obs : pandas.DataFrame
        Observed SSC time series.
    output_dir : str
        Directory for saved outputs.
    file_name : str, optional
        Base output filename.
    obs_time_col : str, optional
        Observed time column name.
    obs_value_col : str, optional
        Observed SSC column name.
    use_storage : bool, optional
        If True, use TempSedRout_storage.
    river_storage : pandas.DataFrame, optional
        Storage/depression system input table.
    storage_data_type : str, optional
        Type of storage input data.
    generation_history : list, optional
        Best parameter and fitness history by generation.
    population_history : list, optional
        Parameter and fitness history for all individuals.

    Returns
    -------
    None
        Saves CSV and PKL files to output_dir.
    """

    import os
    import pickle
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    csv_file = os.path.join(output_dir, f"{file_name}_main.csv")
    pkl_file = os.path.join(output_dir, f"{file_name}_main.pkl")

    obs_sim_csv_file = os.path.join(output_dir, f"{file_name}_obs_sim.csv")
    obs_sim_pkl_file = os.path.join(output_dir, f"{file_name}_obs_sim.pkl")

    ssc_tot_csv_file = os.path.join(output_dir, f"{file_name}_SSC_river_tot_out.csv")
    ssc_tot_pkl_file = os.path.join(output_dir, f"{file_name}_SSC_river_tot_out.pkl")

    ssc_frac_pkl_file = os.path.join(output_dir, f"{file_name}_SSC_river_frac_out.pkl")

    generation_history_csv_file = os.path.join(output_dir, f"{file_name}_generation_history.csv")
    generation_history_pkl_file = os.path.join(output_dir, f"{file_name}_generation_history.pkl")

    population_history_csv_file = os.path.join(output_dir, f"{file_name}_population_history.csv")
    population_history_pkl_file = os.path.join(output_dir, f"{file_name}_population_history.pkl")

    SSC_river_frac_out, SSC_river_tot_out = run_final_tempsedrout(
        param_dict=optimised_param_dict,
        river_gdf=river_gdf,
        sediment_size=sediment_size,
        toml_file=toml_file,
        h=h,
        q=q,
        Q=Q,
        width=width,
        SSC_hru_frac=SSC_hru_frac,
        use_storage=use_storage,
        river_storage=river_storage,
        storage_data_type=storage_data_type
    )

    obs, sim = prepare_obs_sim_series_tempsedrout(
        param_dict=optimised_param_dict,
        river_gdf=river_gdf,
        sediment_size=sediment_size,
        toml_file=toml_file,
        h=h,
        q=q,
        Q=Q,
        width=width,
        SSC_hru_frac=SSC_hru_frac,
        df_SSC_obs=df_SSC_obs,
        obs_time_col=obs_time_col,
        obs_value_col=obs_value_col,
        use_storage=use_storage,
        river_storage=river_storage,
        storage_data_type=storage_data_type
    )

    if obs is None or sim is None:
        obs_sim_df = None
    else:
        obs_df = df_SSC_obs.copy()

        if obs_time_col not in obs_df.columns:
            raise ValueError(f"obs_time_col='{obs_time_col}' not found in df_SSC_obs")

        if obs_value_col not in obs_df.columns:
            numeric_cols = [
                c for c in obs_df.columns
                if c != obs_time_col and pd.api.types.is_numeric_dtype(obs_df[c])
            ]
            if len(numeric_cols) == 0:
                raise ValueError(
                    f"obs_value_col='{obs_value_col}' not found and no numeric fallback found in df_SSC_obs"
                )
            obs_value_col = numeric_cols[0]

        obs_df[obs_time_col] = pd.to_datetime(obs_df[obs_time_col]).dt.round("h")

        obs_df = (
            obs_df.groupby(obs_time_col, as_index=False)[obs_value_col]
            .mean()
            .rename(columns={obs_time_col: "time", obs_value_col: "SSC_obs"})
        )

        reach_out_id = find_outlet_reach_id(river_gdf)

        if isinstance(SSC_river_tot_out, pd.DataFrame) and reach_out_id in SSC_river_tot_out.columns:
            sim_df = SSC_river_tot_out[[reach_out_id]].copy().reset_index()
            sim_df.columns = ["time", "SSC_sim"]
            sim_df["time"] = pd.to_datetime(sim_df["time"]).dt.round("h")

            sim_df = (
                sim_df.groupby("time", as_index=False)["SSC_sim"]
                .mean()
            )

            obs_sim_df = pd.merge(obs_df, sim_df, on="time", how="inner")
        else:
            obs_sim_df = None

    param_rows = []
    for k, v in optimised_param_dict.items():
        param_rows.append({
            "section": "parameters",
            "name": k,
            "value": v.get("value"),
            "low": v.get("low"),
            "up": v.get("up")
        })

    df_params = pd.DataFrame(param_rows)

    df_score = pd.DataFrame([{
        "section": "best_score",
        "name": "objective_value",
        "value": best_score
    }])

    best_ind = hof[0]
    df_best_ind = pd.DataFrame([
        {
            "section": "best_individual",
            "name": f"param_{i}",
            "value": val
        }
        for i, val in enumerate(best_ind)
    ])

    pop_rows = []
    for i, ind in enumerate(pop):
        pop_rows.append({
            "section": "population_final",
            "name": f"ind_{i}",
            "value": ind.fitness.values[0] if len(ind.fitness.values) > 0 else None
        })

    df_pop = pd.DataFrame(pop_rows)

    log_rows = []
    for record in logbook:
        log_rows.append({
            "section": "logbook",
            "generation": record.get("gen"),
            "nevals": record.get("nevals"),
            "min": record.get("min"),
            "mean": record.get("mean"),
            "std": record.get("std")
        })

    df_log = pd.DataFrame(log_rows)

    if obs_sim_df is None or obs_sim_df.empty:
        df_obs_sim_out = pd.DataFrame([{
            "section": "obs_sim",
            "name": "no_data"
        }])
    else:
        df_obs_sim_out = obs_sim_df.copy()
        df_obs_sim_out["section"] = "obs_sim"

    df_all = pd.concat(
        [df_params, df_score, df_best_ind, df_pop, df_log, df_obs_sim_out],
        ignore_index=True,
        sort=False
    )

    df_all.to_csv(csv_file, index=False)

    generation_history_df = pd.DataFrame(generation_history) if generation_history is not None else pd.DataFrame()
    population_history_df = pd.DataFrame(population_history) if population_history is not None else pd.DataFrame()

    if not generation_history_df.empty:
        generation_history_df.to_csv(generation_history_csv_file, index=False)
        with open(generation_history_pkl_file, "wb") as f:
            pickle.dump(generation_history_df, f)

    if not population_history_df.empty:
        population_history_df.to_csv(population_history_csv_file, index=False)
        with open(population_history_pkl_file, "wb") as f:
            pickle.dump(population_history_df, f)

    with open(pkl_file, "wb") as f:
        pickle.dump(
            {
                "optimised_param_dict": optimised_param_dict,
                "best_score": best_score,
                "pop": pop,
                "logbook": logbook,
                "hof": hof,
                "obs_sim_df": obs_sim_df,
                "SSC_river_frac_out": SSC_river_frac_out,
                "SSC_river_tot_out": SSC_river_tot_out,
                "generation_history": generation_history_df,
                "population_history": population_history_df
            },
            f
        )

    if obs_sim_df is not None and not obs_sim_df.empty:
        obs_sim_df.to_csv(obs_sim_csv_file, index=False)
        with open(obs_sim_pkl_file, "wb") as f:
            pickle.dump(obs_sim_df, f)

    if isinstance(SSC_river_tot_out, pd.DataFrame):
        SSC_river_tot_out.to_csv(ssc_tot_csv_file, index=True)
        with open(ssc_tot_pkl_file, "wb") as f:
            pickle.dump(SSC_river_tot_out, f)

    with open(ssc_frac_pkl_file, "wb") as f:
        pickle.dump(SSC_river_frac_out, f)

    if isinstance(SSC_river_frac_out, dict):
        for frac_name, frac_obj in SSC_river_frac_out.items():
            frac_safe = str(frac_name).replace(" ", "_").replace("/", "_")
            frac_csv_file = os.path.join(
                output_dir,
                f"{file_name}_SSC_river_frac_out_{frac_safe}.csv"
            )
            frac_pkl_file = os.path.join(
                output_dir,
                f"{file_name}_SSC_river_frac_out_{frac_safe}.pkl"
            )

            if isinstance(frac_obj, pd.DataFrame):
                frac_obj.to_csv(frac_csv_file, index=True)
                with open(frac_pkl_file, "wb") as f:
                    pickle.dump(frac_obj, f)
            else:
                with open(frac_pkl_file, "wb") as f:
                    pickle.dump(frac_obj, f)

    print(f"Saved CSV: {csv_file}")
    print(f"Saved PKL: {pkl_file}")

    if not generation_history_df.empty:
        print(f"Saved generation history CSV: {generation_history_csv_file}")
        print(f"Saved generation history PKL: {generation_history_pkl_file}")

    if not population_history_df.empty:
        print(f"Saved population history CSV: {population_history_csv_file}")
        print(f"Saved population history PKL: {population_history_pkl_file}")

    if obs_sim_df is not None and not obs_sim_df.empty:
        print(f"Saved obs/sim CSV: {obs_sim_csv_file}")
        print(f"Saved obs/sim PKL: {obs_sim_pkl_file}")

    if isinstance(SSC_river_tot_out, pd.DataFrame):
        print(f"Saved SSC_river_tot_out CSV: {ssc_tot_csv_file}")
        print(f"Saved SSC_river_tot_out PKL: {ssc_tot_pkl_file}")

    print(f"Saved SSC_river_frac_out PKL: {ssc_frac_pkl_file}")


#%% ==================
# multiprocessing for optimisation2
#   ==================

def optimise2_deap_mp(
    param_dict,
    river_gdf,
    sediment_size,
    toml_file,
    h,
    q,
    Q,
    width,
    SSC_hru_frac,
    df_SSC_obs,
    use_storage=False,
    river_storage=None,
    storage_data_type="length",
    obs_time_col="time",
    obs_value_col="SSC",
    objective="log_rmse",
    optimize_only=None,
    n_generations=30,
    population_size=40,
    cxpb=0.6,
    mutpb=0.3,
    eta=20.0,
    seed=42,
    checkpoint_path="optimise2_deap_mp_checkpoint.pkl",
    early_stop=True,
    early_stop_rounds=None,
    early_stop_tol=1e-4,
    n_cores=None,
    chunksize=1,
):
    """
    Multiprocessing version of optimise2_deap.

    Optimisation2:
    TempSedRout calibration against observed SSC.

    Returns
    -------
    optimized_param_dict, best_score, pop, logbook, hof,
    generation_history, population_history
    """

    random.seed(seed)
    np.random.seed(seed)

    objective = str(objective).lower()
    allowed_objectives = {"rmse", "log_rmse", "mse", "kge", "nkge", "nsh", "nse"}
    if objective not in allowed_objectives:
        raise ValueError(
            "objective must be one of: 'rmse', 'log_rmse', 'mse', 'kge', 'nkge', 'nsh', 'nse'"
        )

    early_stop = bool(early_stop)

    if early_stop:
        if early_stop_rounds is None:
            raise ValueError(
                "When early_stop=True, early_stop_rounds must be provided."
            )
        early_stop_rounds = int(early_stop_rounds)
        if early_stop_rounds <= 0:
            raise ValueError("early_stop_rounds must be a positive integer.")
        early_stop_tol = float(early_stop_tol)
        if early_stop_tol < 0:
            raise ValueError("early_stop_tol must be >= 0.")
    else:
        early_stop_rounds = None
        early_stop_tol = None

    allowed_param_names = [
        "dispers1_TempSedRout",
        "dispers2_TempSedRout",
        "dispers3_TempSedRout",
        "Fd1_TempSedRout",
        "Fd2_TempSedRout",
        "Fd3_TempSedRout",
        "cr1_TempSedRout",
        "cr2_TempSedRout",
        "cr3_TempSedRout",
    ]
    
    if use_storage:
        allowed_param_names += [
            "fl_storage",
            "fh_storage",
            "fw_storage",
            "fa_storage",
        ]
    missing = [k for k in allowed_param_names if k not in param_dict]
    if missing:
        raise KeyError(f"These required parameters are missing from param_dict: {missing}")

    if optimize_only is None:
        param_names = [
            k for k in allowed_param_names
            if float(param_dict[k]["up"]) > float(param_dict[k]["low"])
        ]
        optimize_only_out = None
    else:
        optimize_only = list(optimize_only)

        invalid = [k for k in optimize_only if k not in allowed_param_names]
        if invalid:
            raise ValueError(
                f"These optimize_only parameters are not allowed: {invalid}"
            )

        param_names = [
            k for k in optimize_only
            if float(param_dict[k]["up"]) > float(param_dict[k]["low"])
        ]
        optimize_only_out = optimize_only

    bounds = [(float(param_dict[k]["low"]), float(param_dict[k]["up"])) for k in param_names]

    if len(param_names) == 0:
        raise ValueError(
            "No parameters selected for optimisation. Check optimize_only and bounds."
        )

    if "FitnessMin2MP" not in creator.__dict__:
        creator.create("FitnessMin2MP", base.Fitness, weights=(-1.0,))
    if "Individual2MP" not in creator.__dict__:
        creator.create("Individual2MP", list, fitness=creator.FitnessMin2MP)

    toolbox = base.Toolbox()

    def init_individual():
        vals = []
        for low, up in bounds:
            if low == up:
                vals.append(low)
            else:
                vals.append(random.uniform(low, up))
        return creator.Individual2MP(vals)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate",
        evaluate_param_set_tempsedrout,
        param_names=param_names,
        base_param_dict=param_dict,
        river_gdf=river_gdf,
        sediment_size=sediment_size,
        toml_file=toml_file,
        h=h,
        q=q,
        Q=Q,
        width=width,
        SSC_hru_frac=SSC_hru_frac,
        df_SSC_obs=df_SSC_obs,
        obs_time_col=obs_time_col,
        obs_value_col=obs_value_col,
        objective=objective,
        use_storage=use_storage,
        river_storage=river_storage,
        storage_data_type=storage_data_type,
    )

    toolbox.register("mate", tools.cxBlend, alpha=0.2)
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        eta=eta,
        low=[b[0] for b in bounds],
        up=[b[1] for b in bounds],
        indpb=0.2
    )
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("mean", np.mean)
    stats.register("std", np.std)

    generation_start_time = time.perf_counter()
    total_start_time = time.perf_counter()

    core_type = platform.processor()
    if not core_type:
        core_type = platform.machine()
    total_logical_cores = mp.cpu_count()

    if n_cores is None:
        n_cores_used = max(1, total_logical_cores - 1)
    else:
        n_cores_used = max(1, min(int(n_cores), total_logical_cores))

    generation_history = []
    population_history = []

    def record_population_history(generation, population):
        rows = []
        for ind_idx, ind in enumerate(population):
            row = {
                "generation": generation,
                "individual_id": ind_idx,
                "fitness": ind.fitness.values[0] if ind.fitness.valid else np.nan,
            }
            for name, value in zip(param_names, ind):
                row[name] = float(value)
            rows.append(row)
        return rows

    def record_generation_history(generation, population):
        best_ind_gen = tools.selBest(population, 1)[0]
        best_score_gen = best_ind_gen.fitness.values[0]

        row = {
            "generation": generation,
            "best_score": best_score_gen,
        }

        stat_record = stats.compile(population)
        for k, v in stat_record.items():
            row[f"fitness_{k}"] = float(v)

        for name, value in zip(param_names, best_ind_gen):
            row[name] = float(value)

        best_param_dict_gen = copy.deepcopy(param_dict)
        for name, value in zip(param_names, best_ind_gen):
            best_param_dict_gen[name]["value"] = float(value)

        for k in allowed_param_names:
            row[f"full_{k}"] = best_param_dict_gen[k]["value"]

        return row

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=n_cores_used)

    def parallel_map(func, iterable):
        return pool.map(func, iterable, chunksize)

    toolbox.register("map", parallel_map)

    try:
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        hof.update(pop)
        best_score_so_far = hof[0].fitness.values[0]
        stagnant_generations = 0

        generation_history.append(record_generation_history(0, pop))
        population_history.extend(record_population_history(0, pop))

        def print_generation_status(gen, population):
            best_ind_gen = tools.selBest(population, 1)[0]
            best_score_gen = best_ind_gen.fitness.values[0]

            best_param_dict_gen = copy.deepcopy(param_dict)
            for name, value in zip(param_names, best_ind_gen):
                best_param_dict_gen[name]["value"] = float(value)

            routed_params = {
                k: best_param_dict_gen[k]["value"]
                for k in allowed_param_names
            }

            generation_elapsed_min = (time.perf_counter() - generation_start_time) / 60.0
            total_elapsed_min = (time.perf_counter() - total_start_time) / 60.0

            print(f"\nGeneration {gen}/{n_generations}")
            print(f"Objective ({objective}) = {best_score_gen}")
            print(f"Elapsed this generation = {generation_elapsed_min:.2f} min")
            print(f"Elapsed total = {total_elapsed_min:.2f} min")
            print(f"Cores used = {n_cores_used} / {total_logical_cores}")
            print(f"Core/CPU type = {core_type}")
            print(f"TempSedRout params = {routed_params}")
            print(f"use_storage = {use_storage}")
            print(f"storage_data_type = {storage_data_type}")
            print(f"Optimized only = {param_names if optimize_only_out is not None else 'all eligible parameters'}")

            if early_stop:
                print(
                    f"Early stopping monitor = {stagnant_generations}/{early_stop_rounds} "
                    f"(tol={early_stop_tol})"
                )

        print_generation_status(0, pop)

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

        record = stats.compile(pop) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        print(logbook.stream)

        save_optimisation_checkpoint2(
            output_path=checkpoint_path,
            generation=0,
            param_dict=param_dict,
            param_names=param_names,
            population=pop,
            logbook=logbook,
            hof=hof,
            objective=objective,
            optimize_only=optimize_only_out,
            generation_history=generation_history,
            population_history=population_history
        )

        for gen in range(1, n_generations + 1):
            generation_start_time = time.perf_counter()

            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            for ind in offspring:
                for j, (low, up) in enumerate(bounds):
                    if ind[j] < low:
                        ind[j] = low
                    elif ind[j] > up:
                        ind[j] = up

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring
            hof.update(pop)

            if early_stop:
                current_best_score = hof[0].fitness.values[0]
                improvement = best_score_so_far - current_best_score

                if improvement > early_stop_tol:
                    best_score_so_far = current_best_score
                    stagnant_generations = 0
                else:
                    stagnant_generations += 1

            generation_history.append(record_generation_history(gen, pop))
            population_history.extend(record_population_history(gen, pop))

            record = stats.compile(pop) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(logbook.stream)

            print_generation_status(gen, pop)

            save_optimisation_checkpoint2(
                output_path=checkpoint_path,
                generation=gen,
                param_dict=param_dict,
                param_names=param_names,
                population=pop,
                logbook=logbook,
                hof=hof,
                objective=objective,
                optimize_only=optimize_only_out,
                generation_history=generation_history,
                population_history=population_history
            )

            if early_stop and stagnant_generations >= early_stop_rounds:
                print(
                    f"\nEarly stopping triggered at generation {gen}: "
                    f"best objective did not improve by more than {early_stop_tol} "
                    f"for {early_stop_rounds} consecutive generations."
                )
                break

    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        print(f"Last fully completed generation saved at: {checkpoint_path}")
        raise

    finally:
        pool.close()
        pool.join()

    best_ind = hof[0]
    best_score = hof[0].fitness.values[0]

    optimized_param_dict = copy.deepcopy(param_dict)
    for name, value in zip(param_names, best_ind):
        optimized_param_dict[name]["value"] = float(value)

    return (
        optimized_param_dict,
        best_score,
        pop,
        logbook,
        hof,
        generation_history,
        population_history
    )

#%% ==================
# Optimisation 3
# Combined ErosionModel + TempSedRout
#   ==================

'''
ErosionModel parameters → SSC_hru_frac → TempSedRout → outlet SSC objective

'''
def prepare_obs_sim_series3(
    param_dict,
    model_input,
    df_runoff,
    rain,
    df_swe,
    sand_hru_stat,
    silt_hru_stat,
    river_gdf,
    sediment_size,
    toml_file,
    h,
    q,
    Q,
    width,
    df_SSC_obs,
    obs_time_col="time",
    obs_value_col="SSC",
    cold_region=True,
    zero_landcover_class0=True,
    number_fractions=3,
    use_storage=False,
    river_storage=None,
    storage_data_type="length",
):
    """
    Prepare observed and simulated SSC series for combined SedHydro optimisation.
    Workflow: 
        -> Erosion model parameters 
        -> grid SSC 
        -> HRU SSC 
        -> gamma routing at HRU level 
        -> fraction SSC_hru_frac 
        -> TempSedRout 
        -> outlet SSC 
        -> compare with observations
    Parameters
    ----------
    param_dict : dict
        Model parameter dictionary.
    model_input : pandas.DataFrame
        Spatial model input table.
    df_runoff : pandas.DataFrame
        Runoff time series.
    rain : pandas.DataFrame
        Rainfall time series.
    df_swe : pandas.DataFrame
        Snow water equivalent time series.
    sand_hru_stat, silt_hru_stat : pandas.DataFrame
        HRU sediment fraction tables.
    river_gdf : geopandas.GeoDataFrame
        River network.
    sediment_size : pandas.DataFrame
        Sediment size class table.
    toml_file : str
        TempSedRout configuration file.
    h, q, Q, width : pandas.DataFrame
        Hydraulic input time series.
    df_SSC_obs : pandas.DataFrame
        Observed SSC time series.
    obs_time_col : str, optional
        Observed time column name.
    obs_value_col : str, optional
        Observed SSC column name.
    cold_region : bool, optional
        If True, apply snow attenuation.
    zero_landcover_class0 : bool, optional
        If True, force landcover class 0 to zero SSC.
    number_fractions : int, optional
        Number of sediment size fractions.
    use_storage : bool, optional
        If True, use TempSedRout_storage.
    river_storage : pandas.DataFrame, optional
        Storage/depression system input table.
    storage_data_type : str, optional
        Type of storage input data.

    Returns
    -------
    obs : numpy.ndarray
        Aligned observed SSC values.
    sim : numpy.ndarray
        Aligned simulated outlet SSC values.
    """

    param_dict = copy.deepcopy(param_dict)
    param_dict = apply_order_constraints_to_param_dict(param_dict)

    # -------------------------
    # Erosion model
    # -------------------------
    model_sed = build_model_sed_from_params(param_dict, model_input)

    df_runoff_a, rain_a, common_times = align_forcing_data(
        df_runoff,
        rain
    )

    model_sed, time_cols = add_time_columns_to_model(
        model_sed,
        common_times
    )

    model_sed = calculate_grid_ssc(
        model_sed,
        df_runoff_a,
        rain_a,
        time_cols,
        df_swe=df_swe,
        cold_region=cold_region,
        zero_landcover_class0=zero_landcover_class0
    )

    # -------------------------
    # Grid SSC -> HRU SSC
    # Same HRU product used by save_optimisation_results1b
    # -------------------------
    SSC_hru = compute_hru_ssc_from_grids_pergridrunoff(
        model_sed=model_sed,
        df_runoff=df_runoff_a,
        grid_hru_col="HRU_ID",
        runoff_hru_col="hruId",
        runoff_col="averageRoutedRunoff",
        return_wide=True
    )

    # -------------------------
    # Gamma routing at HRU level
    # -------------------------
    a_rout = float(param_dict["a_rout"]["value"])
    mt_rout = float(param_dict["mt_rout"]["value"])
    K_rout = int(round(param_dict["K_rout"]["value"]))

    SSC_hru_routed = route_ssc_hru_gamma(
        SSC_hru=SSC_hru,
        a=a_rout,
        mt=mt_rout,
        K=K_rout,
        hru_col="HRU_ID"
    )

    # -------------------------
    # HRU SSC fractions
    # -------------------------
    SSC_hru_frac = create_ssc_hru_fraction_dict(
        SSC_hru=SSC_hru_routed,
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

    # -------------------------
    # Match SSC_hru_frac times with TempSedRout hydraulic times
    # -------------------------
    h_times = pd.DatetimeIndex(h.index).sort_values()
    common_routing_times = h_times.copy()

    for frac, df in SSC_hru_frac.items():
        ssc_times = pd.to_datetime(
            df.columns,
            format="t_%Y%m%d_%H%M%S"
        ).sort_values()

        common_routing_times = common_routing_times.intersection(ssc_times)

    h = h.loc[common_routing_times].copy()
    q = q.loc[common_routing_times].copy()
    Q = Q.loc[common_routing_times].copy()
    width = width.loc[common_routing_times].copy()

    SSC_hru_frac_common = {}

    for frac, df in SSC_hru_frac.items():
        col_time_map = {
            pd.to_datetime(col, format="t_%Y%m%d_%H%M%S"): col
            for col in df.columns
        }

        keep_cols = [
            col_time_map[t]
            for t in common_routing_times
            if t in col_time_map
        ]

        SSC_hru_frac_common[frac] = df.loc[:, keep_cols].copy()

    SSC_hru_frac = SSC_hru_frac_common

    # -------------------------
    # Add missing HRUs/reaches required by TempSedRout
    # -------------------------
    from utils import fill_missing_hru

    SSC_hru_frac = fill_missing_hru(
        SSC_hru_frac,
        river_gdf,
        id_col="LINKNO"
    )

    # -------------------------
    # TempSedRout and obs/sim
    # -------------------------
    obs, sim = prepare_obs_sim_series_tempsedrout(
        param_dict=param_dict,
        river_gdf=river_gdf,
        sediment_size=sediment_size,
        toml_file=toml_file,
        h=h,
        q=q,
        Q=Q,
        width=width,
        SSC_hru_frac=SSC_hru_frac,
        df_SSC_obs=df_SSC_obs,
        obs_time_col=obs_time_col,
        obs_value_col=obs_value_col,
        use_storage=use_storage,
        river_storage=river_storage,
        storage_data_type=storage_data_type
    )

    return obs, sim


def evaluate_param_set3(
    individual,
    param_names,
    base_param_dict,
    model_input,
    df_runoff,
    rain,
    df_swe,
    sand_hru_stat,
    silt_hru_stat,
    river_gdf,
    sediment_size,
    toml_file,
    h,
    q,
    Q,
    width,
    df_SSC_obs,
    obs_time_col="time",
    obs_value_col="SSC",
    objective="log_rmse",
    cold_region=True,
    zero_landcover_class0=False,
    number_fractions=3,
    use_storage=False,
    river_storage=None,
    storage_data_type="length",
):
    """
    Evaluate one combined SedHydro parameter set.

    Parameters
    ----------
    individual : list
        Candidate parameter values.
    param_names : list
        Names corresponding to individual values.
    base_param_dict : dict
        Base model parameter dictionary.
    model_input : pandas.DataFrame
        Spatial model input table.
    df_runoff : pandas.DataFrame
        Runoff time series.
    rain : pandas.DataFrame
        Rainfall time series.
    df_swe : pandas.DataFrame
        Snow water equivalent time series.
    sand_hru_stat, silt_hru_stat : pandas.DataFrame
        HRU sediment fraction tables.
    river_gdf : geopandas.GeoDataFrame
        River network.
    sediment_size : pandas.DataFrame
        Sediment size class table.
    toml_file : str
        TempSedRout configuration file.
    h, q, Q, width : pandas.DataFrame
        Hydraulic input time series.
    df_SSC_obs : pandas.DataFrame
        Observed SSC time series.
    obs_time_col : str, optional
        Observed time column name.
    obs_value_col : str, optional
        Observed SSC column name.
    objective : str, optional
        Objective function name.
    cold_region : bool, optional
        If True, apply snow attenuation.
    zero_landcover_class0 : bool, optional
        If True, force landcover class 0 to zero SSC.
    number_fractions : int, optional
        Number of sediment size fractions.
    use_storage : bool, optional
        If True, use TempSedRout_storage.
    river_storage : pandas.DataFrame, optional
        Storage/depression system input table.
    storage_data_type : str, optional
        Type of storage input data.

    Returns
    -------
    tuple
        One-element objective value tuple for DEAP minimisation.
    """

    try:
        param_dict_eval = copy.deepcopy(base_param_dict)

        for name, value in zip(param_names, individual):
            param_dict_eval[name]["value"] = float(value)

        param_dict_eval = apply_order_constraints_to_param_dict(param_dict_eval)

        obs, sim = prepare_obs_sim_series3(
            param_dict=param_dict_eval,
            model_input=model_input,
            df_runoff=df_runoff,
            rain=rain,
            df_swe=df_swe,
            sand_hru_stat=sand_hru_stat,
            silt_hru_stat=silt_hru_stat,
            river_gdf=river_gdf,
            sediment_size=sediment_size,
            toml_file=toml_file,
            h=h,
            q=q,
            Q=Q,
            width=width,
            df_SSC_obs=df_SSC_obs,
            obs_time_col=obs_time_col,
            obs_value_col=obs_value_col,
            cold_region=cold_region,
            zero_landcover_class0=zero_landcover_class0,
            number_fractions=number_fractions,
            use_storage=use_storage,
            river_storage=river_storage,
            storage_data_type=storage_data_type
        )

        if obs is None or sim is None:
            return (1e12,)

        score = objective_from_series(obs, sim, objective=objective)

        if not np.isfinite(score):
            score = 1e12

        return (score,)

    except Exception as e:
        print(f"Optimise3 evaluation failed: {e}")
        return (1e12,)


def optimise3_deap(
    param_dict,
    model_input,
    df_runoff,
    rain,
    df_swe,
    sand_hru_stat,
    silt_hru_stat,
    river_gdf,
    sediment_size,
    toml_file,
    h,
    q,
    Q,
    width,
    df_SSC_obs,
    use_storage=False,
    river_storage=None,
    storage_data_type="length",
    obs_time_col="time",
    obs_value_col="SSC",
    objective="log_rmse",
    cold_region=True,
    zero_landcover_class0=True,
    optimize_hill_routing_params=True,
    optimise_only_erosion=None,
    optimise_only_routing=None,
    number_fractions=3,
    n_generations=30,
    population_size=40,
    cxpb=0.6,
    mutpb=0.3,
    eta=20.0,
    seed=42,
    checkpoint_path="optimise3_deap_checkpoint.pkl",
    early_stop_rounds=None,
    early_stop_tol=1e-4
):
    """
    Combined single-core calibration of ErosionModel and TempSedRout.
    Combined calibration of:
        1) Erosion model parameters from optimise1b_deap
        2) TempSedRout parameters from optimise2_deap
    
    This function optimises selected ErosionModel parameters and selected
    TempSedRout parameters together in one DEAP optimisation loop. For each
    DEAP individual, the full model chain is recalculated:
    
        Erosion parameters
        -> grid SSC
        -> HRU SSC from grids
        -> HRU gamma routing
        -> HRU sediment-size fractions
        -> TempSedRout / TempSedRout_storage
        -> outlet SSC
        -> objective value against observed SSC
    
    This differs from separate optimisation because the erosion-generated
    SSC_hru_frac is recalculated during every objective evaluation instead of
    using a fixed SSC_hru_frac from a previous ErosionModel run.
    
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
    
    Parameters
    ----------
    param_dict : dict
        Dictionary of all model parameters. Each parameter must have:
            "value", "low", "up", and optionally "priority".
    
        Only parameters with up > low can be optimised. Parameters with
        low == up are kept fixed even if they are included in
        optimise_only_erosion or optimise_only_routing.
    
    model_input : pandas.DataFrame or geopandas.GeoDataFrame
        Grid-based ErosionModel input table. Required columns include:
            HRU_ID,
            median_slope,
            dominant_class_landcover,
            dominant_class_geol.
    
    df_runoff : pandas.DataFrame
        SUMMA runoff dataframe in long format. Required columns:
            time,
            hruId,
            averageRoutedRunoff.
    
    rain : pandas.DataFrame
        Rainfall dataframe in long format. Required columns:
            time,
            hruId,
            pptrate.
    
    df_swe : pandas.DataFrame
        Snow water equivalent dataframe in long format. Required columns:
            time,
            hruId,
            scalarSWE.
    
        Used when cold_region=True.
    
    sand_hru_stat : pandas.DataFrame
        HRU-level sand statistics. Required columns:
            HRU_ID,
            mean_sand.
    
    silt_hru_stat : pandas.DataFrame
        HRU-level silt statistics. Required columns:
            HRU_ID,
            mean_silt.
    
    river_gdf : geopandas.GeoDataFrame
        River network used by TempSedRout. Must include LINKNO and DSLINKNO.
    
    sediment_size : pandas.DataFrame
        Sediment size class table used by TempSedRout.
    
    toml_file : str
        Path to the TempSedRout constants/configuration TOML file.
    
    h : pandas.DataFrame
        Channel depth matrix. Index must be time and columns must be segment IDs.
    
    q : pandas.DataFrame
        Unit flow matrix. Index must be time and columns must be segment IDs.
    
    Q : pandas.DataFrame
        Discharge matrix. Index must be time and columns must be segment IDs.
    
    width : pandas.DataFrame
        Channel width matrix. Index must be time and columns must be segment IDs.
    
    df_SSC_obs : pandas.DataFrame
        Observed SSC time series. Default required columns:
            time,
            SSC.
    
    use_storage : bool, default=False
        If True, use TempSedRout_storage.
        If False, use standard TempSedRout.
    
    river_storage : pandas.DataFrame or None, default=None
        Storage/depressional-system table. Required only when use_storage=True.
    
    storage_data_type : str, default="length"
        Storage data type used by TempSedRout_storage.
        Common options:
            "length", "width", "area".
    
    obs_time_col : str, default="time"
        Time column name in df_SSC_obs.
    
    obs_value_col : str, default="SSC"
        Observed SSC column name in df_SSC_obs.
    
    objective : str, default="log_rmse"
        Objective function used for calibration. Supported options:
            "rmse",
            "log_rmse",
            "mse",
            "kge",
            "nkge",
            "nsh",
            "nse".
    
        RMSE, log-RMSE, and MSE are minimised directly.
        KGE, nKGE, and NSE/NSH are converted to minimisation form internally.
    
    cold_region : bool, default=True
        If True, apply SWE-based snow attenuation in the erosion model.
    
    zero_landcover_class0 : bool, default=True
        If True, force SSC to zero for grid cells with landcover class 0.
    
    optimize_hill_routing_params : bool, default=True
        Controls whether erosion-side gamma-routing parameters are eligible
        for optimisation.
    
        If False, a_rout and mt_rout are removed from the erosion optimisation
        candidate list.
    
    optimise_only_erosion : list[str] or None, default=None
        Optional list of erosion-side parameters to optimise.
    
        If None, all eligible erosion-side parameters with up > low are used,
        excluding TempSedRout parameters.
    
        Example for main erosion parameters:
            [
                "abase", "bbase",
                "as", "bs",
                "crain", "ceros", "ksnow"
            ]
    
        Example for landcover/geology parameters:
            [
                "al0", "al1", "al2", "al3", "al4", "al5",
                "bl0", "bl1", "bl2", "bl3", "bl4", "bl5",
                "ag1", "ag2", "ag3", "ag4", "ag5", "ag6",
                "ag7", "ag8", "ag9", "ag10", "ag11", "ag12", "ag13",
                "bg1", "bg2", "bg3", "bg4", "bg5", "bg6",
                "bg7", "bg8", "bg9", "bg10", "bg11", "bg12", "bg13"
            ]
    
        Example for HRU gamma-routing:
            [
                "a_rout", "mt_rout", "K_rout"
            ]
    
    optimise_only_routing : list[str] or None, default=None
        Optional list of TempSedRout-side parameters to optimise.
    
        If None, all eligible TempSedRout parameters with up > low are used.
    
        Main TempSedRout routing parameters:
            [
                "dispers1_TempSedRout",
                "dispers2_TempSedRout",
                "dispers3_TempSedRout",
                "Fd1_TempSedRout",
                "Fd2_TempSedRout",
                "Fd3_TempSedRout",
                "cr1_TempSedRout",
                "cr2_TempSedRout",
                "cr3_TempSedRout"
            ]
    
        If these are included in tempsedrout_allowed_param_names, the following
        can also be optimised:
            [
                "median_diam_TempSedRout",
                "SF_TempSedRout",
                "interp_TempSedRout"
            ]
    
        If use_storage=True, storage parameters can be optimised:
            [
                "fl_storage",
                "fh_storage",
                "fw_storage",
                "fa_storage"
            ]
    
    number_fractions : int, default=3
        Number of sediment fractions used to split HRU SSC.
        Current workflow assumes:
            0 = clay,
            1 = silt,
            2 = sand.
    
    n_generations : int, default=30
        Number of DEAP generations.
    
    population_size : int, default=40
        Number of individuals in the DEAP population.
    
    cxpb : float, default=0.6
        Crossover probability.
    
    mutpb : float, default=0.3
        Mutation probability.
    
    eta : float, default=20.0
        Distribution index for polynomial bounded mutation.
    
    seed : int, default=42
        Random seed for reproducibility.
    
    checkpoint_path : str, default="optimise3_deap_checkpoint.pkl"
        Path where the optimisation checkpoint is written after each completed
        generation.
    
    early_stop_rounds : int or None, default=None
        If not None, stop when the best score has not improved by more than
        early_stop_tol for this many consecutive generations.
    
    early_stop_tol : float, default=1e-4
        Minimum improvement required to reset the early-stopping counter.
    
    Outputs
    -------
    optimised_param_dict : dict
        Copy of param_dict with optimised parameter values updated.
    
    best_score : float
        Best objective value found by DEAP.
        Lower is better for the stored objective value.
    
    pop : list
        Final DEAP population.
    
    logbook : deap.tools.Logbook
        Generation-by-generation DEAP statistics.
    
    hof : deap.tools.HallOfFame
        Best individual found during optimisation.
    
    Returns
    -------
    tuple
        (
            optimised_param_dict,
            best_score,
            pop,
            logbook,
            hof
        )
    
    Notes
    -----
    - This is the single-core version. For multiprocessing use
      optimise3_deap_mp.
    - Parameter ordering constraints are applied to al, ag, bl, and bg
      parameter families using their priority values in param_dict.
    - Checkpoints are written after generation 0 and after every completed
      generation.
    - If interrupted with KeyboardInterrupt, the last completed generation is
      preserved in checkpoint_path.
    - The actual optimised parameter set is always:
          param_names = erosion_param_names + tempsedrout_param_names
    
      where each list is filtered by:
          parameter exists in param_dict,
          up > low,
          optional optimise_only list,
          optional storage setting,
          optional optimize_hill_routing_params setting.
    """

    random.seed(seed)
    np.random.seed(seed)

    objective = str(objective).lower()
    allowed_objectives = {"rmse", "log_rmse", "mse", "kge", "nkge", "nsh", "nse"}

    if objective not in allowed_objectives:
        raise ValueError(
            "objective must be one of: 'rmse', 'log_rmse', 'mse', 'kge', 'nkge', 'nsh', 'nse'"
        )

    if early_stop_rounds is not None:
        early_stop_rounds = int(early_stop_rounds)
        if early_stop_rounds <= 0:
            raise ValueError("early_stop_rounds must be a positive integer or None.")
        early_stop_tol = float(early_stop_tol)
        if early_stop_tol < 0:
            raise ValueError("early_stop_tol must be >= 0.")

    # -------------------------
    # Erosion-model parameters
    # Same selection logic as optimise1b_deap
    # -------------------------
    erosion_param_names = [
        k for k, v in param_dict.items()
        if float(v["up"]) > float(v["low"])
    ]

    routing_gamma_params = ["a_rout", "mt_rout"]

    if not optimize_hill_routing_params:
        erosion_param_names = [
            k for k in erosion_param_names
            if k not in routing_gamma_params
        ]

    tempsedrout_allowed_param_names = [
        "dispers1_TempSedRout",
        "dispers2_TempSedRout",
        "dispers3_TempSedRout",
        "Fd1_TempSedRout",
        "Fd2_TempSedRout",
        "Fd3_TempSedRout",
        "cr1_TempSedRout",
        "cr2_TempSedRout",
        "cr3_TempSedRout",
    ]

    if use_storage:
        tempsedrout_allowed_param_names += [
            "fl_storage",
            "fh_storage",
            "fw_storage",
            "fa_storage",
        ]

    erosion_param_names = [
        k for k in erosion_param_names
        if k not in tempsedrout_allowed_param_names
    ]

    if optimise_only_erosion is not None:
        optimise_only_erosion = list(optimise_only_erosion)

        unknown_erosion = [
            k for k in optimise_only_erosion
            if k not in param_dict
        ]

        if unknown_erosion:
            raise ValueError(
                f"These parameters in optimise_only_erosion are not in param_dict: {unknown_erosion}"
            )

        erosion_param_names = [
            k for k in erosion_param_names
            if k in optimise_only_erosion
        ]

    # -------------------------
    # TempSedRout parameters
    # Same selection logic as optimise2_deap
    # -------------------------
    missing_routing = [
        k for k in tempsedrout_allowed_param_names
        if k not in param_dict
    ]

    if missing_routing:
        raise KeyError(
            f"These required TempSedRout parameters are missing from param_dict: {missing_routing}"
        )

    if optimise_only_routing is None:
        tempsedrout_param_names = [
            k for k in tempsedrout_allowed_param_names
            if float(param_dict[k]["up"]) > float(param_dict[k]["low"])
        ]
    else:
        optimise_only_routing = list(optimise_only_routing)

        invalid_routing = [
            k for k in optimise_only_routing
            if k not in tempsedrout_allowed_param_names
        ]

        if invalid_routing:
            raise ValueError(
                f"These optimise_only_routing parameters are not allowed: {invalid_routing}"
            )

        tempsedrout_param_names = [
            k for k in optimise_only_routing
            if float(param_dict[k]["up"]) > float(param_dict[k]["low"])
        ]

    param_names = erosion_param_names + tempsedrout_param_names

    if len(param_names) == 0:
        raise ValueError(
            "No parameters selected for optimise3_deap. Check optimise_only_erosion, "
            "optimise_only_routing, bounds, and optimize_hill_routing_params."
        )

    bounds = [
        (float(param_dict[k]["low"]), float(param_dict[k]["up"]))
        for k in param_names
    ]

    if "FitnessMin3" not in creator.__dict__:
        creator.create("FitnessMin3", base.Fitness, weights=(-1.0,))

    if "Individual3" not in creator.__dict__:
        creator.create("Individual3", list, fitness=creator.FitnessMin3)

    toolbox = base.Toolbox()

    def init_individual():
        vals = []
        for low, up in bounds:
            if low == up:
                vals.append(low)
            else:
                vals.append(random.uniform(low, up))

        ind = creator.Individual3(vals)

        ind = repair_individual_with_order_constraints(
            individual=ind,
            param_names=param_names,
            base_param_dict=param_dict
        )

        return ind

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate",
        evaluate_param_set3,
        param_names=param_names,
        base_param_dict=param_dict,
        model_input=model_input,
        df_runoff=df_runoff,
        rain=rain,
        df_swe=df_swe,
        sand_hru_stat=sand_hru_stat,
        silt_hru_stat=silt_hru_stat,
        river_gdf=river_gdf,
        sediment_size=sediment_size,
        toml_file=toml_file,
        h=h,
        q=q,
        Q=Q,
        width=width,
        df_SSC_obs=df_SSC_obs,
        obs_time_col=obs_time_col,
        obs_value_col=obs_value_col,
        objective=objective,
        cold_region=cold_region,
        zero_landcover_class0=zero_landcover_class0,
        number_fractions=number_fractions,
        use_storage=use_storage,
        river_storage=river_storage,
        storage_data_type=storage_data_type
    )

    toolbox.register("mate", tools.cxBlend, alpha=0.2)
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        eta=eta,
        low=[b[0] for b in bounds],
        up=[b[1] for b in bounds],
        indpb=0.2
    )
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("mean", np.mean)
    stats.register("std", np.std)

    generation_start_time = time.perf_counter()
    total_start_time = time.perf_counter()

    n_cores_used = 1
    core_type = platform.processor() or platform.machine()
    total_logical_cores = mp.cpu_count()

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_ind))

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)

    best_score_so_far = hof[0].fitness.values[0]
    stagnant_generations = 0

    def print_generation_status(gen, population):
        best_ind_gen = tools.selBest(population, 1)[0]
        best_score_gen = best_ind_gen.fitness.values[0]

        best_param_dict_gen = copy.deepcopy(param_dict)

        for name, value in zip(param_names, best_ind_gen):
            best_param_dict_gen[name]["value"] = float(value)

        best_param_dict_gen = apply_order_constraints_to_param_dict(
            best_param_dict_gen
        )

        erosion_params = {
            k: best_param_dict_gen[k]["value"]
            for k in erosion_param_names
        }

        tempsedrout_params = {
            k: best_param_dict_gen[k]["value"]
            for k in tempsedrout_param_names
        }

        generation_elapsed_min = (
            time.perf_counter() - generation_start_time
        ) / 60.0

        total_elapsed_min = (
            time.perf_counter() - total_start_time
        ) / 60.0

        print(f"\nGeneration {gen}/{n_generations}")
        print(f"Objective ({objective}) = {best_score_gen}")
        print(f"Elapsed this generation = {generation_elapsed_min:.2f} min")
        print(f"Elapsed total = {total_elapsed_min:.2f} min")
        print(f"Cores used = {n_cores_used} / {total_logical_cores}")
        print(f"Core/CPU type = {core_type}")
        print(f"Erosion params = {erosion_params}")
        print(f"TempSedRout params = {tempsedrout_params}")
        print(f"use_storage = {use_storage}")
        print(f"storage_data_type = {storage_data_type}")
        print(f"optimise_only_erosion = {optimise_only_erosion}")
        print(f"optimise_only_routing = {optimise_only_routing}")

        if early_stop_rounds is not None:
            print(
                f"Early stopping monitor = {stagnant_generations}/{early_stop_rounds} "
                f"(tol={early_stop_tol})"
            )

    print_generation_status(0, pop)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    print(logbook.stream)

    save_optimisation_checkpoint2(
        output_path=checkpoint_path,
        generation=0,
        param_dict=param_dict,
        param_names=param_names,
        population=pop,
        logbook=logbook,
        hof=hof,
        objective=objective,
        optimize_only={
            "optimise_only_erosion": optimise_only_erosion,
            "optimise_only_routing": optimise_only_routing
        }
    )

    try:
        for gen in range(1, n_generations + 1):
            generation_start_time = time.perf_counter()

            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            for ind in offspring:
                for j, (low, up) in enumerate(bounds):
                    if ind[j] < low:
                        ind[j] = low
                    elif ind[j] > up:
                        ind[j] = up

                repair_individual_with_order_constraints(
                    individual=ind,
                    param_names=param_names,
                    base_param_dict=param_dict
                )

            invalid_ind = [
                ind for ind in offspring
                if not ind.fitness.valid
            ]

            fitnesses = list(map(toolbox.evaluate, invalid_ind))

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring
            hof.update(pop)

            if early_stop_rounds is not None:
                current_best_score = hof[0].fitness.values[0]
                improvement = best_score_so_far - current_best_score

                if improvement > early_stop_tol:
                    best_score_so_far = current_best_score
                    stagnant_generations = 0
                else:
                    stagnant_generations += 1

            record = stats.compile(pop)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(logbook.stream)

            print_generation_status(gen, pop)

            save_optimisation_checkpoint2(
                output_path=checkpoint_path,
                generation=gen,
                param_dict=param_dict,
                param_names=param_names,
                population=pop,
                logbook=logbook,
                hof=hof,
                objective=objective,
                optimize_only={
                    "optimise_only_erosion": optimise_only_erosion,
                    "optimise_only_routing": optimise_only_routing
                }
            )

            if (
                early_stop_rounds is not None
                and stagnant_generations >= early_stop_rounds
            ):
                print(
                    f"\nEarly stopping triggered at generation {gen}: "
                    f"best objective did not improve by more than {early_stop_tol} "
                    f"for {early_stop_rounds} consecutive generations."
                )
                break

    except KeyboardInterrupt:
        print("\nOptimise3 interrupted by user.")
        print(f"Last fully completed generation saved at: {checkpoint_path}")
        raise

    best_ind = hof[0]
    best_score = hof[0].fitness.values[0]

    optimised_param_dict = copy.deepcopy(param_dict)

    for name, value in zip(param_names, best_ind):
        optimised_param_dict[name]["value"] = float(value)

    optimised_param_dict = apply_order_constraints_to_param_dict(
        optimised_param_dict
    )

    return optimised_param_dict, best_score, pop, logbook, hof

#%% ==================
# multiprocessing for optimisation3
#   ==================


def optimise3_deap_mp(
    param_dict,
    model_input,
    df_runoff,
    rain,
    df_swe,
    sand_hru_stat,
    silt_hru_stat,
    river_gdf,
    sediment_size,
    toml_file,
    h,
    q,
    Q,
    width,
    df_SSC_obs,
    use_storage=False,
    river_storage=None,
    storage_data_type="length",
    obs_time_col="time",
    obs_value_col="SSC",
    objective="log_rmse",
    cold_region=True,
    zero_landcover_class0=True,
    optimize_hill_routing_params=True,
    optimise_only_erosion=None,
    optimise_only_routing=None,
    number_fractions=3,
    n_generations=30,
    population_size=40,
    cxpb=0.6,
    mutpb=0.3,
    eta=20.0,
    seed=42,
    checkpoint_path="optimise3_deap_mp_checkpoint.pkl",
    early_stop=True,
    early_stop_rounds=None,
    early_stop_tol=1e-4,
    n_cores=None,
    chunksize=1,
):

    """
    Multiprocessing version of optimise3_deap.

    This function calibrates the ErosionModel and TempSedRout model together
    in one DEAP optimisation loop. Each DEAP individual contains selected
    erosion parameters and selected TempSedRout parameters. For every
    individual, the function runs the full modelling chain:

        Erosion parameters
        -> grid SSC
        -> HRU SSC from grids
        -> HRU gamma routing
        -> HRU SSC fractions
        -> TempSedRout / TempSedRout_storage
        -> outlet SSC
        -> objective value against observed SSC

    Parameters
    ----------
    param_dict : dict
        Dictionary of all model parameters. Each entry must have:
            "value", "low", "up", and optionally "priority".

        Erosion parameters and TempSedRout parameters are both read from this
        same dictionary.

    model_input : pandas.DataFrame or geopandas.GeoDataFrame
        Grid-based ErosionModel input table. Required columns include:
            HRU_ID,
            median_slope,
            dominant_class_landcover,
            dominant_class_geol.

    df_runoff : pandas.DataFrame
        SUMMA runoff dataframe in long format. Required columns:
            time,
            hruId,
            averageRoutedRunoff.

    rain : pandas.DataFrame
        Rainfall forcing dataframe in long format. Required columns:
            time,
            hruId,
            pptrate.

    df_swe : pandas.DataFrame
        Snow water equivalent dataframe in long format. Required columns:
            time,
            hruId,
            scalarSWE.

        Used only when cold_region=True.

    sand_hru_stat : pandas.DataFrame
        HRU-level sand statistics. Required columns:
            HRU_ID,
            mean_sand.

    silt_hru_stat : pandas.DataFrame
        HRU-level silt statistics. Required columns:
            HRU_ID,
            mean_silt.

    river_gdf : geopandas.GeoDataFrame
        River network used by TempSedRout. Must include LINKNO and DSLINKNO.

    sediment_size : pandas.DataFrame
        Sediment size class table used by TempSedRout.

    toml_file : str
        Path to the TempSedRout constants/configuration TOML file.

    h : pandas.DataFrame
        Channel depth matrix. Index must be time and columns must be segment IDs.

    q : pandas.DataFrame
        Unit flow matrix. Index must be time and columns must be segment IDs.

    Q : pandas.DataFrame
        Discharge matrix. Index must be time and columns must be segment IDs.

    width : pandas.DataFrame
        Channel width matrix. Index must be time and columns must be segment IDs.

    df_SSC_obs : pandas.DataFrame
        Observed SSC time series. Default columns:
            time,
            SSC.

    use_storage : bool, default=False
        If True, use TempSedRout_storage.
        If False, use TempSedRout.

    river_storage : pandas.DataFrame or None, default=None
        Storage/depressional-system input table. Required only when
        use_storage=True.

    storage_data_type : str, default="length"
        Type of storage data used by TempSedRout_storage.
        Common options:
            "length", "width", "area".

    obs_time_col : str, default="time"
        Time column name in df_SSC_obs.

    obs_value_col : str, default="SSC"
        Observed SSC column name in df_SSC_obs.

    objective : str, default="log_rmse"
        Objective function. Supported options:
            "rmse",
            "log_rmse",
            "mse",
            "kge",
            "nkge",
            "nsh",
            "nse".

        RMSE, log-RMSE, and MSE are minimised directly.
        KGE, nKGE, NSE/NSH are converted to minimisation form internally.

    cold_region : bool, default=True
        If True, apply SWE-based snow attenuation in the erosion model.

    zero_landcover_class0 : bool, default=True
        If True, force SSC to zero for grid cells with landcover class 0.

    optimize_hill_routing_params : bool, default=True
        Controls whether erosion-side gamma-routing parameters are eligible
        for optimisation.

        If False, a_rout and mt_rout are excluded from erosion optimisation.
        K_rout is only optimised if it is included in optimise_only_erosion
        and has variable bounds.

    optimise_only_erosion : list[str] or None, default=None
        Optional list of erosion-side parameters to optimise.

        If None, all eligible erosion-side parameters with up > low are used,
        excluding TempSedRout parameters.

        Example:
            [
                "abase", "bbase", "as", "bs",
                "ceros", "ksnow",
                "a_rout", "mt_rout"
            ]

    optimise_only_routing : list[str] or None, default=None
        Optional list of TempSedRout-side parameters to optimise.

        If None, all eligible TempSedRout parameters with up > low are used.

        Current eligible routing parameters are:
            dispers1_TempSedRout,
            dispers2_TempSedRout,
            dispers3_TempSedRout,
            Fd1_TempSedRout,
            Fd2_TempSedRout,
            Fd3_TempSedRout,
            cr1_TempSedRout,
            cr2_TempSedRout,
            cr3_TempSedRout.

        If use_storage=True, these are also eligible:
            fl_storage,
            fh_storage,
            fw_storage,
            fa_storage.

        The following TempSedRout parameters are used from param_dict but are
        intentionally not optimised in this function:
            median_diam_TempSedRout,
            SF_TempSedRout,
            interp_TempSedRout.

    number_fractions : int, default=3
        Number of sediment fractions used to split HRU SSC.
        Current workflow assumes 3 fractions:
            clay, silt, sand.

    n_generations : int, default=30
        Number of DEAP generations.

    population_size : int, default=40
        Number of DEAP individuals in each generation.

    cxpb : float, default=0.6
        Crossover probability.

    mutpb : float, default=0.3
        Mutation probability.

    eta : float, default=20.0
        Distribution index for polynomial bounded mutation.

    seed : int, default=42
        Random seed for reproducibility.

    checkpoint_path : str, default="optimise3_deap_mp_checkpoint.pkl"
        Path used to save optimisation checkpoints after each completed
        generation.

    early_stop : bool, default=True
        If True, stop when the best score has not improved sufficiently for
        early_stop_rounds consecutive generations.

    early_stop_rounds : int or None, default=None
        Number of consecutive stagnant generations allowed when early_stop=True.

    early_stop_tol : float, default=1e-4
        Minimum improvement required to reset the early-stopping counter.

    n_cores : int or None, default=None
        Number of CPU cores to use.

        If None, the function uses:
            multiprocessing.cpu_count() - 1

    chunksize : int, default=1
        Chunk size passed to multiprocessing pool.map.

    Outputs
    -------
    optimised_param_dict : dict
        Copy of param_dict with optimised parameter values updated.

    best_score : float
        Best objective value found by DEAP.
        Lower is better for the stored objective value.

    pop : list
        Final DEAP population.

    logbook : deap.tools.Logbook
        Generation-by-generation DEAP statistics.

    hof : deap.tools.HallOfFame
        Best individual found during optimisation.

    generation_history : list[dict]
        Detailed generation-level history.
        Each row includes:
            generation,
            best_score,
            fitness statistics,
            optimised parameter values,
            full parameter dictionary values.

    population_history : list[dict]
        Detailed individual-level history for every generation.
        Each row includes:
            generation,
            individual_id,
            fitness,
            optimised parameter values.

    Returns
    -------
    tuple
        (
            optimised_param_dict,
            best_score,
            pop,
            logbook,
            hof,
            generation_history,
            population_history
        )

    Notes
    -----
    - Multiprocessing uses the "spawn" context for compatibility with macOS
      and Windows.
    - Keep calls to this function inside:

          if __name__ == "__main__":
              mp.freeze_support()
              main()

      when running from a standalone script.
    - Parameter ordering constraints are applied to al, ag, bl, and bg
      parameter families using their priority values in param_dict.
    - Checkpoints are written after generation 0 and after every completed
      generation.
    - If interrupted with KeyboardInterrupt, the last completed generation is
      preserved in checkpoint_path.
    """
    random.seed(seed)
    np.random.seed(seed)

    objective = str(objective).lower()
    allowed_objectives = {"rmse", "log_rmse", "mse", "kge", "nkge", "nsh", "nse"}
    if objective not in allowed_objectives:
        raise ValueError(
            "objective must be one of: 'rmse', 'log_rmse', 'mse', 'kge', 'nkge', 'nsh', 'nse'"
        )

    early_stop = bool(early_stop)

    if early_stop:
        if early_stop_rounds is None:
            raise ValueError(
                "When early_stop=True, early_stop_rounds must be provided."
            )
        early_stop_rounds = int(early_stop_rounds)
        if early_stop_rounds <= 0:
            raise ValueError("early_stop_rounds must be a positive integer.")
        early_stop_tol = float(early_stop_tol)
        if early_stop_tol < 0:
            raise ValueError("early_stop_tol must be >= 0.")
    else:
        early_stop_rounds = None
        early_stop_tol = None

    # -------------------------
    # Erosion-model parameters
    # Same selection logic as optimise3_deap / optimise1b_deap
    # -------------------------
    erosion_param_names = [
        k for k, v in param_dict.items()
        if float(v["up"]) > float(v["low"])
    ]

    routing_gamma_params = ["a_rout", "mt_rout"]

    if not optimize_hill_routing_params:
        erosion_param_names = [
            k for k in erosion_param_names
            if k not in routing_gamma_params
        ]

    tempsedrout_allowed_param_names = [
        "dispers1_TempSedRout",
        "dispers2_TempSedRout",
        "dispers3_TempSedRout",
        "Fd1_TempSedRout",
        "Fd2_TempSedRout",
        "Fd3_TempSedRout",
        "cr1_TempSedRout",
        "cr2_TempSedRout",
        "cr3_TempSedRout",
    ]

    if use_storage:
        tempsedrout_allowed_param_names += [
            "fl_storage",
            "fh_storage",
            "fw_storage",
            "fa_storage",
        ]

    erosion_param_names = [
        k for k in erosion_param_names
        if k not in tempsedrout_allowed_param_names
    ]

    if optimise_only_erosion is not None:
        optimise_only_erosion = list(optimise_only_erosion)

        unknown_erosion = [
            k for k in optimise_only_erosion
            if k not in param_dict
        ]

        if unknown_erosion:
            raise ValueError(
                f"These parameters in optimise_only_erosion are not in param_dict: {unknown_erosion}"
            )

        erosion_param_names = [
            k for k in erosion_param_names
            if k in optimise_only_erosion
        ]

    # -------------------------
    # TempSedRout parameters
    # Same selection logic as optimise3_deap / optimise2_deap
    # -------------------------
    missing_routing = [
        k for k in tempsedrout_allowed_param_names
        if k not in param_dict
    ]

    if missing_routing:
        raise KeyError(
            f"These required TempSedRout parameters are missing from param_dict: {missing_routing}"
        )

    if optimise_only_routing is None:
        tempsedrout_param_names = [
            k for k in tempsedrout_allowed_param_names
            if float(param_dict[k]["up"]) > float(param_dict[k]["low"])
        ]
        optimise_only_routing_out = None
    else:
        optimise_only_routing = list(optimise_only_routing)

        invalid_routing = [
            k for k in optimise_only_routing
            if k not in tempsedrout_allowed_param_names
        ]

        if invalid_routing:
            raise ValueError(
                f"These optimise_only_routing parameters are not allowed: {invalid_routing}"
            )

        tempsedrout_param_names = [
            k for k in optimise_only_routing
            if float(param_dict[k]["up"]) > float(param_dict[k]["low"])
        ]
        optimise_only_routing_out = optimise_only_routing

    param_names = erosion_param_names + tempsedrout_param_names

    if len(param_names) == 0:
        raise ValueError(
            "No parameters selected for optimise3_deap_mp. Check optimise_only_erosion, "
            "optimise_only_routing, bounds, and optimize_hill_routing_params."
        )

    bounds = [
        (float(param_dict[k]["low"]), float(param_dict[k]["up"]))
        for k in param_names
    ]

    if "FitnessMin3MP" not in creator.__dict__:
        creator.create("FitnessMin3MP", base.Fitness, weights=(-1.0,))

    if "Individual3MP" not in creator.__dict__:
        creator.create("Individual3MP", list, fitness=creator.FitnessMin3MP)

    toolbox = base.Toolbox()

    def init_individual():
        vals = []
        for low, up in bounds:
            if low == up:
                vals.append(low)
            else:
                vals.append(random.uniform(low, up))

        ind = creator.Individual3MP(vals)

        ind = repair_individual_with_order_constraints(
            individual=ind,
            param_names=param_names,
            base_param_dict=param_dict
        )

        return ind

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate",
        evaluate_param_set3,
        param_names=param_names,
        base_param_dict=param_dict,
        model_input=model_input,
        df_runoff=df_runoff,
        rain=rain,
        df_swe=df_swe,
        sand_hru_stat=sand_hru_stat,
        silt_hru_stat=silt_hru_stat,
        river_gdf=river_gdf,
        sediment_size=sediment_size,
        toml_file=toml_file,
        h=h,
        q=q,
        Q=Q,
        width=width,
        df_SSC_obs=df_SSC_obs,
        obs_time_col=obs_time_col,
        obs_value_col=obs_value_col,
        objective=objective,
        cold_region=cold_region,
        zero_landcover_class0=zero_landcover_class0,
        number_fractions=number_fractions,
        use_storage=use_storage,
        river_storage=river_storage,
        storage_data_type=storage_data_type
    )

    toolbox.register("mate", tools.cxBlend, alpha=0.2)
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        eta=eta,
        low=[b[0] for b in bounds],
        up=[b[1] for b in bounds],
        indpb=0.2
    )
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("mean", np.mean)
    stats.register("std", np.std)

    generation_start_time = time.perf_counter()
    total_start_time = time.perf_counter()

    core_type = platform.processor()
    if not core_type:
        core_type = platform.machine()

    total_logical_cores = mp.cpu_count()

    if n_cores is None:
        n_cores_used = max(1, total_logical_cores - 1)
    else:
        n_cores_used = max(1, min(int(n_cores), total_logical_cores))

    generation_history = []
    population_history = []

    def record_population_history(generation, population):
        rows = []

        for ind_idx, ind in enumerate(population):
            row = {
                "generation": generation,
                "individual_id": ind_idx,
                "fitness": ind.fitness.values[0] if ind.fitness.valid else np.nan,
            }

            for name, value in zip(param_names, ind):
                row[name] = float(value)

            rows.append(row)

        return rows

    def record_generation_history(generation, population):
        best_ind_gen = tools.selBest(population, 1)[0]
        best_score_gen = best_ind_gen.fitness.values[0]

        row = {
            "generation": generation,
            "best_score": best_score_gen,
        }

        stat_record = stats.compile(population)
        for k, v in stat_record.items():
            row[f"fitness_{k}"] = float(v)

        for name, value in zip(param_names, best_ind_gen):
            row[name] = float(value)

        best_param_dict_gen = copy.deepcopy(param_dict)

        for name, value in zip(param_names, best_ind_gen):
            best_param_dict_gen[name]["value"] = float(value)

        best_param_dict_gen = apply_order_constraints_to_param_dict(
            best_param_dict_gen
        )

        for k in param_dict.keys():
            row[f"full_{k}"] = best_param_dict_gen[k]["value"]

        return row

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=n_cores_used)

    def parallel_map(func, iterable):
        return pool.map(func, iterable, chunksize)

    toolbox.register("map", parallel_map)

    try:
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        hof.update(pop)

        best_score_so_far = hof[0].fitness.values[0]
        stagnant_generations = 0

        generation_history.append(record_generation_history(0, pop))
        population_history.extend(record_population_history(0, pop))

        def print_generation_status(gen, population):
            best_ind_gen = tools.selBest(population, 1)[0]
            best_score_gen = best_ind_gen.fitness.values[0]

            best_param_dict_gen = copy.deepcopy(param_dict)

            for name, value in zip(param_names, best_ind_gen):
                best_param_dict_gen[name]["value"] = float(value)

            best_param_dict_gen = apply_order_constraints_to_param_dict(
                best_param_dict_gen
            )

            erosion_params = {
                k: best_param_dict_gen[k]["value"]
                for k in erosion_param_names
            }

            tempsedrout_params = {
                k: best_param_dict_gen[k]["value"]
                for k in tempsedrout_param_names
            }

            generation_elapsed_min = (
                time.perf_counter() - generation_start_time
            ) / 60.0

            total_elapsed_min = (
                time.perf_counter() - total_start_time
            ) / 60.0

            print(f"\nGeneration {gen}/{n_generations}")
            print(f"Objective ({objective}) = {best_score_gen}")
            print(f"Elapsed this generation = {generation_elapsed_min:.2f} min")
            print(f"Elapsed total = {total_elapsed_min:.2f} min")
            print(f"Cores used = {n_cores_used} / {total_logical_cores}")
            print(f"Core/CPU type = {core_type}")
            print(f"Erosion params = {erosion_params}")
            print(f"TempSedRout params = {tempsedrout_params}")
            print(f"use_storage = {use_storage}")
            print(f"storage_data_type = {storage_data_type}")
            print(f"optimise_only_erosion = {optimise_only_erosion}")
            print(f"optimise_only_routing = {optimise_only_routing}")

            if early_stop:
                print(
                    f"Early stopping monitor = {stagnant_generations}/{early_stop_rounds} "
                    f"(tol={early_stop_tol})"
                )

        print_generation_status(0, pop)

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

        record = stats.compile(pop) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        print(logbook.stream)

        save_optimisation_checkpoint2(
            output_path=checkpoint_path,
            generation=0,
            param_dict=param_dict,
            param_names=param_names,
            population=pop,
            logbook=logbook,
            hof=hof,
            objective=objective,
            optimize_only={
                "optimise_only_erosion": optimise_only_erosion,
                "optimise_only_routing": optimise_only_routing_out
            },
            generation_history=generation_history,
            population_history=population_history
        )

        for gen in range(1, n_generations + 1):
            generation_start_time = time.perf_counter()

            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            for ind in offspring:
                for j, (low, up) in enumerate(bounds):
                    if ind[j] < low:
                        ind[j] = low
                    elif ind[j] > up:
                        ind[j] = up

                repair_individual_with_order_constraints(
                    individual=ind,
                    param_names=param_names,
                    base_param_dict=param_dict
                )

            invalid_ind = [
                ind for ind in offspring
                if not ind.fitness.valid
            ]

            fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring
            hof.update(pop)

            if early_stop:
                current_best_score = hof[0].fitness.values[0]
                improvement = best_score_so_far - current_best_score

                if improvement > early_stop_tol:
                    best_score_so_far = current_best_score
                    stagnant_generations = 0
                else:
                    stagnant_generations += 1

            generation_history.append(record_generation_history(gen, pop))
            population_history.extend(record_population_history(gen, pop))

            record = stats.compile(pop) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(logbook.stream)

            print_generation_status(gen, pop)

            save_optimisation_checkpoint2(
                output_path=checkpoint_path,
                generation=gen,
                param_dict=param_dict,
                param_names=param_names,
                population=pop,
                logbook=logbook,
                hof=hof,
                objective=objective,
                optimize_only={
                    "optimise_only_erosion": optimise_only_erosion,
                    "optimise_only_routing": optimise_only_routing_out
                },
                generation_history=generation_history,
                population_history=population_history
            )

            if early_stop and stagnant_generations >= early_stop_rounds:
                print(
                    f"\nEarly stopping triggered at generation {gen}: "
                    f"best objective did not improve by more than {early_stop_tol} "
                    f"for {early_stop_rounds} consecutive generations."
                )
                break

    except KeyboardInterrupt:
        print("\nOptimise3 multiprocessing interrupted by user.")
        print(f"Last fully completed generation saved at: {checkpoint_path}")
        raise

    finally:
        pool.close()
        pool.join()

    best_ind = hof[0]
    best_score = hof[0].fitness.values[0]

    optimised_param_dict = copy.deepcopy(param_dict)

    for name, value in zip(param_names, best_ind):
        optimised_param_dict[name]["value"] = float(value)

    optimised_param_dict = apply_order_constraints_to_param_dict(
        optimised_param_dict
    )

    return (
        optimised_param_dict,
        best_score,
        pop,
        logbook,
        hof,
        generation_history,
        population_history
    )


#%%
def save_optimisation_results3_basic(
    optimised_param_dict,
    best_score,
    pop,
    logbook,
    hof,
    output_dir,
    file_name,
    generation_history=None,
    population_history=None
):
    """
    Save Optimisation3 basic results.

    Saved files
    -----------
    Main summary:
        {file_name}_main.csv
        {file_name}_main.pkl

    Separate tables:
        {file_name}_parameters.csv
        {file_name}_parameters.pkl
        {file_name}_logbook.csv
        {file_name}_logbook.pkl
        {file_name}_population_final.csv
        {file_name}_population_final.pkl
        {file_name}_generation_history.csv
        {file_name}_generation_history.pkl
        {file_name}_population_history.csv
        {file_name}_population_history.pkl
    """

    import os
    import pickle
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # file paths
    # -------------------------
    main_csv_file = os.path.join(output_dir, f"{file_name}_main.csv")
    main_pkl_file = os.path.join(output_dir, f"{file_name}_main.pkl")

    parameters_csv_file = os.path.join(output_dir, f"{file_name}_parameters.csv")
    parameters_pkl_file = os.path.join(output_dir, f"{file_name}_parameters.pkl")

    logbook_csv_file = os.path.join(output_dir, f"{file_name}_logbook.csv")
    logbook_pkl_file = os.path.join(output_dir, f"{file_name}_logbook.pkl")

    population_final_csv_file = os.path.join(output_dir, f"{file_name}_population_final.csv")
    population_final_pkl_file = os.path.join(output_dir, f"{file_name}_population_final.pkl")

    generation_history_csv_file = os.path.join(output_dir, f"{file_name}_generation_history.csv")
    generation_history_pkl_file = os.path.join(output_dir, f"{file_name}_generation_history.pkl")

    population_history_csv_file = os.path.join(output_dir, f"{file_name}_population_history.csv")
    population_history_pkl_file = os.path.join(output_dir, f"{file_name}_population_history.pkl")

    # -------------------------
    # 1) all parameters
    # -------------------------
    param_rows = []

    for k, v in optimised_param_dict.items():
        param_rows.append({
            "section": "parameters",
            "name": k,
            "value": v.get("value"),
            "low": v.get("low"),
            "up": v.get("up"),
            "priority": v.get("priority")
        })

    df_params = pd.DataFrame(param_rows)

    df_params.to_csv(parameters_csv_file, index=False)
    with open(parameters_pkl_file, "wb") as f:
        pickle.dump(df_params, f)

    # -------------------------
    # 2) best score
    # -------------------------
    df_score = pd.DataFrame([{
        "section": "best_score",
        "name": "objective_value",
        "value": best_score
    }])

    # -------------------------
    # 3) best individual
    # -------------------------
    best_ind = hof[0]

    df_best_ind = pd.DataFrame([
        {
            "section": "best_individual",
            "name": f"param_{i}",
            "value": val
        }
        for i, val in enumerate(best_ind)
    ])

    # -------------------------
    # 4) final population
    # -------------------------
    pop_rows = []

    for i, ind in enumerate(pop):
        pop_rows.append({
            "section": "population_final",
            "name": f"ind_{i}",
            "fitness": ind.fitness.values[0] if len(ind.fitness.values) > 0 else None,
            "values": list(ind)
        })

    df_population_final = pd.DataFrame(pop_rows)

    df_population_final.to_csv(population_final_csv_file, index=False)
    with open(population_final_pkl_file, "wb") as f:
        pickle.dump(df_population_final, f)

    # -------------------------
    # 5) logbook
    # -------------------------
    log_rows = []

    for record in logbook:
        log_rows.append({
            "section": "logbook",
            "generation": record.get("gen"),
            "nevals": record.get("nevals"),
            "min": record.get("min"),
            "mean": record.get("mean"),
            "std": record.get("std")
        })

    df_logbook = pd.DataFrame(log_rows)

    df_logbook.to_csv(logbook_csv_file, index=False)
    with open(logbook_pkl_file, "wb") as f:
        pickle.dump(df_logbook, f)

    # -------------------------
    # 6) generation history
    # -------------------------
    generation_history_df = (
        pd.DataFrame(generation_history)
        if generation_history is not None
        else pd.DataFrame()
    )

    if not generation_history_df.empty:
        generation_history_df.to_csv(generation_history_csv_file, index=False)
        with open(generation_history_pkl_file, "wb") as f:
            pickle.dump(generation_history_df, f)

    # -------------------------
    # 7) population history
    # -------------------------
    population_history_df = (
        pd.DataFrame(population_history)
        if population_history is not None
        else pd.DataFrame()
    )

    if not population_history_df.empty:
        population_history_df.to_csv(population_history_csv_file, index=False)
        with open(population_history_pkl_file, "wb") as f:
            pickle.dump(population_history_df, f)

    # -------------------------
    # 8) combined main summary
    # -------------------------
    df_all = pd.concat(
        [
            df_params,
            df_score,
            df_best_ind,
            df_population_final,
            df_logbook
        ],
        ignore_index=True,
        sort=False
    )

    df_all.to_csv(main_csv_file, index=False)

    with open(main_pkl_file, "wb") as f:
        pickle.dump(
            {
                "optimised_param_dict": optimised_param_dict,
                "best_score": best_score,
                "pop": pop,
                "logbook": logbook,
                "hof": hof,
                "parameters_df": df_params,
                "population_final_df": df_population_final,
                "logbook_df": df_logbook,
                "generation_history": generation_history_df,
                "population_history": population_history_df,
            },
            f
        )

    # -------------------------
    # 9) print saved outputs
    # -------------------------
    print(f"Saved main CSV: {main_csv_file}")
    print(f"Saved main PKL: {main_pkl_file}")
    print(f"Saved parameters CSV: {parameters_csv_file}")
    print(f"Saved parameters PKL: {parameters_pkl_file}")
    print(f"Saved logbook CSV: {logbook_csv_file}")
    print(f"Saved logbook PKL: {logbook_pkl_file}")
    print(f"Saved final population CSV: {population_final_csv_file}")
    print(f"Saved final population PKL: {population_final_pkl_file}")

    if not generation_history_df.empty:
        print(f"Saved generation history CSV: {generation_history_csv_file}")
        print(f"Saved generation history PKL: {generation_history_pkl_file}")

    if not population_history_df.empty:
        print(f"Saved population history CSV: {population_history_csv_file}")
        print(f"Saved population history PKL: {population_history_pkl_file}")



def save_optimisation_results3_full(
    optimised_param_dict,
    best_score,
    pop,
    logbook,
    hof,

    # Erosion model inputs
    model_input,
    df_runoff,
    rain,
    cat_hru,
    df_SSC_obs,
    sand_hru_stat,
    silt_hru_stat,

    # TempSedRout inputs
    river_gdf,
    sediment_size,
    toml_file,
    h,
    q,
    Q,
    width,

    output_dir,
    file_name="optimised3_parameters",
    obs_time_col="time",
    obs_value_col="SSC",
    zero_landcover_class0=False,
    number_fractions=3,
    df_swe=None,
    cold_region=True,
    use_storage=False,
    river_storage=None,
    storage_data_type="length",
    generation_history=None,
    population_history=None
):
    """
    Save full Optimisation3 results.

    Saves:
    - main optimisation summary
    - generation history
    - population history
    - final model_sed
    - raster-style NetCDF for grid SSC
    - SSC_hru
    - SSC_hru_frac
    - SSC_river_frac_out
    - SSC_river_tot_out
    """

    import os
    import pickle
    import numpy as np
    import pandas as pd
    from netCDF4 import Dataset
    
    # from optimisation_updated import (
    # build_model_sed_from_params,
    # align_forcing_data,
    # add_time_columns_to_model,
    # calculate_grid_ssc,
    # compute_hru_ssc_from_grids_pergridrunoff,
    # route_ssc_hru_gamma,
    # create_ssc_hru_fraction_dict,
    # run_final_tempsedrout,
    # prepare_obs_sim_series_tempsedrout)
    
    os.makedirs(output_dir, exist_ok=True)

    # =====================================================
    # FILE PATHS
    # =====================================================
    main_csv_file = os.path.join(output_dir, f"{file_name}_main.csv")
    main_pkl_file = os.path.join(output_dir, f"{file_name}_main.pkl")

    model_sed_pkl_file = os.path.join(output_dir, f"{file_name}_model_sed_final.pkl")
    model_sed_csv_file = os.path.join(output_dir, f"{file_name}_model_sed_final.csv")
    model_sed_nc_file = os.path.join(output_dir, f"{file_name}_model_sed_final_raster.nc")

    ssc_hru_pkl_file = os.path.join(output_dir, f"{file_name}_SSC_hru.pkl")
    ssc_hru_csv_file = os.path.join(output_dir, f"{file_name}_SSC_hru.csv")

    ssc_hru_frac_pkl_file = os.path.join(output_dir, f"{file_name}_SSC_hru_frac.pkl")

    ssc_river_tot_csv_file = os.path.join(output_dir, f"{file_name}_SSC_river_tot_out.csv")
    ssc_river_tot_pkl_file = os.path.join(output_dir, f"{file_name}_SSC_river_tot_out.pkl")

    ssc_river_frac_pkl_file = os.path.join(output_dir, f"{file_name}_SSC_river_frac_out.pkl")

    generation_history_csv_file = os.path.join(output_dir, f"{file_name}_generation_history.csv")
    generation_history_pkl_file = os.path.join(output_dir, f"{file_name}_generation_history.pkl")

    population_history_csv_file = os.path.join(output_dir, f"{file_name}_population_history.csv")
    population_history_pkl_file = os.path.join(output_dir, f"{file_name}_population_history.pkl")

    obs_sim_csv_file = os.path.join(output_dir, f"{file_name}_obs_sim.csv")
    obs_sim_pkl_file = os.path.join(output_dir, f"{file_name}_obs_sim.pkl")

    # =====================================================
    # SMALL HELPER: SAVE GRID SSC AS RASTER-STYLE NETCDF
    # =====================================================
    def save_model_sed_to_raster_netcdf(model_sed_df, time_cols_in, output_nc):

        required_cols = ["row", "col"]
        missing_cols = [c for c in required_cols if c not in model_sed_df.columns]

        if missing_cols:
            raise ValueError(
                f"model_sed_df must contain columns {required_cols}. Missing: {missing_cols}"
            )

        time_values = pd.to_datetime(
            [c.replace("t_", "") for c in time_cols_in],
            format="%Y%m%d_%H%M%S"
        )

        n_time = len(time_values)

        row_vals = model_sed_df["row"].to_numpy(dtype=int)
        col_vals = model_sed_df["col"].to_numpy(dtype=int)

        row_min = int(np.min(row_vals))
        row_max = int(np.max(row_vals))
        col_min = int(np.min(col_vals))
        col_max = int(np.max(col_vals))

        n_y = row_max - row_min + 1
        n_x = col_max - col_min + 1

        row_idx = row_vals - row_min
        col_idx = col_vals - col_min

        ssc_cube = np.full((n_time, n_y, n_x), np.nan, dtype=np.float32)
        ssc_values = model_sed_df[time_cols_in].to_numpy(dtype=np.float32)

        for i in range(len(model_sed_df)):
            ssc_cube[:, row_idx[i], col_idx[i]] = ssc_values[i, :]

        with Dataset(output_nc, "w", format="NETCDF4") as ds:
            ds.createDimension("time", n_time)
            ds.createDimension("y", n_y)
            ds.createDimension("x", n_x)

            time_var = ds.createVariable("time", str, ("time",))
            y_var = ds.createVariable("y", "i4", ("y",))
            x_var = ds.createVariable("x", "i4", ("x",))

            ssc_var = ds.createVariable(
                "SSC_grid",
                "f4",
                ("time", "y", "x"),
                zlib=True,
                complevel=4,
                fill_value=np.float32(np.nan)
            )

            time_var[:] = np.array(
                [t.strftime("%Y-%m-%d %H:%M:%S") for t in time_values],
                dtype=object
            )

            y_var[:] = np.arange(row_min, row_max + 1, dtype=np.int32)
            x_var[:] = np.arange(col_min, col_max + 1, dtype=np.int32)

            ssc_var[:, :, :] = ssc_cube
            ssc_var.long_name = "Grid suspended sediment concentration"
            ssc_var.units = "same_as_model_output"

            if "grid_id" in model_sed_df.columns:
                grid_id_2d = np.full((n_y, n_x), -9999, dtype=np.int32)
                grid_ids = model_sed_df["grid_id"].to_numpy(dtype=np.int32)

                for i in range(len(model_sed_df)):
                    grid_id_2d[row_idx[i], col_idx[i]] = grid_ids[i]

                grid_id_var = ds.createVariable(
                    "grid_id",
                    "i4",
                    ("y", "x"),
                    zlib=True,
                    complevel=4,
                    fill_value=-9999
                )
                grid_id_var[:, :] = grid_id_2d

            if "HRU_ID" in model_sed_df.columns:
                hru_id_2d = np.full((n_y, n_x), -9999, dtype=np.int32)
                hru_ids = model_sed_df["HRU_ID"].to_numpy(dtype=np.int32)

                for i in range(len(model_sed_df)):
                    hru_id_2d[row_idx[i], col_idx[i]] = hru_ids[i]

                hru_id_var = ds.createVariable(
                    "HRU_ID",
                    "i4",
                    ("y", "x"),
                    zlib=True,
                    complevel=4,
                    fill_value=-9999
                )
                hru_id_var[:, :] = hru_id_2d

            ds.description = "Final ErosionModel3 grid SSC as raster-style NetCDF"

    # =====================================================
    # 1) FINAL EROSION MODEL GRID OUTPUT
    # =====================================================
    model_sed_final = build_model_sed_from_params(
        optimised_param_dict,
        model_input
    )

    df_runoff_a, rain_a, common_times = align_forcing_data(df_runoff, rain)

    model_sed_final, time_cols = add_time_columns_to_model(
        model_sed_final,
        common_times
    )

    model_sed_final = calculate_grid_ssc(
        model_sed_final,
        df_runoff_a,
        rain_a,
        time_cols,
        df_swe=df_swe,
        cold_region=cold_region,
        zero_landcover_class0=zero_landcover_class0
    )

    model_sed_final.to_pickle(model_sed_pkl_file)
    model_sed_final.to_csv(model_sed_csv_file, index=False)

    save_model_sed_to_raster_netcdf(
        model_sed_df=model_sed_final,
        time_cols_in=time_cols,
        output_nc=model_sed_nc_file
    )

    # =====================================================
    # 2) HRU SSC OUTPUT
    # =====================================================
    SSC_hru_unrouted = compute_hru_ssc_from_grids_pergridrunoff(
        model_sed=model_sed_final,
        df_runoff=df_runoff,
        grid_hru_col="HRU_ID",
        runoff_hru_col="hruId",
        runoff_col="averageRoutedRunoff",
        return_wide=True
    )

    a_rout = optimised_param_dict["a_rout"]["value"]
    mt_rout = optimised_param_dict["mt_rout"]["value"]
    K_rout = optimised_param_dict["K_rout"]["value"]

    SSC_hru = route_ssc_hru_gamma(
        SSC_hru=SSC_hru_unrouted,
        a=a_rout,
        mt=mt_rout,
        K=K_rout,
        hru_col="HRU_ID"
    )

    SSC_hru.to_pickle(ssc_hru_pkl_file)
    SSC_hru.to_csv(ssc_hru_csv_file, index=False)

    # =====================================================
    # 3) HRU SSC FRACTION OUTPUT
    # =====================================================
    SSC_hru_frac = create_ssc_hru_fraction_dict(
        SSC_hru=SSC_hru,
        sand_hru_stat=sand_hru_stat,
        silt_hru_stat=silt_hru_stat,
        hru_col="HRU_ID",
        sand_col="mean_sand",
        silt_col="mean_silt",
        number_fractions=number_fractions
    )

    # SSC_hru_frac = {
    #     i: df.set_index("HRU_ID")
    #     for i, df in SSC_hru_frac.items()
    # }

    # with open(ssc_hru_frac_pkl_file, "wb") as f:
    #     pickle.dump(SSC_hru_frac, f)

    SSC_hru_frac = {
        i: df.set_index("HRU_ID")
        for i, df in SSC_hru_frac.items()
    }
    
    # =====================================================
    # Align SSC_hru_frac time columns with TempSedRout hydraulics
    # This prevents mismatch such as:
    # SSC_hru_frac = 37705 timesteps, h/q/Q/width = 37704 timesteps
    # =====================================================
    h_times = pd.DatetimeIndex(h.index).sort_values()
    common_routing_times = h_times.copy()
    
    for frac, df in SSC_hru_frac.items():
        ssc_times = pd.to_datetime(
            df.columns,
            format="t_%Y%m%d_%H%M%S"
        ).sort_values()
    
        common_routing_times = common_routing_times.intersection(ssc_times)
    
    h = h.loc[common_routing_times].copy()
    q = q.loc[common_routing_times].copy()
    Q = Q.loc[common_routing_times].copy()
    width = width.loc[common_routing_times].copy()
    
    SSC_hru_frac_common = {}
    
    for frac, df in SSC_hru_frac.items():
        col_time_map = {
            pd.to_datetime(col, format="t_%Y%m%d_%H%M%S"): col
            for col in df.columns
        }
    
        keep_cols = [
            col_time_map[t]
            for t in common_routing_times
            if t in col_time_map
        ]
    
        SSC_hru_frac_common[frac] = df.loc[:, keep_cols].copy()
    
    SSC_hru_frac = SSC_hru_frac_common
    
    # =====================================================
    # Add missing HRUs/reaches required by TempSedRout
    # =====================================================
    from utils import fill_missing_hru
    
    SSC_hru_frac = fill_missing_hru(
        SSC_hru_frac,
        river_gdf,
        id_col="LINKNO"
    )
    
    with open(ssc_hru_frac_pkl_file, "wb") as f:
        pickle.dump(SSC_hru_frac, f)


    for frac_id, frac_df in SSC_hru_frac.items():
        frac_csv_file = os.path.join(
            output_dir,
            f"{file_name}_SSC_hru_frac_frac{frac_id}.csv"
        )
        frac_df.to_csv(frac_csv_file)

    # =====================================================
    # 4) FINAL TEMPSSEDROUT OUTPUT
    # =====================================================
    SSC_river_frac_out, SSC_river_tot_out = run_final_tempsedrout(
        param_dict=optimised_param_dict,
        river_gdf=river_gdf,
        sediment_size=sediment_size,
        toml_file=toml_file,
        h=h,
        q=q,
        Q=Q,
        width=width,
        SSC_hru_frac=SSC_hru_frac,
        use_storage=use_storage,
        river_storage=river_storage,
        storage_data_type=storage_data_type
    )

    if isinstance(SSC_river_tot_out, pd.DataFrame):
        SSC_river_tot_out.to_csv(ssc_river_tot_csv_file, index=True)
        with open(ssc_river_tot_pkl_file, "wb") as f:
            pickle.dump(SSC_river_tot_out, f)

    with open(ssc_river_frac_pkl_file, "wb") as f:
        pickle.dump(SSC_river_frac_out, f)

    if isinstance(SSC_river_frac_out, dict):
        for frac_name, frac_obj in SSC_river_frac_out.items():
            frac_safe = str(frac_name).replace(" ", "_").replace("/", "_")

            frac_csv_file = os.path.join(
                output_dir,
                f"{file_name}_SSC_river_frac_out_{frac_safe}.csv"
            )

            frac_pkl_file = os.path.join(
                output_dir,
                f"{file_name}_SSC_river_frac_out_{frac_safe}.pkl"
            )

            if isinstance(frac_obj, pd.DataFrame):
                frac_obj.to_csv(frac_csv_file, index=True)

            with open(frac_pkl_file, "wb") as f:
                pickle.dump(frac_obj, f)

    # =====================================================
    # 5) OBSERVED VS SIMULATED OUTLET SSC
    # =====================================================
    obs, sim = prepare_obs_sim_series_tempsedrout(
        param_dict=optimised_param_dict,
        river_gdf=river_gdf,
        sediment_size=sediment_size,
        toml_file=toml_file,
        h=h,
        q=q,
        Q=Q,
        width=width,
        SSC_hru_frac=SSC_hru_frac,
        df_SSC_obs=df_SSC_obs,
        obs_time_col=obs_time_col,
        obs_value_col=obs_value_col,
        use_storage=use_storage,
        river_storage=river_storage,
        storage_data_type=storage_data_type
    )

    if obs is None or sim is None:
        obs_sim_df = None
    else:
        obs_sim_df = pd.DataFrame({
            "SSC_obs": obs,
            "SSC_sim": sim
        })

        obs_sim_df.to_csv(obs_sim_csv_file, index=False)

        with open(obs_sim_pkl_file, "wb") as f:
            pickle.dump(obs_sim_df, f)

    # =====================================================
    # 6) PARAMETERS
    # =====================================================
    param_rows = []

    for k, v in optimised_param_dict.items():
        param_rows.append({
            "section": "parameters",
            "name": k,
            "value": v.get("value"),
            "low": v.get("low"),
            "up": v.get("up"),
            "priority": v.get("priority")
        })

    df_params = pd.DataFrame(param_rows)

    # =====================================================
    # 7) BEST SCORE
    # =====================================================
    df_score = pd.DataFrame([{
        "section": "best_score",
        "name": "objective_value",
        "value": best_score
    }])

    # =====================================================
    # 8) BEST INDIVIDUAL
    # =====================================================
    best_ind = hof[0]

    df_best_ind = pd.DataFrame([
        {
            "section": "best_individual",
            "name": f"param_{i}",
            "value": val
        }
        for i, val in enumerate(best_ind)
    ])

    # =====================================================
    # 9) FINAL POPULATION
    # =====================================================
    pop_rows = []

    for i, ind in enumerate(pop):
        pop_rows.append({
            "section": "population_final",
            "name": f"ind_{i}",
            "fitness": ind.fitness.values[0] if len(ind.fitness.values) > 0 else None,
            "values": list(ind)
        })

    df_pop = pd.DataFrame(pop_rows)

    # =====================================================
    # 10) LOGBOOK
    # =====================================================
    log_rows = []

    for record in logbook:
        log_rows.append({
            "section": "logbook",
            "generation": record.get("gen"),
            "nevals": record.get("nevals"),
            "min": record.get("min"),
            "mean": record.get("mean"),
            "std": record.get("std")
        })

    df_log = pd.DataFrame(log_rows)

    # =====================================================
    # 11) GENERATION AND POPULATION HISTORY
    # =====================================================
    generation_history_df = (
        pd.DataFrame(generation_history)
        if generation_history is not None
        else pd.DataFrame()
    )

    population_history_df = (
        pd.DataFrame(population_history)
        if population_history is not None
        else pd.DataFrame()
    )

    if not generation_history_df.empty:
        generation_history_df.to_csv(generation_history_csv_file, index=False)
        with open(generation_history_pkl_file, "wb") as f:
            pickle.dump(generation_history_df, f)

    if not population_history_df.empty:
        population_history_df.to_csv(population_history_csv_file, index=False)
        with open(population_history_pkl_file, "wb") as f:
            pickle.dump(population_history_df, f)

    # =====================================================
    # 12) MAIN SUMMARY CSV AND PKL
    # =====================================================
    df_all = pd.concat(
        [df_params, df_score, df_best_ind, df_pop, df_log],
        ignore_index=True,
        sort=False
    )

    df_all.to_csv(main_csv_file, index=False)

    with open(main_pkl_file, "wb") as f:
        pickle.dump(
            {
                "optimised_param_dict": optimised_param_dict,
                "best_score": best_score,
                "pop": pop,
                "logbook": logbook,
                "hof": hof,
                "obs_sim_df": obs_sim_df,
                "model_sed_final": model_sed_final,
                "SSC_hru": SSC_hru,
                "SSC_hru_frac": SSC_hru_frac,
                "SSC_river_frac_out": SSC_river_frac_out,
                "SSC_river_tot_out": SSC_river_tot_out,
                "generation_history": generation_history_df,
                "population_history": population_history_df,
            },
            f
        )

    print(f"Saved main CSV: {main_csv_file}")
    print(f"Saved main PKL: {main_pkl_file}")
    print(f"Saved model_sed PKL: {model_sed_pkl_file}")
    print(f"Saved model_sed CSV: {model_sed_csv_file}")
    print(f"Saved model_sed raster NetCDF: {model_sed_nc_file}")
    print(f"Saved SSC_hru PKL: {ssc_hru_pkl_file}")
    print(f"Saved SSC_hru CSV: {ssc_hru_csv_file}")
    print(f"Saved SSC_hru_frac PKL: {ssc_hru_frac_pkl_file}")
    print(f"Saved SSC_river_tot_out CSV: {ssc_river_tot_csv_file}")
    print(f"Saved SSC_river_tot_out PKL: {ssc_river_tot_pkl_file}")
    print(f"Saved SSC_river_frac_out PKL: {ssc_river_frac_pkl_file}")


def save_validation_results3_basic(
    param_dict,
    model_input,
    df_runoff,
    rain,
    df_SSC_obs,
    sand_hru_stat,
    silt_hru_stat,
    river_gdf,
    sediment_size,
    toml_file,
    h,
    q,
    Q,
    width,
    output_dir,
    file_name="validation3",
    obs_time_col="time",
    obs_value_col="SSC",
    zero_landcover_class0=False,
    number_fractions=3,
    df_swe=None,
    cold_region=True,
    use_storage=False,
    river_storage=None,
    storage_data_type="length"
):
    """
    Run ErosionModel3 validation using fixed parameter values and save only:

    1. {file_name}_SSC_river_frac_outlet.csv
    2. {file_name}_obs_sim.csv
    3. {file_name}_SSC_river_frac_out_0.csv
    4. {file_name}_SSC_river_frac_out_1.csv
    5. {file_name}_SSC_river_frac_out_2.csv

    No grid-level, HRU-level, NetCDF, pickle, parameter-summary,
    or total-river-output files are saved.
    """

    import os
    import numpy as np
    import pandas as pd

    from utils import fill_missing_hru

    os.makedirs(output_dir, exist_ok=True)

    # =====================================================
    # FILE PATHS
    # =====================================================
    obs_sim_csv_file = os.path.join(
        output_dir,
        f"{file_name}_obs_sim.csv"
    )

    outlet_frac_csv_file = os.path.join(
        output_dir,
        f"{file_name}_SSC_river_frac_outlet.csv"
    )

    # =====================================================
    # 1) FINAL EROSION MODEL GRID OUTPUT
    #    This output is calculated but is not saved.
    # =====================================================
    model_sed_final = build_model_sed_from_params(
        param_dict,
        model_input
    )

    df_runoff_a, rain_a, common_times = align_forcing_data(
        df_runoff,
        rain
    )

    model_sed_final, time_cols = add_time_columns_to_model(
        model_sed_final,
        common_times
    )

    model_sed_final = calculate_grid_ssc(
        model_sed_final,
        df_runoff_a,
        rain_a,
        time_cols,
        df_swe=df_swe,
        cold_region=cold_region,
        zero_landcover_class0=zero_landcover_class0
    )

    # =====================================================
    # 2) HRU SSC OUTPUT
    #    This output is calculated but is not saved.
    # =====================================================
    SSC_hru_unrouted = compute_hru_ssc_from_grids_pergridrunoff(
        model_sed=model_sed_final,
        df_runoff=df_runoff_a,
        grid_hru_col="HRU_ID",
        runoff_hru_col="hruId",
        runoff_col="averageRoutedRunoff",
        return_wide=True
    )

    a_rout = param_dict["a_rout"]["value"]
    mt_rout = param_dict["mt_rout"]["value"]
    K_rout = param_dict["K_rout"]["value"]

    SSC_hru = route_ssc_hru_gamma(
        SSC_hru=SSC_hru_unrouted,
        a=a_rout,
        mt=mt_rout,
        K=K_rout,
        hru_col="HRU_ID"
    )

    # =====================================================
    # 3) HRU SSC FRACTIONS
    #    These outputs are calculated but are not saved.
    # =====================================================
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
        frac_id: frac_df.set_index("HRU_ID")
        for frac_id, frac_df in SSC_hru_frac.items()
    }

    # =====================================================
    # 4) ALIGN SSC AND HYDRAULIC TIMES
    # =====================================================
    h_times = pd.DatetimeIndex(h.index).sort_values()

    common_routing_times = h_times.copy()

    common_routing_times = common_routing_times.intersection(
        pd.DatetimeIndex(q.index)
    )

    common_routing_times = common_routing_times.intersection(
        pd.DatetimeIndex(Q.index)
    )

    common_routing_times = common_routing_times.intersection(
        pd.DatetimeIndex(width.index)
    )

    for frac_id, frac_df in SSC_hru_frac.items():

        ssc_times = pd.to_datetime(
            frac_df.columns,
            format="t_%Y%m%d_%H%M%S"
        ).sort_values()

        common_routing_times = common_routing_times.intersection(
            ssc_times
        )

    common_routing_times = common_routing_times.sort_values()

    if len(common_routing_times) == 0:
        raise ValueError(
            "No common timestamps were found between SSC_hru_frac "
            "and the TempSedRout hydraulic inputs."
        )

    h = h.loc[common_routing_times].copy()
    q = q.loc[common_routing_times].copy()
    Q = Q.loc[common_routing_times].copy()
    width = width.loc[common_routing_times].copy()

    SSC_hru_frac_common = {}

    for frac_id, frac_df in SSC_hru_frac.items():

        col_time_map = {
            pd.to_datetime(
                col,
                format="t_%Y%m%d_%H%M%S"
            ): col
            for col in frac_df.columns
        }

        keep_cols = [
            col_time_map[time_value]
            for time_value in common_routing_times
            if time_value in col_time_map
        ]

        SSC_hru_frac_common[frac_id] = (
            frac_df.loc[:, keep_cols].copy()
        )

    SSC_hru_frac = SSC_hru_frac_common

    # =====================================================
    # 5) ADD MISSING HRUs/REACHES
    # =====================================================
    SSC_hru_frac = fill_missing_hru(
        SSC_hru_frac,
        river_gdf,
        id_col="LINKNO"
    )

    # =====================================================
    # 6) RUN TEMPSSEDROUT
    # =====================================================
    SSC_river_frac_out, SSC_river_tot_out = run_final_tempsedrout(
        param_dict=param_dict,
        river_gdf=river_gdf,
        sediment_size=sediment_size,
        toml_file=toml_file,
        h=h,
        q=q,
        Q=Q,
        width=width,
        SSC_hru_frac=SSC_hru_frac,
        use_storage=use_storage,
        river_storage=river_storage,
        storage_data_type=storage_data_type
    )

    if not isinstance(SSC_river_frac_out, dict):
        raise TypeError(
            "SSC_river_frac_out must be a dictionary of pandas DataFrames."
        )

    # =====================================================
    # 7) SAVE THE THREE FULL RIVER FRACTION OUTPUTS
    # =====================================================
    expected_fraction_keys = list(range(number_fractions))

    for frac_id in expected_fraction_keys:

        if frac_id not in SSC_river_frac_out:
            raise KeyError(
                f"Fraction {frac_id} is missing from SSC_river_frac_out."
            )

        frac_df = SSC_river_frac_out[frac_id]

        if not isinstance(frac_df, pd.DataFrame):
            raise TypeError(
                f"SSC_river_frac_out[{frac_id}] must be a pandas DataFrame."
            )

        frac_csv_file = os.path.join(
            output_dir,
            f"{file_name}_SSC_river_frac_out_{frac_id}.csv"
        )

        frac_df.to_csv(
            frac_csv_file,
            index=True,
            index_label="time"
        )

        print(f"Saved river fraction {frac_id}: {frac_csv_file}")

    # =====================================================
    # 8) FIND OUTLET REACH
    # =====================================================
    downstream_ids = set(
        river_gdf["DSLINKNO"].dropna()
    )

    link_ids = set(
        river_gdf["LINKNO"].dropna()
    )

    outlet_rows = river_gdf.loc[
        river_gdf["DSLINKNO"].isin(
            downstream_ids - link_ids
        ),
        "LINKNO"
    ]

    if outlet_rows.empty:
        raise ValueError(
            "Could not identify the outlet reach from LINKNO and DSLINKNO."
        )

    reach_out_id = outlet_rows.iat[0]

    # Account for possible integer/string differences in column names.
    first_frac_df = SSC_river_frac_out[expected_fraction_keys[0]]

    if reach_out_id not in first_frac_df.columns:
        reach_out_id_str = str(reach_out_id)

        if reach_out_id_str in first_frac_df.columns:
            reach_out_id = reach_out_id_str
        else:
            raise KeyError(
                f"Outlet reach {reach_out_id} is not present in "
                "SSC_river_frac_out columns."
            )

    # =====================================================
    # 9) SAVE OUTLET SSC FOR EACH FRACTION
    # =====================================================
    SSC_river_frac_outlet = pd.DataFrame({
        f"frac_{frac_id}": SSC_river_frac_out[frac_id][reach_out_id]
        for frac_id in expected_fraction_keys
    })

    SSC_river_frac_outlet.index = pd.to_datetime(
        SSC_river_frac_outlet.index
    )

    SSC_river_frac_outlet.index.name = "time"

    SSC_river_frac_outlet.to_csv(
        outlet_frac_csv_file,
        index=True
    )

    print(
        "Saved outlet fraction SSC:",
        outlet_frac_csv_file
    )

    # =====================================================
    # 10) OBSERVED VS SIMULATED OUTLET SSC
    # =====================================================
    obs, sim = prepare_obs_sim_series_tempsedrout(
        param_dict=param_dict,
        river_gdf=river_gdf,
        sediment_size=sediment_size,
        toml_file=toml_file,
        h=h,
        q=q,
        Q=Q,
        width=width,
        SSC_hru_frac=SSC_hru_frac,
        df_SSC_obs=df_SSC_obs,
        obs_time_col=obs_time_col,
        obs_value_col=obs_value_col,
        use_storage=use_storage,
        river_storage=river_storage,
        storage_data_type=storage_data_type
    )

    if obs is None or sim is None:
        raise ValueError(
            "Observed and simulated outlet SSC series could not be prepared."
        )

    obs_values = np.asarray(obs, dtype=float)
    sim_values = np.asarray(sim, dtype=float)

    if len(obs_values) != len(sim_values):
        raise ValueError(
            "Observed and simulated SSC series have different lengths: "
            f"{len(obs_values)} and {len(sim_values)}."
        )

    obs_sim_df = pd.DataFrame({
        "SSC_obs": obs_values,
        "SSC_sim": sim_values
    })

    obs_sim_df.to_csv(
        obs_sim_csv_file,
        index=False
    )

    print(f"Saved observed vs simulated SSC: {obs_sim_csv_file}")

    # =====================================================
    # 11) RETURN RESULTS WITHOUT SAVING EXTRA FILES
    # =====================================================
    return {
        "reach_out_id": reach_out_id,
        "obs_sim_df": obs_sim_df,
        "SSC_river_frac_outlet": SSC_river_frac_outlet,
        "SSC_river_frac_out": SSC_river_frac_out,
        "SSC_river_tot_out": SSC_river_tot_out
    }

def save_validation_results3_full(
    param_dict,
    model_input,
    df_runoff,
    rain,
    cat_hru,
    df_SSC_obs,
    sand_hru_stat,
    silt_hru_stat,
    river_gdf,
    sediment_size,
    toml_file,
    h,
    q,
    Q,
    width,
    output_dir,
    file_name="validation3",
    model_sed_pkl_name=None,
    obs_time_col="time",
    obs_value_col="SSC",
    zero_landcover_class0=False,
    number_fractions=3,
    df_swe=None,
    cold_region=True,
    use_storage=False,
    river_storage=None,
    storage_data_type="length",
    objective="log_rmse"
):
    """
    Save full ErosionModel3 validation results using fixed parameter values.

    This is similar to save_optimisation_results3_full, but it does not require:
    - best_score
    - pop
    - logbook
    - hof
    - generation_history
    - population_history
    """

    import os
    import pickle
    import numpy as np
    import pandas as pd
    from netCDF4 import Dataset

    os.makedirs(output_dir, exist_ok=True)

    # =====================================================
    # FILE PATHS
    # =====================================================
    main_csv_file = os.path.join(output_dir, f"{file_name}_main.csv")
    main_pkl_file = os.path.join(output_dir, f"{file_name}_main.pkl")

    if model_sed_pkl_name is None:
        model_sed_pkl_file = os.path.join(output_dir, f"{file_name}_model_sed_final.pkl")
    else:
        model_sed_pkl_file = os.path.join(output_dir, model_sed_pkl_name)

    model_sed_csv_file = os.path.join(output_dir, f"{file_name}_model_sed_final.csv")
    model_sed_nc_file = os.path.join(output_dir, f"{file_name}_model_sed_final_raster.nc")

    ssc_hru_pkl_file = os.path.join(output_dir, f"{file_name}_SSC_hru.pkl")
    ssc_hru_csv_file = os.path.join(output_dir, f"{file_name}_SSC_hru.csv")

    ssc_hru_frac_pkl_file = os.path.join(output_dir, f"{file_name}_SSC_hru_frac.pkl")

    ssc_river_tot_csv_file = os.path.join(output_dir, f"{file_name}_SSC_river_tot_out.csv")
    ssc_river_tot_pkl_file = os.path.join(output_dir, f"{file_name}_SSC_river_tot_out.pkl")

    ssc_river_frac_pkl_file = os.path.join(output_dir, f"{file_name}_SSC_river_frac_out.pkl")

    obs_sim_csv_file = os.path.join(output_dir, f"{file_name}_obs_sim.csv")
    obs_sim_pkl_file = os.path.join(output_dir, f"{file_name}_obs_sim.pkl")

    outlet_frac_csv_file = os.path.join(output_dir, f"{file_name}_SSC_river_frac_outlet.csv")
    outlet_frac_pkl_file = os.path.join(output_dir, f"{file_name}_SSC_river_frac_outlet.pkl")

    # =====================================================
    # SMALL HELPER: SAVE GRID SSC AS RASTER-STYLE NETCDF
    # =====================================================
    def save_model_sed_to_raster_netcdf(model_sed_df, time_cols_in, output_nc):

        required_cols = ["row", "col"]
        missing_cols = [c for c in required_cols if c not in model_sed_df.columns]

        if missing_cols:
            raise ValueError(
                f"model_sed_df must contain columns {required_cols}. Missing: {missing_cols}"
            )

        time_values = pd.to_datetime(
            [c.replace("t_", "") for c in time_cols_in],
            format="%Y%m%d_%H%M%S"
        )

        n_time = len(time_values)

        row_vals = model_sed_df["row"].to_numpy(dtype=int)
        col_vals = model_sed_df["col"].to_numpy(dtype=int)

        row_min = int(np.min(row_vals))
        row_max = int(np.max(row_vals))
        col_min = int(np.min(col_vals))
        col_max = int(np.max(col_vals))

        n_y = row_max - row_min + 1
        n_x = col_max - col_min + 1

        row_idx = row_vals - row_min
        col_idx = col_vals - col_min

        ssc_cube = np.full((n_time, n_y, n_x), np.nan, dtype=np.float32)
        ssc_values = model_sed_df[time_cols_in].to_numpy(dtype=np.float32)

        for i in range(len(model_sed_df)):
            ssc_cube[:, row_idx[i], col_idx[i]] = ssc_values[i, :]

        with Dataset(output_nc, "w", format="NETCDF4") as ds:
            ds.createDimension("time", n_time)
            ds.createDimension("y", n_y)
            ds.createDimension("x", n_x)

            time_var = ds.createVariable("time", str, ("time",))
            y_var = ds.createVariable("y", "i4", ("y",))
            x_var = ds.createVariable("x", "i4", ("x",))

            ssc_var = ds.createVariable(
                "SSC_grid",
                "f4",
                ("time", "y", "x"),
                zlib=True,
                complevel=4,
                fill_value=np.float32(np.nan)
            )

            time_var[:] = np.array(
                [t.strftime("%Y-%m-%d %H:%M:%S") for t in time_values],
                dtype=object
            )

            y_var[:] = np.arange(row_min, row_max + 1, dtype=np.int32)
            x_var[:] = np.arange(col_min, col_max + 1, dtype=np.int32)

            ssc_var[:, :, :] = ssc_cube
            ssc_var.long_name = "Grid suspended sediment concentration"
            ssc_var.units = "same_as_model_output"

            if "grid_id" in model_sed_df.columns:
                grid_id_2d = np.full((n_y, n_x), -9999, dtype=np.int32)
                grid_ids = model_sed_df["grid_id"].to_numpy(dtype=np.int32)

                for i in range(len(model_sed_df)):
                    grid_id_2d[row_idx[i], col_idx[i]] = grid_ids[i]

                grid_id_var = ds.createVariable(
                    "grid_id",
                    "i4",
                    ("y", "x"),
                    zlib=True,
                    complevel=4,
                    fill_value=-9999
                )
                grid_id_var[:, :] = grid_id_2d

            if "HRU_ID" in model_sed_df.columns:
                hru_id_2d = np.full((n_y, n_x), -9999, dtype=np.int32)
                hru_ids = model_sed_df["HRU_ID"].to_numpy(dtype=np.int32)

                for i in range(len(model_sed_df)):
                    hru_id_2d[row_idx[i], col_idx[i]] = hru_ids[i]

                hru_id_var = ds.createVariable(
                    "HRU_ID",
                    "i4",
                    ("y", "x"),
                    zlib=True,
                    complevel=4,
                    fill_value=-9999
                )
                hru_id_var[:, :] = hru_id_2d

            ds.description = "Validation ErosionModel3 grid SSC as raster-style NetCDF"

    # =====================================================
    # 1) FINAL EROSION MODEL GRID OUTPUT
    # =====================================================
    model_sed_final = build_model_sed_from_params(
        param_dict,
        model_input
    )

    df_runoff_a, rain_a, common_times = align_forcing_data(df_runoff, rain)

    model_sed_final, time_cols = add_time_columns_to_model(
        model_sed_final,
        common_times
    )

    model_sed_final = calculate_grid_ssc(
        model_sed_final,
        df_runoff_a,
        rain_a,
        time_cols,
        df_swe=df_swe,
        cold_region=cold_region,
        zero_landcover_class0=zero_landcover_class0
    )

    model_sed_final.to_pickle(model_sed_pkl_file)
    model_sed_final.to_csv(model_sed_csv_file, index=False)

    save_model_sed_to_raster_netcdf(
        model_sed_df=model_sed_final,
        time_cols_in=time_cols,
        output_nc=model_sed_nc_file
    )

    # =====================================================
    # 2) HRU SSC OUTPUT
    # =====================================================
    SSC_hru_unrouted = compute_hru_ssc_from_grids_pergridrunoff(
        model_sed=model_sed_final,
        df_runoff=df_runoff,
        grid_hru_col="HRU_ID",
        runoff_hru_col="hruId",
        runoff_col="averageRoutedRunoff",
        return_wide=True
    )

    a_rout = param_dict["a_rout"]["value"]
    mt_rout = param_dict["mt_rout"]["value"]
    K_rout = param_dict["K_rout"]["value"]

    SSC_hru = route_ssc_hru_gamma(
        SSC_hru=SSC_hru_unrouted,
        a=a_rout,
        mt=mt_rout,
        K=K_rout,
        hru_col="HRU_ID"
    )

    SSC_hru.to_pickle(ssc_hru_pkl_file)
    SSC_hru.to_csv(ssc_hru_csv_file, index=False)

    # =====================================================
    # 3) HRU SSC FRACTION OUTPUT
    # =====================================================
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

    h_times = pd.DatetimeIndex(h.index).sort_values()
    common_routing_times = h_times.copy()

    for frac, df in SSC_hru_frac.items():
        ssc_times = pd.to_datetime(
            df.columns,
            format="t_%Y%m%d_%H%M%S"
        ).sort_values()

        common_routing_times = common_routing_times.intersection(ssc_times)

    h = h.loc[common_routing_times].copy()
    q = q.loc[common_routing_times].copy()
    Q = Q.loc[common_routing_times].copy()
    width = width.loc[common_routing_times].copy()

    SSC_hru_frac_common = {}

    for frac, df in SSC_hru_frac.items():
        col_time_map = {
            pd.to_datetime(col, format="t_%Y%m%d_%H%M%S"): col
            for col in df.columns
        }

        keep_cols = [
            col_time_map[t]
            for t in common_routing_times
            if t in col_time_map
        ]

        SSC_hru_frac_common[frac] = df.loc[:, keep_cols].copy()

    SSC_hru_frac = SSC_hru_frac_common

    from utils import fill_missing_hru

    SSC_hru_frac = fill_missing_hru(
        SSC_hru_frac,
        river_gdf,
        id_col="LINKNO"
    )

    with open(ssc_hru_frac_pkl_file, "wb") as f:
        pickle.dump(SSC_hru_frac, f)

    for frac_id, frac_df in SSC_hru_frac.items():
        frac_csv_file = os.path.join(
            output_dir,
            f"{file_name}_SSC_hru_frac_frac{frac_id}.csv"
        )
        frac_df.to_csv(frac_csv_file)

    # =====================================================
    # 4) FINAL TEMPSSEDROUT OUTPUT
    # =====================================================
    SSC_river_frac_out, SSC_river_tot_out = run_final_tempsedrout(
        param_dict=param_dict,
        river_gdf=river_gdf,
        sediment_size=sediment_size,
        toml_file=toml_file,
        h=h,
        q=q,
        Q=Q,
        width=width,
        SSC_hru_frac=SSC_hru_frac,
        use_storage=use_storage,
        river_storage=river_storage,
        storage_data_type=storage_data_type
    )

    if isinstance(SSC_river_tot_out, pd.DataFrame):
        SSC_river_tot_out.to_csv(ssc_river_tot_csv_file, index=True)
        with open(ssc_river_tot_pkl_file, "wb") as f:
            pickle.dump(SSC_river_tot_out, f)

    with open(ssc_river_frac_pkl_file, "wb") as f:
        pickle.dump(SSC_river_frac_out, f)

    if isinstance(SSC_river_frac_out, dict):
        for frac_name, frac_obj in SSC_river_frac_out.items():
            frac_safe = str(frac_name).replace(" ", "_").replace("/", "_")

            frac_csv_file = os.path.join(
                output_dir,
                f"{file_name}_SSC_river_frac_out_{frac_safe}.csv"
            )

            frac_pkl_file = os.path.join(
                output_dir,
                f"{file_name}_SSC_river_frac_out_{frac_safe}.pkl"
            )

            if isinstance(frac_obj, pd.DataFrame):
                frac_obj.to_csv(frac_csv_file, index=True)

            with open(frac_pkl_file, "wb") as f:
                pickle.dump(frac_obj, f)

    # =====================================================
    # 4b) OUTLET FRACTION OUTPUT
    # =====================================================
    SSC_river_frac_outlet = None

    try:
        reach_out_id = river_gdf.loc[
            river_gdf["DSLINKNO"].isin(
                set(river_gdf["DSLINKNO"]) - set(river_gdf["LINKNO"])
            ),
            "LINKNO"
        ].iat[0]

        if isinstance(SSC_river_frac_out, dict):
            SSC_river_frac_outlet = pd.DataFrame({
                f"frac_{frac_key}": frac_df[reach_out_id]
                for frac_key, frac_df in SSC_river_frac_out.items()
                if isinstance(frac_df, pd.DataFrame) and reach_out_id in frac_df.columns
            })

            if not SSC_river_frac_outlet.empty:
                SSC_river_frac_outlet.to_csv(outlet_frac_csv_file, index=True)
                with open(outlet_frac_pkl_file, "wb") as f:
                    pickle.dump(SSC_river_frac_outlet, f)

    except Exception as e:
        print(f"Warning: could not save outlet fraction SSC: {e}")

    # =====================================================
    # 5) OBSERVED VS SIMULATED OUTLET SSC
    # =====================================================
    obs, sim = prepare_obs_sim_series_tempsedrout(
        param_dict=param_dict,
        river_gdf=river_gdf,
        sediment_size=sediment_size,
        toml_file=toml_file,
        h=h,
        q=q,
        Q=Q,
        width=width,
        SSC_hru_frac=SSC_hru_frac,
        df_SSC_obs=df_SSC_obs,
        obs_time_col=obs_time_col,
        obs_value_col=obs_value_col,
        use_storage=use_storage,
        river_storage=river_storage,
        storage_data_type=storage_data_type
    )

    if obs is None or sim is None:
        obs_sim_df = None
        validation_score = None
    else:
        obs_sim_df = pd.DataFrame({
            "SSC_obs": obs,
            "SSC_sim": sim
        })

        validation_score = objective_from_series(
            obs,
            sim,
            objective=objective
        )

        obs_sim_df.to_csv(obs_sim_csv_file, index=False)

        with open(obs_sim_pkl_file, "wb") as f:
            pickle.dump(obs_sim_df, f)

    # =====================================================
    # 6) PARAMETERS
    # =====================================================
    param_rows = []

    for k, v in param_dict.items():
        param_rows.append({
            "section": "parameters",
            "name": k,
            "value": v.get("value"),
            "low": v.get("low"),
            "up": v.get("up"),
            "priority": v.get("priority")
        })

    df_params = pd.DataFrame(param_rows)

    df_score = pd.DataFrame([{
        "section": "validation_score",
        "name": objective,
        "value": validation_score
    }])

    df_all = pd.concat(
        [df_params, df_score],
        ignore_index=True,
        sort=False
    )

    df_all.to_csv(main_csv_file, index=False)

    with open(main_pkl_file, "wb") as f:
        pickle.dump(
            {
                "param_dict": param_dict,
                "validation_score": validation_score,
                "objective": objective,
                "obs_sim_df": obs_sim_df,
                "model_sed_final": model_sed_final,
                "SSC_hru": SSC_hru,
                "SSC_hru_frac": SSC_hru_frac,
                "SSC_river_frac_out": SSC_river_frac_out,
                "SSC_river_tot_out": SSC_river_tot_out,
                "SSC_river_frac_outlet": SSC_river_frac_outlet,
            },
            f
        )

    print(f"Saved validation main CSV: {main_csv_file}")
    print(f"Saved validation main PKL: {main_pkl_file}")
    print(f"Saved model_sed PKL: {model_sed_pkl_file}")
    print(f"Saved model_sed CSV: {model_sed_csv_file}")
    print(f"Saved model_sed raster NetCDF: {model_sed_nc_file}")
    print(f"Saved SSC_hru PKL: {ssc_hru_pkl_file}")
    print(f"Saved SSC_hru CSV: {ssc_hru_csv_file}")
    print(f"Saved SSC_hru_frac PKL: {ssc_hru_frac_pkl_file}")
    print(f"Saved SSC_river_tot_out CSV: {ssc_river_tot_csv_file}")
    print(f"Saved SSC_river_tot_out PKL: {ssc_river_tot_pkl_file}")
    print(f"Saved SSC_river_frac_out PKL: {ssc_river_frac_pkl_file}")

    return {
        "validation_score": validation_score,
        "obs_sim_df": obs_sim_df,
        "model_sed_final": model_sed_final,
        "SSC_hru": SSC_hru,
        "SSC_hru_frac": SSC_hru_frac,
        "SSC_river_frac_out": SSC_river_frac_out,
        "SSC_river_tot_out": SSC_river_tot_out,
        "SSC_river_frac_outlet": SSC_river_frac_outlet,
    }


# end of optimisation_updated.py code