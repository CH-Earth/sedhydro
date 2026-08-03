#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SedHydro Utility Functions
--------------------------

Core utility functions used throughout the SedHydro modelling framework.

This module provides functionality for:

- Reading and processing GIS datasets
- Raster and vector spatial analysis
- Catchment and HRU preprocessing
- NetCDF data extraction and handling
- Coordinate reference system (CRS) management
- Sediment and hydrological data processing
- File and directory management
- Statistical and routing support functions

Imported by:
    SedHydro.py
    SedHydro_mp.py
    TempSedRout_function.py
    TempSedRout_storage_function.py
    optimisation_updated.py

Developed by the CAPE Team
University of Calgary

"""

import rasterio
from rasterio.mask import mask
import numpy as np
import pandas as pd
from pathlib import Path
from pyproj import CRS
import shapely
import geopandas as gpd
import os
import glob
from netCDF4 import Dataset, num2date, date2num
from datetime import datetime
import math
from scipy.stats import gamma
import geopandas as gpd
import shapely.geometry
import rasterio.features
from pyproj import CRS
from netCDF4 import Dataset, num2date
#
def raster_cut(cat_gdf, mapBox):
    """
    Clip a raster to the geometry in cat_gdf and return array + spatial metadata.

    Parameters
    ----------
    cat_gdf : geopandas.GeoDataFrame
        Catchment boundary polygon(s). Must have a CRS.
    mapBox : rasterio.io.DatasetReader OR str/path
        Open rasterio dataset or filepath to raster.

    Returns
    -------
    map_array : numpy.ndarray (float32)
        Clipped raster (single band) with nodata converted to NaN.
    map_shape : tuple
        Shape of clipped raster (rows, cols).
    nodata_out : float
        Output nodata value (NaN).
    transform_out : affine.Affine
        Affine transform for the clipped raster (needed for overlay/writing).
    crs_out : rasterio.crs.CRS
        CRS of the clipped raster (needed for overlay with vector grid).
    profile_out : dict
        Raster profile updated for the clipped raster (useful for writing GeoTIFF).
    """

    # Allow passing a filepath
    if isinstance(mapBox, (str, bytes)):
        with rasterio.open(mapBox) as src:
            return raster_cut(cat_gdf, src)

    src = mapBox

    # --- Ensure CRS match ---
    if cat_gdf.crs != src.crs:
        cat_gdf = cat_gdf.to_crs(src.crs)

    # --- Geometry to mask ---
    geoms = [geom for geom in cat_gdf.geometry if geom is not None]

    # Decide numeric nodata to use during mask
    nodata_in = src.nodata if src.nodata is not None else 0

    # --- Clip ---
    clipped, transform_out = mask(
        src,
        geoms,
        crop=True,
        filled=True,
        nodata=nodata_in
    )

    # Single band -> array
    map_array = clipped[0].astype(np.float32)

    # Convert nodata to NaN for analysis/classification
    map_array[map_array == nodata_in] = np.nan

    map_shape = map_array.shape
    crs_out = src.crs

    # Updated profile for writing output rasters later
    profile_out = src.profile.copy()
    profile_out.update(
        height=map_shape[0],
        width=map_shape[1],
        transform=transform_out,
        count=1,
        dtype="float32",
        nodata=np.nan  # NOTE: many GeoTIFFs prefer numeric nodata; handle on write if needed
    )

    nodata_out = np.nan
    return map_array, map_shape, nodata_out, transform_out, crs_out, profile_out

#

def read_tbl_like_text(path):
    
    """
    Parses SUMMA-style text tables/configs with:
      - comment lines starting with '!'
      - inline comments after '!'
      - 'name | integer' lines
      - 'name value' lines (value may be multi-token)
      - bare 'name' lines

    Returns a DataFrame with columns:
      - name
      - value   (string or None)
      - levels  (int or None)
    """
    records = []

    for raw in Path(path).read_text().splitlines():
        line = raw.strip()

        # skip empty + full-line comments
        if not line or line.startswith("!"):
            continue

        # remove inline comment
        line = line.split("!", 1)[0].strip()
        if not line:
            continue

        # case A: name | levels
        if "|" in line:
            name, rest = [x.strip() for x in line.split("|", 1)]
            levels = int(rest.split()[0])  # first token after |
            records.append({"name": name, "value": None, "levels": levels})
            continue

        # case B/C: split tokens
        parts = line.split()
        name = parts[0]

        # case C: bare variable name
        if len(parts) == 1:
            records.append({"name": name, "value": None, "levels": None})
        else:
            # case B: decision-style "name value..."
            value = " ".join(parts[1:]).strip()
            records.append({"name": name, "value": value, "levels": None})
    # return records     
    return pd.DataFrame(records)


#

def _utm_crs_for_gdf(gdf):
    """Pick a UTM CRS based on the GeoDataFrame centroid."""
    

    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS.")

    g_ll = gdf.to_crs("EPSG:4326")
    lon = float(g_ll.geometry.unary_union.centroid.x)
    lat = float(g_ll.geometry.unary_union.centroid.y)
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def make_catchment_grid(
    cat_gdf,
    cell_size,
    clip_to_catchment=False,
    cat_hru=None,
    hru_id_col="HRU_ID",
    assign_hru_method="largest_overlap"
):
    """
    Create a square grid overlay for a catchment boundary, with optional HRU assignment.

    Parameters
    ----------
    cat_gdf : geopandas.GeoDataFrame
        Catchment boundary polygon(s). Must have a defined CRS.

    cell_size : float
        Grid cell size in meters.

    clip_to_catchment : bool, optional
        If False, keeps full square cells intersecting the catchment.
        If True, clips cells to the exact catchment boundary.

    cat_hru : geopandas.GeoDataFrame, optional
        HRU polygons. If provided, assigns HRU IDs to grid cells.

    hru_id_col : str, optional
        Column name in cat_hru containing HRU IDs. Default is "HRU_ID".

    assign_hru_method : str, optional
        Method to assign HRU ID to each grid cell:
        - "centroid": use grid-cell centroid
        - "largest_overlap": use HRU with largest intersection area

    Returns
    -------
    grid_gdf : geopandas.GeoDataFrame
        Grid cells with columns:
            - row
            - col
            - grid_id
            - geometry
            - HRU_ID (if cat_hru is provided)
    """


    if cat_gdf.crs is None:
        raise ValueError("cat_gdf.crs is None. Define CRS first.")

    # Reproject catchment to projected CRS if needed
    if cat_gdf.crs.is_geographic:
        utm = _utm_crs_for_gdf(cat_gdf)
        g = cat_gdf.to_crs(utm)
    else:
        g = cat_gdf.copy()

    catchment_geom = g.geometry.unary_union

    # Bounding box
    minx, miny, maxx, maxy = catchment_geom.bounds

    # Snap bounds to cell size
    minx = np.floor(minx / cell_size) * cell_size
    miny = np.floor(miny / cell_size) * cell_size
    maxx = np.ceil(maxx / cell_size) * cell_size
    maxy = np.ceil(maxy / cell_size) * cell_size

    # Build grid coords
    xs = np.arange(minx, maxx, cell_size)
    ys = np.arange(maxy, miny, -cell_size)

    cells = []
    rows = []
    cols = []
    ids = []

    r = 0
    for y_top in ys[:-1]:
        c = 0
        y_bottom = y_top - cell_size
        for x_left in xs:
            x_right = x_left + cell_size
            geom = shapely.geometry.box(x_left, y_bottom, x_right, y_top)

            if geom.intersects(catchment_geom):
                cells.append(geom)
                rows.append(r)
                cols.append(c)
                ids.append(r * len(xs) + c + 1)
            c += 1
        r += 1

    grid = gpd.GeoDataFrame(
        {"row": rows, "col": cols, "grid_id": ids},
        geometry=cells,
        crs=g.crs
    )

    # Optional clipping to catchment
    if clip_to_catchment:
        grid["geometry"] = grid.geometry.intersection(catchment_geom)

    # Optional HRU assignment
    if cat_hru is not None:
        if cat_hru.crs is None:
            raise ValueError("cat_hru.crs is None. Define CRS first.")
        if hru_id_col not in cat_hru.columns:
            raise ValueError(f"'{hru_id_col}' not found in cat_hru columns.")

        # Reproject HRUs to same CRS as grid
        hru = cat_hru.to_crs(grid.crs).copy()

        if assign_hru_method == "centroid":
            grid_pts = grid.copy()
            grid_pts["geometry"] = grid.centroid

            joined = gpd.sjoin(
                grid_pts,
                hru[[hru_id_col, "geometry"]],
                how="left",
                predicate="within"
            )

            grid[hru_id_col] = joined[hru_id_col].values

        elif assign_hru_method == "largest_overlap":
            # Intersect grid with HRUs
            inter = gpd.overlay(
                grid[["grid_id", "row", "col", "geometry"]],
                hru[[hru_id_col, "geometry"]],
                how="intersection"
            )

            if len(inter) > 0:
                inter["overlap_area"] = inter.geometry.area

                # For each grid_id, keep HRU with maximum overlap
                inter = inter.sort_values(["grid_id", "overlap_area"], ascending=[True, False])
                best = inter.drop_duplicates(subset="grid_id")

                grid = grid.merge(
                    best[["grid_id", hru_id_col]],
                    on="grid_id",
                    how="left"
                )
            else:
                grid[hru_id_col] = np.nan

        else:
            raise ValueError("assign_hru_method must be 'centroid' or 'largest_overlap'")

    return grid



#
def extract_class_from_grid (
    grid,
    map_array,
    map_shape,
    map_transform,
    map_crs,
    feature_name,
    grid_id_col="grid_id",
    fill_grid_id=np.nan,
    return_long_fractions=False,
):
    """
    Overlay a (classified) raster map with a vector grid and compute:
      1) Dominant class per grid cell
      2) Fraction of each class per grid cell

    Parameters
    ----------
    grid : geopandas.GeoDataFrame
        Grid polygons with an ID column (default: 'grid_id').
    map_array : numpy.ndarray
        2D raster array of classes (NaN allowed; NaNs ignored).
    map_shape : tuple
        Raster shape (rows, cols) matching map_array.
    map_transform : affine.Affine
        Raster affine transform.
    map_crs : rasterio.crs.CRS or pyproj.CRS
        Raster CRS.
    feature_name : str
        Name used to tag output columns, e.g. "landcover" -> dominant_class_landcover.
    grid_id_col : str, optional
        Column name in `grid` that contains unique grid IDs.
    fill_grid_id : int, optional
        Fill value for pixels not covered by any grid cell.
    return_long_fractions : bool, optional
        If True, also return long-format fractions table (grid_id, class, fraction).

    Returns
    -------
    dominant_df : pandas.DataFrame
        Columns: [grid_id_col, 'dominant_class']
    fractions_df : pandas.DataFrame
        Wide-format fractions table: [grid_id_col, <class1>, <class2>, ...]
    (optional) fractions_long_df : pandas.DataFrame
        Long-format fractions: [grid_id_col, 'class', 'fraction']
    """
    feature_name = str(feature_name).strip()
    if not feature_name:
        raise ValueError("feature_name must be a non-empty string (e.g., 'landcover', 'geol').")

    dom_col = f"dominant_class_{feature_name}"
    frac_col = f"fraction_{feature_name}"

    # --- Ensure CRS match ---
    if grid.crs != map_crs:
        grid = grid.to_crs(map_crs)

    if grid_id_col not in grid.columns:
        raise KeyError(f"'{grid_id_col}' not found in grid columns: {list(grid.columns)}")

    # --- Rasterize grid IDs to map pixel grid ---
    # shapes = ((geom, float(gid)) for geom, gid in zip(grid.geometry, grid[grid_id_col]))

    # grid_id_raster = rasterio.features.rasterize(
    #     shapes=shapes,
    #     out_shape=map_shape,
    #     transform=map_transform,
    #     fill=fill_grid_id,
    #     dtype=np.float32
    # )

    # # --- Valid pixels: inside grid cell AND map value finite ---
    # valid = (~np.isnan(grid_id_raster)) & np.isfinite(map_array)

    # gid = grid_id_raster[valid].astype(np.int32)
    
    # Ensure grid IDs are integer and non-missing
    if grid[grid_id_col].isna().any():
        raise ValueError(f"Column '{grid_id_col}' contains NaN values.")
    
    if not np.issubdtype(grid[grid_id_col].dtype, np.integer):
        # convert safely if possible
        if np.all(np.equal(grid[grid_id_col], grid[grid_id_col].astype(int))):
            grid = grid.copy()
            grid[grid_id_col] = grid[grid_id_col].astype(np.int32)
        else:
            raise ValueError(f"Column '{grid_id_col}' must contain integer-like values.")
    
    fill_value = -1
    
    shapes = ((geom, int(gid)) for geom, gid in zip(grid.geometry, grid[grid_id_col]))
    
    grid_id_raster = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=map_shape,
        transform=map_transform,
        fill=fill_value,
        dtype=np.int32
    )
    
    valid = (grid_id_raster != fill_value) & np.isfinite(map_array)
    
    gid = grid_id_raster[valid]
    
    cls = map_array[valid].astype(np.int32)

    df = pd.DataFrame({grid_id_col: gid, "class": cls})

    # 1) Dominant class per grid cell
    dominant_df = (
        df.groupby([grid_id_col, "class"])
          .size()
          .reset_index(name="n")
          .sort_values([grid_id_col, "n"], ascending=[True, False])
          .drop_duplicates(grid_id_col)
          .rename(columns={"class": dom_col})
          .loc[:, [grid_id_col, dom_col]]
    )

    # 2) Fractions per grid cell (long)
    counts = (
        df.groupby([grid_id_col, "class"])
          .size()
          .reset_index(name="n")
    )

    totals = (
        counts.groupby(grid_id_col)["n"]
              .sum()
              .reset_index(name="n_total")
    )

    fractions_long_df = counts.merge(totals, on=grid_id_col)
    fractions_long_df[frac_col] = fractions_long_df["n"] / fractions_long_df["n_total"]
    fractions_long_df = fractions_long_df.loc[:, [grid_id_col, "class", frac_col]]

    # Wide-format (class codes become columns)
    fractions_df = (
        fractions_long_df.pivot_table(
            index=grid_id_col,
            columns="class",
            values=frac_col,
            fill_value=0
        )
        .reset_index()
    )

    if return_long_fractions:
        return dominant_df, fractions_df, fractions_long_df

    return dominant_df, fractions_df



#

def extract_stats_from_grid(
    grid,
    map_array,
    map_shape,
    map_transform,
    map_crs,
    feature_name,
    grid_id_col="grid_id",
    fill_grid_id=np.nan
):
    """
    Overlay a continuous raster with a grid and compute per-grid statistics.

    Output column names will be:
        median_<feature_name>
        mean_<feature_name>
        min_<feature_name>
        max_<feature_name>
        std_dev_<feature_name>
        n_<feature_name>

    Parameters
    ----------
    grid : geopandas.GeoDataFrame
        Grid polygons with an ID column.
    map_array : numpy.ndarray
        2D raster array (float) with NaN for nodata/outside area.
    map_shape : tuple
        Raster shape (rows, cols).
    map_transform : affine.Affine
        Raster affine transform.
    map_crs : rasterio.crs.CRS
        Raster CRS.
    feature_name : str
        Name used to tag output columns (e.g., "slope").
    grid_id_col : str
        Grid ID column name.
    fill_grid_id : number
        Value assigned to pixels not covered by any grid cell.

    Returns
    -------
    stats_df : pandas.DataFrame
        Per-grid statistics with feature-specific column names.
    """
    feature_name = str(feature_name).strip()
    if not feature_name:
        raise ValueError("feature_name must be a non-empty string.")
    # --- Ensure CRS match ---
    if grid.crs != map_crs:
        grid = grid.to_crs(map_crs)

    if grid_id_col not in grid.columns:
        raise KeyError(f"'{grid_id_col}' not found in grid columns: {list(grid.columns)}")

    # --- Rasterize grid IDs to the raster pixel grid ---
    shapes = ((geom, int(gid)) for geom, gid in zip(grid.geometry, grid[grid_id_col]))

    grid_id_raster = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=map_shape,
        transform=map_transform,
        fill=fill_grid_id,
        dtype=np.float32
    )

    # --- Valid pixels: inside grid AND finite raster value ---
    valid = (grid_id_raster != fill_grid_id) & np.isfinite(map_array)

    gid = grid_id_raster[valid].astype(np.int32)
    val = map_array[valid].astype(np.float32)

    df = pd.DataFrame({grid_id_col: gid, "value": val})

     # --- Aggregate statistics ---
    stats_df = (
        df.groupby(grid_id_col)["value"]
          .agg(
              **{
                  f"median_{feature_name}": "median",
                  f"mean_{feature_name}": "mean",
                  f"min_{feature_name}": "min",
                  f"max_{feature_name}": "max",
                  f"std_dev_{feature_name}": "std",
                  f"n_{feature_name}": "count",
              }
          )
          .reset_index()
    )

    return stats_df


#
def extract_stats_from_hru(
    cat_hru,
    map_array,
    map_shape,
    map_transform,
    map_crs,
    feature_name,
    hru_id_col="HRU_ID",
    fill_hru_id=np.nan
):
    """
    Overlay a continuous raster with HRU polygons and compute per-HRU statistics.

    Output column names will be:
        median_<feature_name>
        mean_<feature_name>
        min_<feature_name>
        max_<feature_name>
        std_dev_<feature_name>
        n_<feature_name>

    Parameters
    ----------
    cat_hru : geopandas.GeoDataFrame
        HRU polygons with an HRU ID column.
    map_array : numpy.ndarray
        2D raster array (float) with NaN for nodata/outside area.
    map_shape : tuple
        Raster shape (rows, cols).
    map_transform : affine.Affine
        Raster affine transform.
    map_crs : rasterio.crs.CRS
        Raster CRS.
    feature_name : str
        Name used to tag output columns, e.g. "sand".
    hru_id_col : str, default "HRU_ID"
        HRU ID column name in cat_hru.
    fill_hru_id : number, default np.nan
        Value assigned to pixels not covered by any HRU polygon.

    Returns
    -------
    stats_df : pandas.DataFrame
        Per-HRU statistics with feature-specific column names.
    """


    feature_name = str(feature_name).strip()
    if not feature_name:
        raise ValueError("feature_name must be a non-empty string.")

    if hru_id_col not in cat_hru.columns:
        raise KeyError(f"'{hru_id_col}' not found in cat_hru columns: {list(cat_hru.columns)}")

    # Ensure CRS match
    if cat_hru.crs != map_crs:
        cat_hru = cat_hru.to_crs(map_crs)

    # Rasterize HRU IDs to the raster pixel grid
    shapes = (
        (geom, int(hru_id))
        for geom, hru_id in zip(cat_hru.geometry, cat_hru[hru_id_col])
        if geom is not None
    )

    hru_id_raster = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=map_shape,
        transform=map_transform,
        fill=fill_hru_id,
        dtype=np.float32
    )

    # Valid pixels: inside HRU AND finite raster value
    valid = (hru_id_raster != fill_hru_id) & np.isfinite(map_array)

    hru_ids = hru_id_raster[valid].astype(np.int32)
    values = map_array[valid].astype(np.float32)

    df = pd.DataFrame({
        hru_id_col: hru_ids,
        "value": values
    })

    # Aggregate stats by HRU
    stats_df = (
        df.groupby(hru_id_col)["value"]
        .agg(
            **{
                f"median_{feature_name}": "median",
                f"mean_{feature_name}": "mean",
                f"min_{feature_name}": "min",
                f"max_{feature_name}": "max",
                f"std_dev_{feature_name}": "std",
                f"n_{feature_name}": "count",
            }
        )
        .reset_index()
    )

    return stats_df


# [this function is also copied in optimisation_updated.py]
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


#
def fill_missing_hru(SSC_hru_frac, river_gdf, id_col="LINKNO"):
    """
    Ensure all IDs in river_gdf[id_col] exist in SSC_hru_frac dataframes.
    Missing IDs are added with zero values.

    Parameters
    ----------
    SSC_hru_frac : dict[int, pd.DataFrame]
        Dict of dataframes indexed by HRU_ID
    river_gdf : pd.DataFrame
        DataFrame containing ID column (e.g., LINKNO)
    id_col : str
        Column name in river_gdf to match (default: 'LINKNO')

    Returns
    -------
    SSC_hru_frac : dict[int, pd.DataFrame]
        Updated dictionary
    """

    # Extract target IDs
    target_ids = river_gdf[id_col].astype(int).values

    for k, df in SSC_hru_frac.items():
        df = df.copy()

        # Ensure index is integer
        df.index = df.index.astype(int)

        # Find missing IDs
        missing_ids = np.setdiff1d(target_ids, df.index.values)

        if len(missing_ids) > 0:
            # Create zero rows
            zero_df = pd.DataFrame(
                0,
                index=missing_ids,
                columns=df.columns
            )

            # Append
            df = pd.concat([df, zero_df])

        # Reorder to match river_gdf order
        df = df.loc[target_ids]

        # Save back
        SSC_hru_frac[k] = df

    return SSC_hru_frac


# -------------------------
# Snow attenuation factor
# -------------------------
def snow_attenuation_factor(swe_array, ksnow):
    """
    Calculate snow attenuation factor for rainfall-driven erosion.

    Parameters
    ----------
    swe_array : np.ndarray
        Snow water equivalent in mm, or kg m-2.
        For water equivalent, 1 kg m-2 = 1 mm.
    ksnow : float
        Snow attenuation coefficient [mm-1].
        Larger ksnow means stronger erosion reduction under snow.

    Returns
    -------
    np.ndarray
        Snow attenuation factor between 0 and 1.
        1 = no snow attenuation
        0 = near-complete attenuation
    """

    swe_array = np.nan_to_num(swe_array, nan=0.0)
    swe_array = np.clip(swe_array, 0.0, None)

    return np.exp(-ksnow * swe_array)


def _to_py_datetime_array(time_var, tnum_slice):
    """Convert numeric CF-time slice -> pandas datetime (robust to cftime)."""
    units = getattr(time_var, "units")
    cal = getattr(time_var, "calendar", "standard")
    t_cf = num2date(tnum_slice, units=units, calendar=cal)

    # Convert cftime objects to python datetimes (drop sub-second noise)
    t_py = [datetime(t.year, t.month, t.day, t.hour, t.minute, int(t.second)) for t in t_cf]
    return pd.to_datetime(t_py)

def _get_coord_values(nc, name):
    """
    Try to get coordinate-like values for 'name' from:
    - variable with same name
    - dimension (if exists) with a matching variable
    Returns None if not found.
    """
    if name in nc.variables:
        return np.asarray(nc.variables[name][:])
    # If it's only a dimension, sometimes there is no coordinate variable; return index values.
    if name in nc.dimensions and name not in nc.variables:
        n = len(nc.dimensions[name])
        return np.arange(n)
    return None

def _maybe_scalar(arr):
    """Return python scalar if arr is size 1, else return array."""
    if arr is None:
        return None
    a = np.asarray(arr).squeeze()
    if a.size == 1:
        return int(a) if np.issubdtype(a.dtype, np.integer) else float(a)
    return a

#
def _build_long_from_var(var_data, dims, time_name="time"):
    """
    Given sliced var_data and its dims, return:
      - base DataFrame with index columns for non-time dims (if any)
      - value column as 1D aligned with base rows
    Assumes var_data already sliced to the desired time window.
    """
    arr = np.asarray(var_data)
    # Ensure time axis is first for easier raveling with repeat/tile logic
    if time_name in dims:
        t_axis = dims.index(time_name)
        if t_axis != 0:
            arr = np.moveaxis(arr, t_axis, 0)
            dims = (time_name,) + tuple(d for i, d in enumerate(dims) if i != t_axis)

    # Now arr shape is (T, d1, d2, ...)
    shape = arr.shape
    T = shape[0]
    other_dims = list(dims[1:])
    other_shape = shape[1:]

    if not other_dims:
        # Only time
        base = pd.DataFrame({"_tpos": np.arange(T)})
        vals = arr.reshape(-1)
        return base, vals, other_dims

    # Create cartesian index for other dims
    # total rows = T * prod(other_shape)
    grids = np.meshgrid(*[np.arange(n) for n in other_shape], indexing="ij")
    # Each grid is shape other_shape; ravel to 1D
    other_index = {d: g.ravel() for d, g in zip(other_dims, grids)}

    # Repeat for each time step
    rep = int(np.prod(other_shape))
    base = pd.DataFrame({
        "_tpos": np.repeat(np.arange(T), rep),
        **{d: np.tile(other_index[d], T) for d in other_dims}
    })

    vals = arr.reshape(T * rep)
    return base, vals, other_dims

#
def extract_mizu_window(
    nc_path,
    start_time,
    end_time,
    var_name="IRFroutedRunoff",
    seg_dim="seg",
    time_var="time",
    id_vars=("reachID", "basinID"),
    output="long",   # "long" or "wide"
):
    """
    Extract mizuRoute variables for a time window.

    Parameters
    ----------
    nc_path : str
        Path to mizu NetCDF file.
    start_time, end_time : datetime
        Time window (inclusive).
    var_name : str
        Routed runoff variable name (default 'IRFroutedRunoff').
    seg_dim : str
        Segment dimension name (default 'seg').
    time_var : str
        Time coordinate variable name (default 'time').
    id_vars : tuple[str]
        Variables that identify segments (default ('reachID','basinID')).
    output : str
        'long' -> columns: time, seg_index, reachID, basinID, value
        'wide' -> index: time, columns: reachID (or seg_index if reachID missing)

    Returns
    -------
    df : pandas.DataFrame
        Long or wide format dataframe for the requested window.
    """

    with Dataset(nc_path) as nc:
        # --- time info ---
        tvar = nc.variables[time_var]
        t_units = getattr(tvar, "units")
        t_cal = getattr(tvar, "calendar", "standard")

        # Convert requested window to numeric time
        t0 = date2num(start_time, units=t_units, calendar=t_cal)
        t1 = date2num(end_time,   units=t_units, calendar=t_cal)

        # Read numeric times and find slice indices
        tnum = tvar[:]  # numpy array
        idx = np.where((tnum >= t0) & (tnum <= t1))[0]
        if idx.size == 0:
            raise ValueError("No timesteps found in the requested time window.")

        i0, i1 = idx[0], idx[-1] + 1  # slice end exclusive

        # Convert ONLY selected slice to python datetime
        time_cf = num2date(tnum[i0:i1], units=t_units, calendar=t_cal)
        time_py = [
            datetime(t.year, t.month, t.day, t.hour, t.minute, int(t.second))
            for t in time_cf
        ]
        time_dt = pd.to_datetime(time_py)

        # --- segment IDs (per seg) ---
        n_seg = nc.dimensions[seg_dim].size if seg_dim in nc.dimensions else None

        ids = {}
        for vname in id_vars:
            if vname in nc.variables:
                ids[vname] = np.asarray(nc.variables[vname][:]).squeeze()
            else:
                ids[vname] = None

        reachID = ids.get("reachID", None)
        basinID = ids.get("basinID", None)

        # --- main variable ---
        if var_name not in nc.variables:
            raise KeyError(f"{var_name} not found. Available vars: {list(nc.variables.keys())}")

        v = nc.variables[var_name]

        if time_var not in v.dimensions:
            raise ValueError(f"'{var_name}' does not have '{time_var}' dimension. dims={v.dimensions}")

        # Determine axes
        time_axis = v.dimensions.index(time_var)
        slicer = [slice(None)] * v.ndim
        slicer[time_axis] = slice(i0, i1)
        data = np.asarray(v[tuple(slicer)])

        # Ensure (time, seg) ordering for output
        # If seg exists and not in axis 1, move it.
        if seg_dim in v.dimensions:
            seg_axis = v.dimensions.index(seg_dim)
            data_ts = np.moveaxis(data, [time_axis, seg_axis], [0, 1])  # -> (time, seg, ...)
        else:
            # No seg dimension: just treat as (time,)
            data_ts = np.moveaxis(data, time_axis, 0)

        # Squeeze trailing singleton dims (e.g., ens=1)
        data_ts = np.squeeze(data_ts)

        # Now either:
        # - data_ts is (time, seg)
        # - or (time,) if no seg dim
        if output == "wide":
            if data_ts.ndim == 1:
                # time-only
                df = pd.DataFrame({var_name: data_ts}, index=time_dt)
                df.index.name = "time"
                return df

            # columns as reachID if available else seg index
            if reachID is not None and len(reachID) == data_ts.shape[1]:
                cols = [str(int(r)) for r in reachID]
            else:
                cols = [f"seg_{i}" for i in range(data_ts.shape[1])]

            df = pd.DataFrame(data_ts, index=time_dt, columns=cols)
            df.index.name = "time"
            return df

        # ---- long format ----
        if data_ts.ndim == 1:
            # time-only
            df = pd.DataFrame({"time": time_dt, var_name: data_ts})
            return df

        # time×seg -> long table
        n_time, nseg2 = data_ts.shape
        seg_index = np.arange(nseg2, dtype=int)

        df_long = pd.DataFrame({
            "time": np.repeat(time_dt.values, nseg2),
            "seg_index": np.tile(seg_index, n_time),
            var_name: data_ts.reshape(-1)
        })

        # Attach IDs if available
        if reachID is not None and len(reachID) == nseg2:
            df_long["reachID"] = np.tile(reachID.astype(np.int64), n_time)
        if basinID is not None and len(basinID) == nseg2:
            df_long["basinID"] = np.tile(basinID.astype(np.int64), n_time)

        return df_long



# unused

def make_catchment_timeseries(
    model_sed,
    cat_gdf,
    time_prefix="t_",
    predicate="intersects"   # or "within"
):
    """
    Aggregate grid-based timestep values inside a catchment polygon.

    Parameters
    ----------
    model_sed : GeoDataFrame
        Grid cells with timestep columns (e.g., 't_YYYYMMDD_HHMMSS').
    cat_gdf : GeoDataFrame
        Catchment boundary polygon(s).
    time_prefix : str
        Prefix used for timestep columns (default "t_").
    predicate : str
        Spatial relation: "intersects" (default) or "within".

    Returns
    -------
    timeseries_df : pandas.DataFrame
        Columns: ['time', 'SSC']
        where SSC is the sum across all grids for each timestep.
    """

    # --- Ensure GeoDataFrame ---
    model_sed_gdf = gpd.GeoDataFrame(
        model_sed,
        geometry="geometry",
        crs=model_sed.crs
    )

    # --- Match CRS ---
    if model_sed_gdf.crs != cat_gdf.crs:
        model_sed_gdf = model_sed_gdf.to_crs(cat_gdf.crs)

    # --- Merge catchment geometry ---
    catchment_geom = cat_gdf.geometry.unary_union

    # --- Select grid cells inside/intersecting catchment ---
    if predicate == "within":
        mask = model_sed_gdf.geometry.within(catchment_geom)
    else:
        mask = model_sed_gdf.geometry.intersects(catchment_geom)

    model_in = model_sed_gdf.loc[mask].copy()

    # --- Identify timestep columns ---
    time_cols = [
        c for c in model_in.columns
        if isinstance(c, str) and c.startswith(time_prefix)
    ]

    if len(time_cols) == 0:
        raise ValueError(f"No timestep columns found starting with '{time_prefix}'")

    # --- Sum across grid cells ---
    sum_by_time = model_in[time_cols].sum(axis=0)

    # --- Convert column names to datetime ---
    sum_by_time.index = pd.to_datetime(
        sum_by_time.index.str.replace(time_prefix, ""),
        format="%Y%m%d_%H%M%S",
        errors="coerce"
    )

    # --- Build output DataFrame ---
    timeseries_df = sum_by_time.reset_index()
    timeseries_df.columns = ["time", "SSC"]

    return timeseries_df



# unused
def make_hru_timeseries_from_grid(
    model_sed,
    cat_hru,
    gru_id_col="GRU_ID",
    time_prefix="t_",
    join_predicate="intersects"   # "within" is stricter
):
    """
    Sum grid time-step columns within each HRU polygon.

    Returns
    -------
    hru_ts : pandas.DataFrame
        Index = datetime (timesteps), columns = GRU_ID, values = sum over grids in each HRU.
    """
    # --- Ensure GeoDataFrames and matching CRS ---
    model_gdf = gpd.GeoDataFrame(model_sed, geometry="geometry", crs=model_sed.crs)

    if model_gdf.crs != cat_hru.crs:
        model_gdf = model_gdf.to_crs(cat_hru.crs)

    if gru_id_col not in cat_hru.columns:
        raise KeyError(f"'{gru_id_col}' not in cat_hru columns: {list(cat_hru.columns)}")

    # --- Identify time columns in model_sed ---
    time_cols = [c for c in model_gdf.columns if isinstance(c, str) and c.startswith(time_prefix)]
    if len(time_cols) == 0:
        raise ValueError(f"No timestep columns found starting with '{time_prefix}'")

    # --- Spatial join: assign each grid cell to an HRU (GRU_ID) ---
    # Keep only needed columns to speed things up
    left = model_gdf[["grid_id", "geometry"] + time_cols].copy()
    right = cat_hru[[gru_id_col, "geometry"]].copy()

    joined = gpd.sjoin(left, right, how="inner", predicate=join_predicate)

    # Now joined has GRU_ID for each grid cell (possibly multiple if overlaps)
    # --- Aggregate: sum time columns per GRU_ID ---
    summed = joined.groupby(gru_id_col)[time_cols].sum()

    # --- Convert to "time-indexed" DataFrame with GRU_ID as columns ---
    hru_ts = summed.T
    hru_ts.index = pd.to_datetime(hru_ts.index.str.replace(time_prefix, ""),
                                  format="%Y%m%d_%H%M%S",
                                  errors="coerce")

    # Sort by time
    hru_ts = hru_ts.sort_index()

    return hru_ts


#
def forcingnc_to_dataframe(
    directory,
    var_name="pptrate",
    time_name="time",
    hru_name="hruId",
    start_date="2024-01-15",
    end_date="2025-01-15",
    save_csv=False,
    output_csv=None
):
    """
    Read forcing NetCDF files from a directory and convert a selected variable
    to a long-format pandas DataFrame filtered by date.

    Parameters
    ----------
    directory : str or path-like
        Directory containing NetCDF forcing files.
    var_name : str, optional
        Name of the forcing variable to extract (e.g., "pptrate",
        "airtemp", "spechum"). Default is "pptrate".
    time_name : str, optional
        Name of the time variable in the NetCDF files.
        Default is "time".
    hru_name : str, optional
        Name of the HRU identifier variable in the NetCDF files.
        Default is "hruId".
    start_date : str or datetime-like, optional
        Start date for extraction (inclusive).
    end_date : str or datetime-like, optional
        End date for extraction (inclusive).
    save_csv : bool, optional
        If True, save the extracted DataFrame to CSV.
        Default is False.
    output_csv : str, optional
        Output CSV file path. If None and save_csv=True,
        a filename is generated automatically.

    Returns
    -------
    df_all : pandas.DataFrame
        Long-format DataFrame containing:
        - time : timestamp
        - hruId (or specified hru_name) : HRU identifier
        - selected variable values
        - source_file : source NetCDF filename

        Returns an empty DataFrame if no data are found within
        the specified period.
    """

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    nc_files = sorted(glob.glob(os.path.join(directory, "*.nc")))
    dfs = []

    for full_path in nc_files:
        file_name = os.path.basename(full_path)

        try:
            with Dataset(full_path, 'r') as ds:
                # Use parameterized variable names
                time_var = ds.variables[time_name]
                hru_ids = np.array(ds.variables[hru_name][:])
                data_var = np.array(ds.variables[var_name][:])

                # Convert time
                time_values = num2date(
                    time_var[:],
                    units=time_var.units,
                    calendar=getattr(time_var, 'calendar', 'standard')
                )
                time_values = pd.to_datetime([str(t) for t in time_values])

                # Filter by date
                time_mask = (time_values >= start_date) & (time_values <= end_date)
                if not np.any(time_mask):
                    continue

                filtered_time = time_values[time_mask]
                filtered_data = data_var[time_mask, :]

                # Handle fill values
                fill_value = getattr(ds.variables[var_name], '_FillValue', None)
                if fill_value is not None:
                    filtered_data = np.where(filtered_data == fill_value, np.nan, filtered_data)

                ntime = len(filtered_time)
                nhru = len(hru_ids)

                df_file = pd.DataFrame({
                    'time': np.repeat(filtered_time, nhru),
                    hru_name: np.tile(hru_ids, ntime),
                    var_name: filtered_data.reshape(-1),
                    'source_file': file_name
                })

                dfs.append(df_file)

        except Exception as e:
            print(f"Error reading {file_name}: {e}")

    if not dfs:
        print("No data found in the selected period.")
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = (
        df_all
        .sort_values(['time', hru_name])
        .drop_duplicates(['time', hru_name])
        .reset_index(drop=True)
    )

    if save_csv:
        if output_csv is None:
            output_csv = os.path.join(
                directory,
                f"forcing_{var_name}_{start_date.date()}_to_{end_date.date()}.csv"
            )
        df_all.to_csv(output_csv, index=False)
        print(f"CSV saved to: {output_csv}")

    return df_all


#
def forcingnc_to_dict_by_hru(
    directory,
    var_name="pptrate",
    time_name="time",
    hru_name="hruId",
    start_date="2024-01-15",
    end_date="2025-01-15"
):
    """
    Read forcing NetCDF files from a directory and return a dictionary of
    time series DataFrames indexed by HRU.

    Parameters
    ----------
    directory : str or path-like
        Directory containing NetCDF forcing files.
    var_name : str, optional
        Name of the forcing variable to extract (e.g., "pptrate",
        "airtemp", "spechum"). Default is "pptrate".
    time_name : str, optional
        Name of the time variable in the NetCDF files.
        Default is "time".
    hru_name : str, optional
        Name of the HRU identifier variable in the NetCDF files.
        Default is "hruId".
    start_date : str or datetime-like, optional
        Start date for extraction (inclusive).
    end_date : str or datetime-like, optional
        End date for extraction (inclusive).

    Returns
    -------
    hru_dict : dict
        Dictionary where keys are HRU identifiers and values are
        pandas DataFrames containing:
        
        - time : timestamp
        - selected variable values
        
        Each DataFrame is sorted by time with duplicate timestamps removed.

        Returns an empty dictionary if no data are found within the
        specified period.
    """

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    nc_files = sorted(glob.glob(os.path.join(directory, "*.nc")))
    hru_dict = {}

    for full_path in nc_files:
        file_name = os.path.basename(full_path)

        try:
            with Dataset(full_path, "r") as ds:
                time_var = ds.variables[time_name]
                hru_ids = np.array(ds.variables[hru_name][:])
                data_var = np.array(ds.variables[var_name][:])

                time_values = num2date(
                    time_var[:],
                    units=time_var.units,
                    calendar=getattr(time_var, "calendar", "standard")
                )
                time_values = pd.to_datetime([str(t) for t in time_values])

                time_mask = (time_values >= start_date) & (time_values <= end_date)
                if not np.any(time_mask):
                    continue

                filtered_time = time_values[time_mask]
                filtered_data = data_var[time_mask, :]

                fill_value = getattr(ds.variables[var_name], "_FillValue", None)
                if fill_value is not None:
                    filtered_data = np.where(filtered_data == fill_value, np.nan, filtered_data)

                for i, hru in enumerate(hru_ids):
                    # convert NumPy scalar to normal Python type
                    hru_key = hru.item() if hasattr(hru, "item") else hru

                    df_hru = pd.DataFrame({
                        "time": filtered_time,
                        var_name: filtered_data[:, i]
                    })

                    if hru_key in hru_dict:
                        hru_dict[hru_key] = pd.concat(
                            [hru_dict[hru_key], df_hru],
                            ignore_index=True
                        )
                    else:
                        hru_dict[hru_key] = df_hru

        except Exception as e:
            print(f"Error reading {file_name}: {e}")

    for hru_key in hru_dict:
        hru_dict[hru_key] = (
            hru_dict[hru_key]
            .sort_values("time")
            .drop_duplicates(subset="time")
            .reset_index(drop=True)
        )

    if not hru_dict:
        print("No data found in the selected period.")

    return hru_dict



#
def extract_runoff_nc(
    nc_file,
    var_name="averageRoutedRunoff",
    time_name="time",
    hru_name="hruId",
    start_date=None,
    end_date=None,
    save_csv=False,
    output_csv=None
):      
    """
    Extract runoff data from a SUMMA NetCDF output file and convert it to a
    long-format pandas DataFrame filtered by date.

    Parameters
    ----------
    nc_file : str or path-like
        Path to the SUMMA NetCDF file.
    var_name : str, optional
        Name of the runoff variable to extract.
        Default is "averageRoutedRunoff".
    time_name : str, optional
        Name of the time variable in the NetCDF file.
        Default is "time".
    hru_name : str, optional
        Name of the HRU identifier variable in the NetCDF file.
        Default is "hruId".
    start_date : str or datetime-like, optional
        Start date for extraction, inclusive.
    end_date : str or datetime-like, optional
        End date for extraction, inclusive.
    save_csv : bool, optional
        If True, save the extracted DataFrame to CSV.
        Default is False.
    output_csv : str, optional
        Output CSV file path. If None and save_csv=True,
        a filename is generated automatically.

    Returns
    -------
    df : pandas.DataFrame
        Long-format DataFrame containing:

        - time : timestamp
        - hruId or specified hru_name : HRU identifier
        - selected runoff variable values

        Returns an empty DataFrame if no data are found in the selected period.

    Raises
    ------
    RuntimeError
        If the NetCDF file cannot be read or the requested variables are missing.
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    try:
        with Dataset(nc_file, "r") as ds:
            time_var = ds.variables[time_name]
            hru_ids = np.array(ds.variables[hru_name][:])
            data_var = np.array(ds.variables[var_name][:])

            time_values = num2date(
                time_var[:],
                units=time_var.units,
                calendar=getattr(time_var, "calendar", "standard")
            )
            time_values = pd.to_datetime([t.isoformat() for t in time_values], format="mixed")

            time_mask = (time_values >= start_date) & (time_values <= end_date)
            if not np.any(time_mask):
                print("No data found in the selected period.")
                return pd.DataFrame()

            filtered_time = time_values[time_mask]
            filtered_data = data_var[time_mask, :]

            fill_value = getattr(ds.variables[var_name], "_FillValue", None)
            if fill_value is not None:
                filtered_data = np.where(filtered_data == fill_value, np.nan, filtered_data)

            ntime = len(filtered_time)
            nhru = len(hru_ids)

            df = pd.DataFrame({
                "time": np.repeat(filtered_time, nhru),
                hru_name: np.tile(hru_ids, ntime),
                var_name: filtered_data.reshape(-1),
            })

            df = (
                df.sort_values(["time", hru_name])
                  .drop_duplicates(["time", hru_name])
                  .reset_index(drop=True)
            )

            if save_csv:
                if output_csv is None:
                    output_csv = os.path.join(
                        os.path.dirname(nc_file),
                        f"forcing_{var_name}_{start_date.date()}_to_{end_date.date()}.csv"
                    )
                df.to_csv(output_csv, index=False)
                print(f"CSV saved to: {output_csv}")

            return df

    except Exception as e:
        raise RuntimeError(
            f"Error reading {os.path.basename(nc_file)}: {e}"
        )
    
#
def extract_runoff_nc_to_dict_by_hru(
    nc_file,
    var_name="averageRoutedRunoff",
    time_name="time",
    hru_name="hruId",
    start_date=None,
    end_date=None,
    save_csv=False,
    output_csv=None
):
    """
    Extract runoff data from a SUMMA NetCDF output file and return a
    dictionary of time series DataFrames indexed by HRU.

    Parameters
    ----------
    nc_file : str or path-like
        Path to the SUMMA NetCDF file.
    var_name : str, optional
        Name of the runoff variable to extract.
        Default is "averageRoutedRunoff".
    time_name : str, optional
        Name of the time variable in the NetCDF file.
        Default is "time".
    hru_name : str, optional
        Name of the HRU identifier variable in the NetCDF file.
        Default is "hruId".
    start_date : str or datetime-like
        Start date for extraction, inclusive. Must be provided.
    end_date : str or datetime-like
        End date for extraction, inclusive. Must be provided.
    save_csv : bool, optional
        If True, save each HRU time series as a separate CSV file.
        Default is False.
    output_csv : str or path-like, optional
        Output directory for saved CSV files. If None and save_csv=True,
        CSV files are saved in the same directory as nc_file.

    Returns
    -------
    hru_dict : dict
        Dictionary where keys are HRU identifiers and values are
        pandas DataFrames containing:

        - time : timestamp
        - selected runoff variable values

        Each DataFrame is sorted by time with duplicate timestamps removed.
        Returns an empty dictionary if no data are found in the selected period
        or if the NetCDF file cannot be read.

    Raises
    ------
    ValueError
        If start_date or end_date is not provided.
    """
    
    if start_date is None or end_date is None:
        raise ValueError("start_date and end_date must be provided.")

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    hru_dict = {}

    try:
        with Dataset(nc_file, "r") as ds:
            # Read variables
            time_var = ds.variables[time_name]
            hru_ids = np.array(ds.variables[hru_name][:])
            data_var = np.array(ds.variables[var_name][:])

            # Convert time
            time_values = num2date(
                time_var[:],
                units=time_var.units,
                calendar=getattr(time_var, "calendar", "standard")
            )
            time_values = pd.to_datetime([t.isoformat() for t in time_values], format="mixed")

            # Filter by date range
            time_mask = (time_values >= start_date) & (time_values <= end_date)
            if not np.any(time_mask):
                print("No data found in the selected period.")
                return {}

            filtered_time = time_values[time_mask]
            filtered_data = data_var[time_mask, :]

            # Handle fill values
            fill_value = getattr(ds.variables[var_name], "_FillValue", None)
            if fill_value is not None:
                filtered_data = np.where(filtered_data == fill_value, np.nan, filtered_data)

            # Build dict keyed by HRU ID
            for i, hru in enumerate(hru_ids):
                hru_key = hru.item() if hasattr(hru, "item") else hru

                df_hru = pd.DataFrame({
                    "time": filtered_time,
                    var_name: filtered_data[:, i]
                })

                hru_dict[hru_key] = (
                    df_hru.sort_values("time")
                          .drop_duplicates(subset="time")
                          .reset_index(drop=True)
                )

            # Optional CSV export
            if save_csv:
                output_dir = output_csv if output_csv is not None else os.path.dirname(nc_file)

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                for hru_key, df_hru in hru_dict.items():
                    csv_path = os.path.join(
                        output_dir,
                        f"{var_name}_hru_{hru_key}_{start_date.date()}_to_{end_date.date()}.csv"
                    )
                    df_hru.to_csv(csv_path, index=False)

                print(f"CSV files saved in: {output_dir}")

            return hru_dict

    except Exception as e:
        print(f"Error reading {os.path.basename(nc_file)}: {e}")
        return {}
    

# 
def extract_flow_variable_nc(
    nc_file,
    var_name,
    time_name="time",
    seg_name="segId",
    start_date=None,
    end_date=None,
    save_csv=False,
    output_csv=None
):
    """
    Extract a river flow-related variable from a mizuRoute NetCDF file and
    convert it to a long-format pandas DataFrame filtered by date.

    Parameters
    ----------
    nc_file : str or path-like
        Path to the mizuRoute NetCDF file.
    var_name : str
        Name of the variable to extract, such as discharge, flow velocity,
        flow depth, or channel width.
    time_name : str, optional
        Name of the time variable in the NetCDF file.
        Default is "time".
    seg_name : str, optional
        Name of the river segment identifier variable in the NetCDF file.
        Default is "segId".
    start_date : str or datetime-like
        Start date for extraction, inclusive. Must be provided.
    end_date : str or datetime-like
        End date for extraction, inclusive. Must be provided.
    save_csv : bool, optional
        If True, save the extracted DataFrame to CSV.
        Default is False.
    output_csv : str or path-like, optional
        Output CSV file path. If None and save_csv=True,
        a filename is generated automatically in the same directory as nc_file.

    Returns
    -------
    df : pandas.DataFrame
        Long-format DataFrame containing:

        - time : timestamp
        - segId or specified seg_name : river segment identifier
        - selected variable values

        Returns an empty DataFrame if no data are found in the selected period
        or if the NetCDF file cannot be read.

    Raises
    ------
    ValueError
        If start_date or end_date is not provided, or if the selected variable
        is not 2D with shape (time, segment).
    KeyError
        If var_name, time_name, or seg_name is not found in the NetCDF file.
    """

    if start_date is None or end_date is None:
        raise ValueError("start_date and end_date must be provided.")

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    try:
        with Dataset(nc_file, "r") as ds:
            if var_name not in ds.variables:
                raise KeyError(f"Variable '{var_name}' not found in file.")

            if time_name not in ds.variables:
                raise KeyError(f"Time variable '{time_name}' not found in file.")

            if seg_name not in ds.variables:
                raise KeyError(f"Segment ID variable '{seg_name}' not found in file.")

            # Read variables
            time_var = ds.variables[time_name]
            seg_ids = np.array(ds.variables[seg_name][:])
            data_var = np.array(ds.variables[var_name][:])

            # Convert time
            # time_values = num2date(
            #     time_var[:],
            #     units=time_var.units,
            #     calendar=getattr(time_var, "calendar", "standard")
            # )
            # time_values = pd.to_datetime([str(t) for t in time_values])
            # #------
            raw_time = np.asarray(time_var[:])
            time_units = str(getattr(time_var, "units", "")).strip()
            
            if time_units.lower().startswith("nanoseconds since"):
                origin = time_units.split("since", 1)[1].strip()
            
                time_values = (
                    pd.to_datetime(origin)
                    + pd.to_timedelta(raw_time, unit="ns")
                )
            
                # Optional: remove the tiny nanosecond offset
                time_values = pd.DatetimeIndex(time_values).round("s")
            
            else:
                decoded_time = num2date(
                    raw_time,
                    units=time_units,
                    calendar=getattr(time_var, "calendar", "standard")
                )
            
                time_values = pd.to_datetime(
                    [str(t) for t in decoded_time]
                )
            # #------
            # Filter by date range
            time_mask = (time_values >= start_date) & (time_values <= end_date)
            if not np.any(time_mask):
                print("No data found in the selected period.")
                return pd.DataFrame()

            filtered_time = time_values[time_mask]

            # Handle 2D variables: (time, seg)
            if data_var.ndim != 2:
                raise ValueError(
                    f"Variable '{var_name}' must be 2D with shape (time, seg). "
                    f"Got shape {data_var.shape}."
                )

            filtered_data = data_var[time_mask, :]

            # Handle fill values
            fill_value = getattr(ds.variables[var_name], "_FillValue", None)
            if fill_value is not None:
                filtered_data = np.where(filtered_data == fill_value, np.nan, filtered_data)

            ntime = len(filtered_time)
            nseg = len(seg_ids)

            # Build dataframe
            df = pd.DataFrame({
                "time": np.repeat(filtered_time, nseg),
                seg_name: np.tile(seg_ids, ntime),
                var_name: filtered_data.reshape(-1),
            })

            df = (
                df.sort_values(["time", seg_name])
                  .drop_duplicates(["time", seg_name])
                  .reset_index(drop=True)
            )

            # Optional CSV save
            if save_csv:
                if output_csv is None:
                    output_csv = os.path.join(
                        os.path.dirname(nc_file),
                        f"{var_name}_{start_date.date()}_to_{end_date.date()}.csv"
                    )
                df.to_csv(output_csv, index=False)
                print(f"CSV saved to: {output_csv}")

            return df

    except Exception as e:
        print(f"Error reading {os.path.basename(nc_file)}: {e}")
        return pd.DataFrame()
    

#

def extract_flow_variable_nc_to_dict_by_segid(
    nc_file,
    var_name,
    time_name="time",
    seg_name="segId",
    start_date=None,
    end_date=None,
    save_csv=False,
    output_dir=None
):

    """
    Extract a river flow-related variable from a mizuRoute NetCDF file and
    return a dictionary of time series DataFrames indexed by river segment ID.

    Parameters
    ----------
    nc_file : str or path-like
        Path to the mizuRoute NetCDF file.
    var_name : str
        Name of the variable to extract, such as discharge, flow velocity,
        flow depth, or channel width.
    time_name : str, optional
        Name of the time variable in the NetCDF file.
        Default is "time".
    seg_name : str, optional
        Name of the river segment identifier variable in the NetCDF file.
        Default is "segId".
    start_date : str or datetime-like
        Start date for extraction, inclusive. Must be provided.
    end_date : str or datetime-like
        End date for extraction, inclusive. Must be provided.
    save_csv : bool, optional
        If True, save each segment time series as a separate CSV file.
        Default is False.
    output_dir : str or path-like, optional
        Output directory for saved CSV files. If None and save_csv=True,
        a new directory named "{var_name}_by_seg" is created beside nc_file.

    Returns
    -------
    seg_dict : dict
        Dictionary where keys are river segment IDs and values are
        pandas DataFrames containing:

        - time : timestamp
        - selected flow variable values

        Each DataFrame is sorted by time with duplicate timestamps removed.
        Returns an empty dictionary if no data are found in the selected period
        or if the NetCDF file cannot be read.

    Raises
    ------
    ValueError
        If start_date or end_date is not provided, or if the selected variable
        is not 2D with shape (time, segment).
    KeyError
        If var_name, time_name, or seg_name is not found in the NetCDF file.
    """
    if start_date is None or end_date is None:
        raise ValueError("start_date and end_date must be provided.")

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    seg_dict = {}

    try:
        with Dataset(nc_file, "r") as ds:
            if var_name not in ds.variables:
                raise KeyError(f"Variable '{var_name}' not found in file.")
            if time_name not in ds.variables:
                raise KeyError(f"Time variable '{time_name}' not found in file.")
            if seg_name not in ds.variables:
                raise KeyError(f"Segment variable '{seg_name}' not found in file.")

            time_var = ds.variables[time_name]
            seg_ids = np.array(ds.variables[seg_name][:])
            data_var = np.array(ds.variables[var_name][:])

            if data_var.ndim != 2:
                raise ValueError(
                    f"Variable '{var_name}' must be 2D with shape (time, seg). "
                    f"Got shape {data_var.shape}."
                )

            # Convert time
            # time_values = num2date(
            #     time_var[:],
            #     units=time_var.units,
            #     calendar=getattr(time_var, "calendar", "standard")
            # )
            # time_values = pd.to_datetime([str(t) for t in time_values])
            # #--------
            raw_time = np.asarray(time_var[:])
            time_units = str(getattr(time_var, "units", "")).strip()
            
            if time_units.lower().startswith("nanoseconds since"):
                origin = time_units.split("since", 1)[1].strip()
            
                time_values = (
                    pd.to_datetime(origin)
                    + pd.to_timedelta(raw_time, unit="ns")
                )
            
                # Optional: remove the tiny nanosecond offset
                time_values = pd.DatetimeIndex(time_values).round("s")
            
            else:
                decoded_time = num2date(
                    raw_time,
                    units=time_units,
                    calendar=getattr(time_var, "calendar", "standard")
                )
            
                time_values = pd.to_datetime(
                    [str(t) for t in decoded_time]
                )
            # #-------- 
            # Filter time range
            time_mask = (time_values >= start_date) & (time_values <= end_date)
            if not np.any(time_mask):
                print("No data found in the selected period.")
                return {}

            filtered_time = time_values[time_mask]
            filtered_data = data_var[time_mask, :]

            # Handle fill values
            fill_value = getattr(ds.variables[var_name], "_FillValue", None)
            if fill_value is not None:
                filtered_data = np.where(filtered_data == fill_value, np.nan, filtered_data)

            # Build dict by segId
            for i, seg in enumerate(seg_ids):
                seg_key = seg.item() if hasattr(seg, "item") else seg

                df_seg = pd.DataFrame({
                    "time": filtered_time,
                    var_name: filtered_data[:, i]
                })

                seg_dict[seg_key] = (
                    df_seg.sort_values("time")
                          .drop_duplicates(subset="time")
                          .reset_index(drop=True)
                )

            # Optional CSV save: one file per segId
            if save_csv:
                if output_dir is None:
                    output_dir = os.path.join(
                        os.path.dirname(nc_file),
                        f"{var_name}_by_seg"
                    )

                os.makedirs(output_dir, exist_ok=True)

                for seg_key, df_seg in seg_dict.items():
                    csv_file = os.path.join(
                        output_dir,
                        f"{var_name}_seg_{seg_key}_{start_date.date()}_to_{end_date.date()}.csv"
                    )
                    df_seg.to_csv(csv_file, index=False)

                print(f"CSV files saved in: {output_dir}")

            return seg_dict

    except Exception as e:
        print(f"Error reading {os.path.basename(nc_file)}: {e}")
        return {}

# function to compute catchment SSC timeseries from HRU

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
    runoff_long = df_runoff[[ "time", runoff_hru_col, runoff_col ]].rename(
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




# functions for TempSedRout
def long_to_wide_timeseries(
    df,
    id_col,
    value_col,
    time_col="time",
    time_round="h",
    prefix="t_",
    time_format="%Y%m%d_%H%M%S",
    rename_id_to=None
):
    """
    Convert long-format dataframe (time, id, value) to wide format.
    
    Compute catchment sediment concentration (SSC) as a runoff-area weighted average over HRUs.

    Formula
    -------
    The catchment SSC at each time step is computed as:
    
        SSC_catchment(t) = sum_j [ SSC_j(t) * q_j(t) * A_j ] / sum_j [ q_j(t) * A_j ]
    
    where:
        SSC_j(t) : sediment concentration at HRU j [mg/m^3]
        q_j(t)   : runoff depth rate at HRU j [mm/s or consistent unit]
        A_j      : area of HRU j [m^2 or consistent unit]
    
    Since discharge Q_j ∝ q_j * A_j, this is equivalent to:
    
        SSC_catchment(t) = sum_j [ SSC_j(t) * Q_j(t) ] / sum_j [ Q_j(t) ]
    
    Interpretation
    --------------
    - This represents a flow-weighted average SSC across HRUs.
    - Units of runoff and area cancel out as long as they are consistent across HRUs.
    - Physically, this equals total sediment flux divided by total water flux.

    Parameters
    ----------
    df : pandas.DataFrame
        Input long dataframe.
    id_col : str
        Column representing spatial unit (e.g., segId, HRU_ID).
    value_col : str
        Column with values (e.g., channel_depth, discharge).
    time_col : str, default "time"
        Time column name.
    time_round : str, default "H"
        Pandas rounding frequency (e.g., "H" for hourly).
    prefix : str, default "t_"
        Prefix for output time columns.
    time_format : str
        Format for time column names.
    rename_id_to : str or None
        Rename id column (e.g., "HRU_ID").

    Returns
    -------
    df_wide : pandas.DataFrame
        Wide dataframe with id column + time columns.
    """


    # 1) copy and standardize time
    df2 = df.copy()
    df2[time_col] = pd.to_datetime(df2[time_col]).dt.round(time_round)

    # 2) pivot
    df_wide = df2.pivot_table(
        index=id_col,
        columns=time_col,
        values=value_col,
        aggfunc="first"
    ).reset_index()

    # 3) rename columns
    df_wide.columns = [
        id_col if col == id_col else f"{prefix}{col.strftime(time_format)}"
        for col in df_wide.columns
    ]

    # 4) optional rename of id column
    if rename_id_to is not None:
        df_wide = df_wide.rename(columns={id_col: rename_id_to})

    # 5) sort
    df_wide = df_wide.sort_values(df_wide.columns[0]).reset_index(drop=True)

    return df_wide


#

def long_to_time_index_matrix(
    df,
    id_col="segId",
    value_col="channel_depth",
    time_col="time",
    time_round="h",
    sort_index=True
):
    """
    Convert long dataframe to matrix with:
        index = time
        columns = id (e.g., segId)
        values = variable (e.g., channel_depth)

    Parameters
    ----------
    df : pandas.DataFrame
        Input long dataframe.
    id_col : str
        Column representing spatial unit (e.g., segId).
    value_col : str
        Column with values.
    time_col : str, default "time"
        Time column.
    time_round : str, default "H"
        Time rounding frequency.
    sort_index : bool, default True
        Sort time index.

    Returns
    -------
    df_matrix : pandas.DataFrame
        DataFrame with time index and segId columns.
    """


    # 1) copy and standardize time
    df2 = df.copy()
    df2[time_col] = pd.to_datetime(df2[time_col]).dt.round(time_round)

    # 2) pivot
    df_matrix = df2.pivot_table(
        index=time_col,
        columns=id_col,
        values=value_col,
        aggfunc="first"
    )

    # 3) optional sorting
    if sort_index:
        df_matrix = df_matrix.sort_index()

    return df_matrix


#%% Sum SSC for each HRU from grids [this function is also copied in optimisation_updated.py]
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

#%% Sum SSC for whole catchment from grids

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



#%% routing gamma


def routing_gamma (qinst, a, mt, K, dt):
    """
    Route an input time series using a gamma-distribution unit hydrograph.

    Parameters
    ----------
    qinst : array-like
        Input flow, concentration, or sediment time series to be routed.
    a : float
        Shape parameter of the gamma distribution.
    mt : float
        Mean travel time or mean delay of the routing distribution.
    K : int
        Routing window length, in number of time steps.
    dt : float
        Model time step length, in days.

    Returns
    -------
    qrouted : numpy.ndarray
        Routed output time series with the same length as qinst.
    """
    a = float(a)
    mt = float(mt)
    dt = float(dt)
    K = int(round(K))
    # gamma scale parameter
    theta= mt/a
    
    n= len (qinst)
    
    # -----------------------------
    # gamma unit hydrograph
    # -----------------------------
    
    # ------ $$$
    # use math
    # ------
    def gamma_pdf(t, a, theta):
        if t < 0:
            return 0.0
        return (t ** (a - 1)) * math.exp(-t / theta) / (math.gamma(a) * (theta ** a))

    # w = np.zeros(K)
    # for k in range(K):
    #     t_mid = (k + 0.5) * dt
    #     w[k] = gamma_pdf(t_mid, a, theta) * dt
    # ------ $$$
    
    # ------ $$$
    # use scipy
    # ------
    t = np.arange(0, K+1) * dt
    F= gamma.cdf(t, a, scale=theta)
    # discrete weights
    w = F[1:] - F[:-1]
    # ------ $$$
    
    # normalize so weights sum to 1
    w = w / w.sum()
    
    # -----------------------------
    # route runoff by convolution
    # -----------------------------
    qrouted = np.zeros(n)
    
    
    for i in range(n):
        s = 0.0
        for k in range(K):
            if i - k >= 0:
                s += w[k] * qinst[i - k]
        qrouted[i] = s
        
    return qrouted

# [this function is also copied in optimisation_updated.py]
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

    
    