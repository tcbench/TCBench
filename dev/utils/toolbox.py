#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:45:42 2023

This file contains a "toolbox" library - functions and constants that will be
used in other scripts to perform tasks associated with TCBench and post-processing
models

@author: mgomezd1
"""

# %% Imports and base paths

# OS and IO
import os
import traceback
import subprocess
import gc

# Base Libraries
import pandas as pd
import numpy as np
import xarray as xr

# import xesmf as xe
import joblib as jl
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.mpl.patch import geos_to_path
import itertools
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from datetime import datetime
import dask.array as da
import pickle
import hashlib

# TCBench Libraries
try:
    from utils import constants, data_lib
except:
    # import constants, data_lib
    print("error")

# Retrieve Repository Path
repo_path = "/" + os.path.join(*os.getcwd().split("/")[:-1])

# print(repo_path)


# %% Auxilliary Functions


def plot_facecolors(**kwargs):
    figcolor = kwargs.get("figcolor", np.array([1, 21, 38]) / 255)
    axcolor = kwargs.get("axcolor", np.array([141, 166, 166]) / 255)
    textcolor = kwargs.get("textcolor", np.array([242, 240, 228]) / 255)

    if kwargs.get("fig", None) is not None:
        fig = kwargs.get("fig")
        # Change the facecolor of the figure
        fig.set_facecolor(figcolor)

    if kwargs.get("axes", None) is not None:
        if isinstance(kwargs.get("axes"), np.ndarray):
            axes = kwargs.get("axes")
        elif isinstance(kwargs.get("axes"), plt.Axes):
            axes = np.array([kwargs.get("axes")])
        else:
            raise ValueError(
                f"Invalid type for axes: {type(kwargs.get('axes'))}. Expected np.ndarray or plt.Axes"
            )

        # Change the facecolor of the axes
        for ax in axes.flatten():
            ax.set_facecolor(axcolor)

        # Change the color of the title
        for ax in axes.flatten():
            title = ax.get_title()
            ax.set_title(title, color=textcolor)

        # Change the color of the x and y axis labels
        for ax in axes.flatten():
            ax.xaxis.label.set_color(textcolor)
            ax.yaxis.label.set_color(textcolor)

        # Change the color of the tick labels
        for ax in axes.flatten():
            for label in ax.get_xticklabels():
                label.set_color(textcolor)
            for label in ax.get_yticklabels():
                label.set_color(textcolor)


def rolling_window(arr, window_size, step=1):
    # Calculate the number of windows
    num_windows = ((arr.size - window_size) // step) + 1
    # Calculate the shape of the new array
    new_shape = (num_windows, window_size)
    # Calculate the new strides
    new_strides = (arr.strides[0] * step, arr.strides[0])
    # Create the rolling window view
    return np.lib.stride_tricks.as_strided(arr, shape=new_shape, strides=new_strides)


def axis_generator(**kwargs):
    origin = kwargs.get("origin", (0, 0))
    resolution = kwargs.get("resolution", 0.25)
    lat_limits = kwargs.get("lat_limits", None)
    lon_limits = kwargs.get("lon_limits", None)
    lon_mode = kwargs.get("lon_mode", 360)
    use_poles = kwargs.get("use_poles", True)

    if lat_limits is not None:
        assert (
            origin[0] >= lat_limits[0]
        ), "Origin lower limit greater than origin value"
        assert origin[0] <= lat_limits[1], "Origin upper limit less than origin value"
    else:
        lat_limits = (-90, 90)

    if lon_limits is not None:
        assert (
            origin[1] >= lon_limits[0]
        ), "Origin lower limit greater than origin value"
        assert origin[1] <= lon_limits[1], "Origin upper limit less than origin value"
    else:
        lon_limits = (0, 360) if lon_mode == 360 else (-180, 180)

    lat_vector = np.hstack(
        [
            np.flip(
                np.arange(
                    origin[0], lat_limits[0] - resolution * use_poles, -resolution
                )
            ),
            np.arange(origin[0], lat_limits[1] + resolution * use_poles, resolution),
        ]
    )

    if lon_mode == 360:
        lon_vector = np.hstack(
            [
                np.flip(np.arange(origin[1], lon_limits[0] - resolution, -resolution)),
                np.arange(origin[1], lon_limits[1], resolution),
            ]
        )

    else:
        lon_vector = np.hstack(
            [
                np.flip(np.arange(origin[1], lon_limits[0], -resolution)),
                np.arange(origin[1], lon_limits[1] + resolution, resolution),
            ]
        )

    # Sanitize outputs
    lat_vector = np.unique(lat_vector)
    lon_vector = np.unique(lon_vector)

    return lat_vector, lon_vector


def regression_2_RI(track_df, **kwargs):
    df_mode = kwargs.get("mode", "ibtracs")
    if df_mode == "ibtracs":
        cols = {
            "SID": "SID",
            "lat": "LAT",
            "lon": "LON",
            "initial_time": "ISO_TIME",
        }
    elif df_mode == "preds":
        cols = {
            "SID": "SID",
            "lat": "lat",
            "lon": "lon",
            "initial_time": "Initial Time",
            "valid_time": "Valid Time",
        }
    else:
        raise ValueError(f"Invalid mode: {df_mode}. Expected 'ibtracs' or 'preds'.")


def fnv1a_hash(data):
    # FNV-1a 32-bit hash parameters
    fnv_prime = 0x01000193
    fnv_offset_basis = 0x811C9DC5

    hash_value = fnv_offset_basis
    for byte in data.encode():
        hash_value ^= byte
        hash_value *= fnv_prime
        hash_value &= 0xFFFFFFFF  # Ensure hash_value stays within 32-bit range

    return hash_value


def make_short_id(*args):
    # Step 1: Concatenate the strings
    concatenated_string = "".join(args)

    # Step 2: Hash the concatenated string using FNV-1a
    hash_value = fnv1a_hash(concatenated_string)

    # Step 3: Convert the hash to a hexadecimal string and truncate to 8 characters
    identifier = f"{hash_value:08x}"

    return identifier


# Lat-lon grid generator
def ll_gridder(**kwargs):
    """
    ll_gridder is a lat-lon grid generator that's useful for generator the
    data masks associated with the

    Parameters
    ----------
    origin : TYPE, optional
        DESCRIPTION. The default is (0,0).
    resolution : TYPE, optional
        DESCRIPTION. The default is 0.25.
    lon_mode : TYPE, optional
        DESCRIPTION. The default is 'pos'.
     : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    origin = kwargs.get("origin", (0, 0))
    resolution = kwargs.get("resolution", 0.25)
    lon_mode = kwargs.get("lon_mode", 360)
    lat_limits = kwargs.get("lat_limits", None)
    lon_limits = kwargs.get("lon_limits", None)
    use_poles = kwargs.get("use_poles", True)

    assert isinstance(
        use_poles, bool
    ), f"Unexpected type for use_poles:{type(use_poles)}. Expected boolean."
    assert (
        lon_mode == 180 or lon_mode == 360
    ), f"Invalid long mode in ll_gridder: {lon_mode}. Expected 180 or 360 as ints"

    lat_vector, lon_vector = axis_generator(
        origin=origin,
        resolution=resolution,
        lat_limits=lat_limits,
        lon_limits=lon_limits,
        lon_mode=lon_mode,
        use_poles=use_poles,
    )

    lats, lons = np.meshgrid(lat_vector, lon_vector)

    return lats, lons


# %%
# Great circle distance calculations
def haversine(latp, lonp, lat_list, lon_list, **kwargs):
    """──────────────────────────────────────────────────────────────────────────┐
      Haversine formula for calculating distance between target point (latp,
      lonp) and series of points (lat_list, lon_list). This function can handle
      2D lat/lon lists, but has been used with flattened data

      Based on:
      https://medium.com/@petehouston/calculate-distance-of-two-locations-on-earth-using-python-1501b1944d97


      Inputs:
          latp - latitude of target point

          lonp - longitude of target point

          lat_list - list of latitudess (lat_p-1, lat_p-2 ... lon_p-n)

          lon_list - list of longitudess (lon_p-1, lon_p-2 ... lon_p-n)

      Outputs:

    └──────────────────────────────────────────────────────────────────────────"""
    kwargs.get("epsilon", 1e-6)

    latp = np.radians(latp)
    lonp = np.radians(lonp)
    lat_list = np.radians(lat_list)
    lon_list = np.radians(lon_list)

    dlon = lonp - lon_list
    dlat = latp - lat_list
    a = np.power(np.sin(dlat / 2), 2) + np.cos(lat_list) * np.cos(latp) * np.power(
        np.sin(dlon / 2), 2
    )

    # Assert that sqrt(a) is within machine precision of 1
    # assert np.all(np.sqrt(a) <= 1 + epsilon), 'Invalid argument for arcsin'

    # Check if square root of a is a valid argument for arcsin within machine precision
    # If not, set to 1 or -1 depending on sign of a
    a = np.where(np.sqrt(a) <= 1, a, np.sign(a))

    return 2 * 6371 * np.arcsin(np.sqrt(a))


def latlon_2_cartesian(lat, lon, **kwargs):
    """
    Convert latitude/longitude to simple equirectangular Cartesian coordinates
    referenced to (0,0).

    Parameters
    ----------
    lat : array-like
        Latitude in degrees.
    lon : array-like
        Longitude in degrees.
    radius : float, optional
        Sphere radius in kilometers. Default is 6371 km.

    Returns
    -------
    x, y : array-like
        Cartesian coordinates in **kilometers**.

    Notes
    -----
    Previous implementation incorrectly used `lon * lat` (both in degrees)
    for the x-coordinate and omitted converting longitude to radians, which
    led to values larger by several orders of magnitude and occasional
    overflows. The corrected formula is:

        x = R * cos(lat_rad) * lon_rad
        y = R * lat_rad

    which produces arc-length distances along the tangential directions in km.
    """
    radius = kwargs.get("radius", 6371.0)

    lat = np.asarray(lat)
    lon = np.asarray(lon)

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    x = radius * np.cos(lat_rad) * lon_rad
    y = radius * lat_rad

    return x, y


# %%


def list_to_list_haversine(point_list1, point_list2, **kwargs):
    """
    For two list of points (including lat/lon pairs, and lists of equal size),
    calculate the great circle distance between each pair of points in the two lists.

    Inputs:

    point_list1 - list of lat/lon pairs (lat1, lon1), (lat2, lon2) ... (latn, lonn)

    point_list2 - list of lat/lon pairs (lat1, lon1), (lat2, lon2) ... (latn, lonn)

    Outputs:

    distance - list of distances between each pair of points in the two lists (dist1,
               dist2, ... distn) in kilometers.

    """

    lat1 = np.radians(point_list1[:, 0])[:, None]
    lon1 = np.radians(point_list1[:, 1])[:, None]
    lat2 = np.radians(point_list2[:, 0])[:, None]
    lon2 = np.radians(point_list2[:, 1])[:, None]

    dlon = lon1 - lon2
    dlat = lat1 - lat2
    a = np.power(np.sin(dlat / 2), 2) + np.cos(lat2) * np.cos(lat1) * np.power(
        np.sin(dlon / 2), 2
    )

    # Assert that sqrt(a) is within machine precision of 1
    # assert np.all(np.sqrt(a) <= 1 + epsilon), 'Invalid argument for arcsin'

    # Check if square root of a is a valid argument for arcsin within machine precision
    # If not, set to 1 or -1 depending on sign of a
    a = np.where(np.sqrt(a) <= 1, a, np.sign(a))

    return 2 * 6371 * np.arcsin(np.sqrt(a))


# %%


def get_coord_vars(dataset):
    lon_coord = lat_coord = time_coord = level_coord = leadtime_coord = False
    coord_array = np.array(dataset.coords)

    for idx, coord in enumerate(coord_array):
        # retrieve the longitude coordinate if available
        if not lon_coord:
            lon_coord = (
                coord_array[idx]
                if np.any(
                    [
                        valid_name in coord
                        for valid_name in constants.valid_coords["longitude"]
                    ]
                )
                else False
            )

        if not lat_coord:
            lat_coord = (
                coord_array[idx]
                if np.any(
                    [
                        valid_name in coord
                        for valid_name in constants.valid_coords["latitude"]
                    ]
                )
                else False
            )

        if not time_coord:
            time_coord = (
                coord_array[idx]
                if np.any(
                    [
                        valid_name == coord
                        for valid_name in constants.valid_coords["time"]
                    ]
                )
                else False
            )

        if not leadtime_coord:
            leadtime_coord = (
                coord_array[idx]
                if np.any(
                    [
                        valid_name in coord
                        for valid_name in constants.valid_coords["leadtime"]
                    ]
                )
                else False
            )

        if not level_coord:
            level_coord = (
                coord_array[idx]
                if np.any(
                    [
                        valid_name in coord
                        for valid_name in constants.valid_coords["level"]
                    ]
                )
                else False
            )

    assert np.all(
        [bool(lon_coord), bool(lat_coord), bool(time_coord)]
    ), "Missing lat, lon, or time coordinates in dataset"

    return lat_coord, lon_coord, time_coord, level_coord, leadtime_coord


def hurricane_symbol():
    # Code to generate a hurricane symbol marker for plotting
    # Code taken from:
    # https://stackoverflow.com/questions/44726675/custom-markers-using-python-matplotlib
    u = np.array(
        [
            [2.444, 7.553],
            [0.513, 7.046],
            [-1.243, 5.433],
            [-2.353, 2.975],
            [-2.578, 0.092],
            [-2.075, -1.795],
            [-0.336, -2.870],
            [2.609, -2.016],
        ]
    )
    u[:, 0] -= 0.098
    codes = [1] + [2] * (len(u) - 2) + [2]
    u = np.append(u, -u[::-1], axis=0)
    codes += codes

    return mpl.path.Path(3 * u, codes, closed=False)


def get_TC_seasons(
    **kwargs,
):
    tracks_path = kwargs.get(
        "tracks_path",
        "/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/tracks/ibtracs/",
    )

    cols = kwargs.get("track_columns", constants.ibtracs_cols)
    min_season = kwargs.get("min_season", 1990)
    track_data = read_hist_track_file(
        track_cols=cols,
        backend=cols._track_cols__metadata.get("loader"),
        **kwargs,
    )

    uidx = cols._track_cols__metadata.get("UID")
    name = cols._track_cols__metadata.get("COSMETIC_NAME")
    x = cols._track_cols__metadata.get("X_coord")
    y = cols._track_cols__metadata.get("Y_coord")
    t = cols._track_cols__metadata.get("TIME_coord")
    alt_id = cols._track_cols__metadata.get("ALT_ID")
    wind = cols._track_cols__metadata.get("WIND")
    pres = cols._track_cols__metadata.get("PRES")
    season_coord = cols._track_cols__metadata.get("SEASON", None)

    if season_coord is not None:
        track_data = track_data[track_data[season_coord] >= min_season]
        unique_seasons = track_data[season_coord].unique()
    else:
        track_data = track_data[track_data[t].dt.year >= min_season]
        unique_seasons = track_data[t].dt.year.unique()

    if kwargs.get("season_list", None):
        try:
            assert isinstance(
                kwargs.get("season_list"), list
            ), f"Invalid type for season_list: {type(kwargs.get('season_list'))}. Expected list"
            unique_seasons = [
                selection
                for selection in unique_seasons
                if selection in kwargs.get("season_list")
            ]
        except:
            print("Invalid season list. Returning all seasons.")
            pass

    season_dict = {}
    for season in unique_seasons:
        if season is not None:
            unique_storms = track_data[track_data[season_coord] == season][
                uidx
            ].unique()
        else:
            unique_storms = track_data[track_data[t].dt.year == season][uidx].unique()

        season_storms = []
        for storm in unique_storms:
            storm_data = track_data[track_data[uidx] == storm]
            temp_alt = storm_data[alt_id][storm_data[alt_id].str.strip() != ""]
            temp_alt = temp_alt.mode()
            if len(temp_alt) > 1:
                print(f"Storm {storm} has multiple alternative IDs: {temp_alt}")
                print(f"Using the first alternative ID: {temp_alt[0]}")
                temp_alt = temp_alt[0]
            elif len(temp_alt) == 1:
                temp_alt = temp_alt.item()
            else:
                temp_alt = None
            track = tc_track(
                UID=storm_data[uidx].iloc[0],
                NAME=storm_data[name].iloc[0],
                track=storm_data[[y, x]].to_numpy(),
                timestamps=storm_data[t].to_numpy(),
                ALT_ID=temp_alt,
                wind=storm_data[wind].to_numpy(),
                pres=storm_data[pres].to_numpy(),
                storm_season=season,
                **kwargs,
            )
            season_storms.append(track)
        season_dict[season] = season_storms

    return season_dict


def get_timeseries_sets(splits: dict, season_dict: dict, **kwargs):
    assert isinstance(
        splits, dict
    ), f"Invalid type for splits: {type(splits)}. Expected dict"

    assert isinstance(
        season_dict, dict
    ), f"Invalid type for season data: {type(season_dict)}. Expected dict"

    assert np.all(
        [key in ["train", "test", "validation"] for key in splits.keys()]
    ), "Invalid split values. Unknown key in splits dictionary; expected 'train', 'test', and 'validation' keys"

    datadir_path = kwargs.get("datadir", os.path.join(repo_path, "data"))
    timeseries_steps = kwargs.get("timeseries_length", 6)
    delta_target = kwargs.get("delta_target", 24)  # Target intensity lead time in hours

    available_seasons = list(season_dict.keys())

    test_strategy = kwargs.get("test_strategy", "last")
    if test_strategy in ["last", "random", "first"]:
        train_set = None
        val_set = None
        test_set = None
        assert (np.sum([value for value in splits.values()]) <= 1) and (
            np.all([value > 0 for value in splits.values()])
        ), (
            "Invalid split values. Sum of splits should be less than "
            "or equal to 1, and all splits must be greater than 0"
        )
        if test_strategy == "last":
            test_size = int(np.ceil(splits["test"] * len(available_seasons)))
            test_set = available_seasons[-test_size:]
            other_seasons = available_seasons[:-test_size]
            if "validation" in splits.keys():
                val_size = int(np.ceil(splits["validation"] * len(available_seasons)))
                val_set = other_seasons[-val_size:]
                train_set = other_seasons[:-val_size]
            else:
                if splits["test"] + splits["train"] < 1:
                    validation_size = 1 - splits["test"] - splits["train"]
                    if validation_size > splits["test"]:
                        round_func = np.floor
                    else:
                        round_func = np.ceil
                    val_size = int(round_func(validation_size * len(available_seasons)))
                    val_set = other_seasons[-val_size:]
                    train_set = other_seasons[:-val_size]
                else:
                    train_set = other_seasons
        elif test_strategy == "random":
            random_generator = kwargs.get("random_generator", np.random.default_rng(42))
            test_size = int(np.ceil(splits["test"] * len(available_seasons)))
            test_set = random_generator.choice(
                available_seasons, test_size, replace=False
            )
            other_seasons = np.setdiff1d(available_seasons, test_set)
            if "validation" in splits.keys():
                val_size = int(np.ceil(splits["validation"] * len(available_seasons)))
                val_set = random_generator.choice(
                    other_seasons, val_size, replace=False
                )
                train_set = np.setdiff1d(other_seasons, val_set)
            else:
                if splits["test"] + splits["train"] < 1:
                    validation_size = 1 - splits["test"] - splits["train"]
                    if validation_size > splits["test"]:
                        round_func = np.floor
                    else:
                        round_func = np.ceil
                    val_size = int(round_func(validation_size * len(available_seasons)))
                    val_set = random_generator.choice(
                        other_seasons, val_size, replace=False
                    )
                    train_set = np.setdiff1d(other_seasons, val_set)
                else:
                    train_set = other_seasons
        elif test_strategy == "first":
            test_size = int(np.ceil(splits["test"] * len(available_seasons)))
            test_set = available_seasons[:test_size]
            other_seasons = available_seasons[test_size:]
            if "validation" in splits.keys():
                val_size = int(np.ceil(splits["validation"] * len(available_seasons)))
                val_set = other_seasons[:val_size]
                train_set = other_seasons[val_size:]
            else:
                if splits["test"] + splits["train"] < 1:
                    validation_size = 1 - splits["test"] - splits["train"]
                    if validation_size > splits["test"]:
                        round_func = np.floor
                    else:
                        round_func = np.ceil
                    val_size = int(round_func(validation_size * len(available_seasons)))
                    val_set = other_seasons[:val_size]
                    train_set = other_seasons[val_size:]
                else:
                    train_set = other_seasons
    elif test_strategy == "custom":
        for key, item in splits.items():
            assert isinstance(
                item, list
            ), f"Invalid type for {key} in splits: {type(item)}. Expected list."
            assert np.all(
                [isinstance(value, int) for value in item]
            ), f"Invalid type for values in {key} in splits: {type(item)}. Expected int."
            assert np.all(
                [value in available_seasons for value in item]
            ), f"Invalid values for {key} in splits: {item}. Season not found in datadir."
            assert key in ["train", "test", "validation"], "Unsupported key in splits."
            assert len(item) > 0, f"Empty list for {key} in splits."
            if key == "train":
                train_set = item
            elif key == "test":
                test_set = item
            else:
                val_set = item

    if train_set is not None:
        train = []
        for season in train_set:
            train.extend(season_dict[season])

    if val_set is not None:
        validation = []
        for season in val_set:
            validation.extend(season_dict[season])

    if test_set is not None:
        test = []
        for season in test_set:
            test.extend(season_dict[season])

    data_dict = {}
    for dataset in ["train", "validation", "test"]:
        setdata = {
            "data": [],
            "intensity_deltas": [],
            "base_intensities": [],
            "base_times": [],
        }
        for storm in eval(dataset):
            # Get the 6 hourly timestamps
            candidate_timestamps = pd.to_datetime(storm.timestamps)
            # Filter to only include timestamps in 00h, 06h, 12h, 18h
            candidate_timestamps = candidate_timestamps[
                np.isin(candidate_timestamps.hour, [0, 6, 12, 18])
            ].to_numpy()
            try:
                data, timestamps = storm.timeseries.serve(**kwargs)
            except Exception as e:
                print(f"Storm {str(storm)} produced no timeseries data. Skipping...")
                # print the traceback if verbose
                if kwargs.get("verbose", False):
                    print(traceback.format_exc())
                continue
            if data is not None and (len(candidate_timestamps) > timeseries_steps):
                rolling_timestamps = rolling_window(
                    candidate_timestamps, timeseries_steps
                )

                bool_mask = np.all(np.isin(rolling_timestamps, timestamps), axis=1)
                rolling_timestamps = rolling_timestamps[bool_mask]

                # get the ground truth values for the intensities
                intensity, time = storm.get_ground_truth(**kwargs)

                # find the target times
                target_times = time + np.timedelta64(delta_target, "h")
                valid_mask = np.isin(time, target_times)
                target_times = target_times[valid_mask]

                # initiate a target intensity array with nans
                target_intensities = np.full(
                    (target_times.shape[0], intensity.shape[1]), np.nan
                )

                # Use the target times to find the corresponding intensities in the ground truth intensity
                # mask = np.isin(time, target_times)
                valid_mask = np.isin(target_times, time)
                target_intensities[valid_mask] = intensity[np.isin(time, target_times)]

                base_times = target_times - np.timedelta64(delta_target, "h")

                # Let's initialize the base intensities with nans
                base_intensities = np.full_like(target_intensities, np.nan)
                base_times = np.full_like(target_times, np.nan)
                mask = np.isin(time, base_times)

                base_intensities = intensity[mask]
                base_times = time[mask]

                intensity_deltas = target_intensities - base_intensities

                rolling_mask = np.isin(base_times, rolling_timestamps[:, -1])
                rolling_base = base_intensities[rolling_mask]
                rolling_time = base_times[rolling_mask]
                rolling_delta = intensity_deltas[rolling_mask]

                data_series = np.full(
                    (*rolling_timestamps.shape, data.shape[1]), np.nan
                )

                for i in range(len(data_series)):
                    mask = np.isin(timestamps, rolling_timestamps[i])
                    data_series[i] = data[mask]

                # Ensure no nans in output
                data_check = np.any(np.any(np.isnan(data_series), axis=2), axis=1)
                delta_check = np.any(np.isnan(rolling_delta), axis=1)
                base_check = np.any(np.isnan(rolling_base), axis=1)
                time_check = np.any(np.isnan(rolling_time))
                check = np.any(
                    np.vstack([data_check, delta_check, base_check, time_check]), axis=0
                )

                rolling_delta = rolling_delta[~check]
                rolling_base = rolling_base[~check]
                roilling_time = rolling_time[~check]
                data_series = data_series[~check]

                if data_series.shape[-1] != len(
                    kwargs.get(
                        "variables",
                        constants.default_ships_vars,
                    )
                ):
                    print(f"At least one empty variable in storm {storm}. Skipping...")
                else:
                    setdata["data"].append(data_series)
                    setdata["intensity_deltas"].append(rolling_delta)
                    setdata["base_intensities"].append(rolling_base)
                    setdata["base_times"].append(rolling_time)

        setdata["data"] = np.vstack(setdata["data"])
        setdata["intensity_deltas"] = np.vstack(setdata["intensity_deltas"])
        setdata["base_intensities"] = np.vstack(setdata["base_intensities"])
        setdata["base_times"] = np.vstack(setdata["base_times"])
        data_dict[dataset] = setdata
    return data_dict


def get_ai_sets(splits: dict, **kwargs):
    """
    Function to retrieve the training, validation, and testing set for TC tracks.
    The function will return a list of TC_tracks and a dictionary with the information
    retrieved by the serving function. By default we will assume we will be working
    with AI_data

    inputs:
        splits: dict, required
            Dictionary containing the split information for the data. The dictionary
            should contain the following keys:
                - train
                - val (optional)
                - test
            Each key should contain a list of floats corresponding to the relative size
            between the splits.
            If the test_strategy is 'custom', the test split should contain a list of
            storm seasons to use for the test set.
        test_strategy: STR, optional
            Strategy to use for the test set. The default is 'last', which will leave
            the most recent storm seasons for the test set. Other options include:
                - 'random': Randomly select the test set
                - 'first': Leave the first storm seasons for the test set
                - 'custom': Use a custom list of storm seasons for the test set

    outputs:
        sets: dict
            Dictionary containing the sets of storm seasons for training, validation, and
            testing
        data: dict
            Dictionary containing the AI data for each set
            Keys: 'train', 'val', 'test'

    parameters:
        datadir: STR, optional
            Path to the directory containing the storm data. The default is
            os.path.join(repo_path, 'data')
        debug: BOOL, optional
            Flag to print out debugging information. The default is False
        verbose: Bool, optional
            Flag to print out verbose information. The default is False
    """

    assert isinstance(
        splits, dict
    ), f"Invalid type for splits: {type(splits)}. Expected dict"

    assert np.all(
        [key in ["train", "test", "validation"] for key in splits.keys()]
    ), "Invalid split values. Unknown key in splits dictionary; expected 'train', 'test', and 'validation' keys"

    datadir_path = kwargs.get("datadir", os.path.join(repo_path, "data"))

    # get the folders in datadir and filter out the ones that are not storm seasons
    season_folders = [folder for folder in os.listdir(datadir_path) if folder.isdigit()]
    season_folders = list(np.sort(np.array(season_folders).astype(int)))

    test_strategy = kwargs.get("test_strategy", "last")
    if test_strategy in ["last", "random", "first"]:
        train_set = None
        val_set = None
        test_set = None
        assert (np.sum([value for value in splits.values()]) <= 1) and (
            np.all([value > 0 for value in splits.values()])
        ), (
            "Invalid split values. Sum of splits should be less than "
            "or equal to 1, and all splits must be greater than 0"
        )
        if test_strategy == "last":
            test_size = int(np.ceil(splits["test"] * len(season_folders)))
            test_set = season_folders[-test_size:]
            other_folders = season_folders[:-test_size]
            if "validation" in splits.keys():
                val_size = int(np.ceil(splits["validation"] * len(season_folders)))
                val_set = other_folders[-val_size:]
                train_set = other_folders[:-val_size]
            else:
                if splits["test"] + splits["train"] < 1:
                    validation_size = 1 - splits["test"] - splits["train"]
                    if validation_size > splits["test"]:
                        round_func = np.floor
                    else:
                        round_func = np.ceil
                    val_size = int(round_func(validation_size * len(season_folders)))
                    val_set = other_folders[-val_size:]
                    train_set = other_folders[:-val_size]
                else:
                    train_set = other_folders
        elif test_strategy == "random":
            test_size = int(np.ceil(splits["test"] * len(season_folders)))
            test_set = np.random.choice(season_folders, test_size, replace=False)
            other_folders = np.setdiff1d(season_folders, test_set)
            if "validation" in splits.keys():
                val_size = int(np.ceil(splits["validation"] * len(season_folders)))
                val_set = np.random.choice(other_folders, val_size, replace=False)
                train_set = np.setdiff1d(other_folders, val_set)
            else:
                if splits["test"] + splits["train"] < 1:
                    validation_size = 1 - splits["test"] - splits["train"]
                    if validation_size > splits["test"]:
                        round_func = np.floor
                    else:
                        round_func = np.ceil
                    val_size = int(round_func(validation_size * len(season_folders)))
                    val_set = np.random.choice(other_folders, val_size, replace=False)
                    train_set = np.setdiff1d(other_folders, val_set)
                else:
                    train_set = other_folders
        elif test_strategy == "first":
            test_size = int(np.ceil(splits["test"] * len(season_folders)))
            test_set = season_folders[:test_size]
            other_folders = season_folders[test_size:]
            if "validation" in splits.keys():
                val_size = int(np.ceil(splits["validation"] * len(season_folders)))
                val_set = other_folders[:val_size]
                train_set = other_folders[val_size:]
            else:
                if splits["test"] + splits["train"] < 1:
                    validation_size = 1 - splits["test"] - splits["train"]
                    if validation_size > splits["test"]:
                        round_func = np.floor
                    else:
                        round_func = np.ceil
                    val_size = int(round_func(validation_size * len(season_folders)))
                    val_set = other_folders[:val_size]
                    train_set = other_folders[val_size:]
                else:
                    train_set = other_folders

        sets = {
            "train": get_TC_seasons(season_list=train_set, **kwargs),
            "test": get_TC_seasons(season_list=test_set, **kwargs),
        }
        if val_set is not None:
            sets["validation"] = get_TC_seasons(season_list=val_set, **kwargs)
    elif test_strategy == "custom":
        sets = {}
        for key, item in splits.items():
            assert isinstance(
                item, list
            ), f"Invalid type for {key} in splits: {type(item)}. Expected list."
            assert np.all(
                [isinstance(value, int) for value in item]
            ), f"Invalid type for values in {key} in splits: {type(item)}. Expected int."
            assert np.all(
                [value in season_folders for value in item]
            ), f"Invalid values for {key} in splits: {item}. Season not found in datadir."
            assert key in ["train", "test", "validation"], "Unsupported key in splits."
            assert len(item) > 0, f"Empty list for {key} in splits."
            sets[key] = get_TC_seasons(
                season_list=item, datadir_path=datadir_path, **kwargs
            )

    else:
        raise ValueError(f"Unsupported test strategy {test_strategy}")

    data = {}
    for key, data_set in sets.items():
        if isinstance(data_set, dict):
            input_data = None
            target_data = None
            t_data = None
            lead_data = None

            if kwargs.get("base_intensity", True):
                intensity = None
                if kwargs.get("base_position", False):
                    position = None

            # Parallel processing
            if kwargs.get("parallel", True):
                for season, storms in data_set.items():
                    progress = kwargs.get("progress_indicator", True)
                    if progress:
                        print(f"Processing {season}", end="", flush=True)

                        def processor(storm, **kwargs):
                            result = storm.ai.serve(**kwargs)

                            if kwargs.get("base_position", False):
                                assert kwargs.get(
                                    "base_intensity", True
                                ), "Base position requires that the base intensity be requested. Set base_intensity to True."

                            if (
                                kwargs.get("base_intensity", True)
                                and result is not None
                                and result[0].size > 0
                            ):
                                _, _, t, leads = result
                                base_time = t - leads.astype("timedelta64[h]")

                                matches = base_time[:, None] == storm.timestamps
                                # Use numpy.nonzero to find the indices of the matches
                                indices = np.nonzero(matches)[1]

                                base_intensity = np.vstack(
                                    [
                                        storm.wind[indices].astype(int),
                                        storm.pressure[indices].astype(int),
                                    ]
                                ).T

                                if kwargs.get("base_position", False):
                                    base_position = np.vstack(
                                        [
                                            storm.track[indices, 0].astype(float),
                                            storm.track[indices, 1].astype(float),
                                        ]
                                    ).T

                                    result = (*result, base_intensity, base_position)
                                else:
                                    result = (*result, base_intensity)

                                if progress:
                                    print(".", end="", flush=True)

                                return result

                    n_jobs = kwargs.get("n_jobs", jl.cpu_count())
                    temp_data = jl.Parallel(n_jobs=n_jobs)(
                        jl.delayed(processor)(storm, **kwargs) for storm in storms
                    )
                    if progress:
                        print(" Done!", flush=True)

                    # Remove any empty entries
                    temp_data = [entry for entry in temp_data if entry is not None]

                    for result in temp_data:
                        if kwargs.get("base_intensity", True):
                            if kwargs.get("base_position", False):
                                (
                                    inputs,
                                    outputs,
                                    t,
                                    leads,
                                    base_intensity,
                                    base_position,
                                ) = result
                            else:
                                inputs, outputs, t, leads, base_intensity = result
                        else:
                            inputs, outputs, t, leads = result

                        if input_data is None:
                            input_data = inputs
                            target_data = outputs
                            t_data = t
                            lead_data = leads
                            if kwargs.get("base_intensity", True):
                                if kwargs.get("base_position", False):
                                    position = base_position
                                intensity = base_intensity
                        else:
                            input_data = da.vstack((input_data, inputs))
                            target_data = da.vstack((target_data, outputs))
                            t_data = np.hstack((t_data, t))
                            lead_data = np.hstack((lead_data, leads))
                            if kwargs.get("base_intensity", True):
                                intensity = da.vstack((intensity, base_intensity))
                                if kwargs.get("base_position", False):
                                    position = da.vstack((position, base_position))

            # Serial processing
            else:
                if kwargs.get("base_intensity", True):
                    raise NotImplementedError(
                        "Base intensity calculation not implemented for serial processing"
                    )
                raise NotImplementedError("Serial processing not implemented")
                for season, storms in data_set.items():
                    if kwargs.get("progress_indicator", True):
                        print(f"Processing {season}", end="", flush=True)
                    for idx, storm in enumerate(storms):
                        if kwargs.get("progress_indicator", True) and (idx // 10 == 0):
                            print(".", end="", flush=True)
                        inputs = None
                        outputs = None
                        try:
                            inputs, outputs, t, leads = storm.serve_ai_data(**kwargs)
                        except Exception as e:
                            if kwargs.get("debug", False):
                                print(f"Failed to process {str(storm)}")
                                print(f"Error: {e}")
                        if (inputs is not None) and (outputs is not None):
                            if input_data is None:
                                input_data = inputs
                                target_data = outputs
                                t_data = t
                                lead_data = leads
                            else:
                                input_data = da.vstack((input_data, inputs))
                                target_data = da.vstack((target_data, outputs))
                                t_data = np.hstack((t_data, t))
                                lead_data = np.hstack((lead_data, leads))
                    if kwargs.get("progress_indicator", True):
                        print(" Done!", flush=True)

            data[key] = {
                "inputs": input_data,
                "outputs": target_data,
                "time": t_data[:, None],
                "leadtime": lead_data[:, None],
            }
            if kwargs.get("base_intensity", True):
                data[key]["base_intensity"] = intensity
                if kwargs.get("base_position", False):
                    data[key]["base_position"] = position

    return sets, data


def analyze_displacement_hist(storm_set, leadtime, **kwargs):
    global_distance = None
    global_dlon = None
    global_dlat = None
    for season, storms in storm_set.items():
        for storm in storms:
            distance, dlon, dlat = storm.historical_displacement(
                leadtime=leadtime, **kwargs
            )
            # print(distance.shape, dlon.shape, dlat.shape)
            if global_distance is None and len(distance) > 0:
                global_distance = distance
                global_dlon = dlon
                global_dlat = dlat
            elif len(distance) > 0:
                global_distance = np.vstack((global_distance, distance))
                global_dlon = np.vstack((global_dlon, dlon))
                global_dlat = np.vstack((global_dlat, dlat))

    return global_distance, global_dlon, global_dlat


def get_reanal_sets(splits: dict, **kwargs):
    """
    Function to retrieve the training, validation, and testing set for TC tracks.
    The function will return a list of TC_tracks and a dictionary with the information
    retrieved by the serving function. By default we will assume we will be working
    with AI_data

    inputs:
        splits: dict, required
            Dictionary containing the split information for the data. The dictionary
            should contain the following keys:
                - train
                - val (optional)
                - test
            Each key should contain a list of floats corresponding to the relative size
            between the splits.
            If the test_strategy is 'custom', the test split should contain a list of
            storm seasons to use for the test set.
        test_strategy: STR, optional
            Strategy to use for the test set. The default is 'last', which will leave
            the most recent storm seasons for the test set. Other options include:
                - 'random': Randomly select the test set
                - 'first': Leave the first storm seasons for the test set
                - 'custom': Use a custom list of storm seasons for the test set

    outputs:
        sets: dict
            Dictionary containing the sets of storm seasons for training, validation, and
            testing
        data: dict
            Dictionary containing the AI data for each set
            Keys: 'train', 'val', 'test'

    parameters:
        datadir: STR, optional
            Path to the directory containing the storm data. The default is
            os.path.join(repo_path, 'data')
        debug: BOOL, optional
            Flag to print out debugging information. The default is False
        verbose: Bool, optional
            Flag to print out verbose information. The default is False
    """

    assert isinstance(
        splits, dict
    ), f"Invalid type for splits: {type(splits)}. Expected dict"

    assert np.all(
        [key in ["train", "test", "validation"] for key in splits.keys()]
    ), "Invalid split values. Unknown key in splits dictionary; expected 'train', 'test', and 'validation' keys"

    datadir_path = kwargs.get("datadir", os.path.join(repo_path, "data"))

    # get the folders in datadir and filter out the ones that are not storm seasons
    season_folders = [folder for folder in os.listdir(datadir_path) if folder.isdigit()]
    season_folders = list(np.sort(np.array(season_folders).astype(int)))

    test_strategy = kwargs.get("test_strategy", "last")
    if test_strategy in ["last", "random", "first"]:
        train_set = None
        val_set = None
        test_set = None
        assert (np.sum([value for value in splits.values()]) <= 1) and (
            np.all([value > 0 for value in splits.values()])
        ), (
            "Invalid split values. Sum of splits should be less than "
            "or equal to 1, and all splits must be greater than 0"
        )
        if test_strategy == "last":
            test_size = int(np.ceil(splits["test"] * len(season_folders)))
            test_set = season_folders[-test_size:]
            other_folders = season_folders[:-test_size]
            if "validation" in splits.keys():
                val_size = int(np.ceil(splits["validation"] * len(season_folders)))
                val_set = other_folders[-val_size:]
                train_set = other_folders[:-val_size]
            else:
                if splits["test"] + splits["train"] < 1:
                    validation_size = 1 - splits["test"] - splits["train"]
                    if validation_size > splits["test"]:
                        round_func = np.floor
                    else:
                        round_func = np.ceil
                    val_size = int(round_func(validation_size * len(season_folders)))
                    val_set = other_folders[-val_size:]
                    train_set = other_folders[:-val_size]
                else:
                    train_set = other_folders
        elif test_strategy == "random":
            test_size = int(np.ceil(splits["test"] * len(season_folders)))
            test_set = np.random.choice(season_folders, test_size, replace=False)
            other_folders = np.setdiff1d(season_folders, test_set)
            if "validation" in splits.keys():
                val_size = int(np.ceil(splits["validation"] * len(season_folders)))
                val_set = np.random.choice(other_folders, val_size, replace=False)
                train_set = np.setdiff1d(other_folders, val_set)
            else:
                if splits["test"] + splits["train"] < 1:
                    validation_size = 1 - splits["test"] - splits["train"]
                    if validation_size > splits["test"]:
                        round_func = np.floor
                    else:
                        round_func = np.ceil
                    val_size = int(round_func(validation_size * len(season_folders)))
                    val_set = np.random.choice(other_folders, val_size, replace=False)
                    train_set = np.setdiff1d(other_folders, val_set)
                else:
                    train_set = other_folders
        elif test_strategy == "first":
            test_size = int(np.ceil(splits["test"] * len(season_folders)))
            test_set = season_folders[:test_size]
            other_folders = season_folders[test_size:]
            if "validation" in splits.keys():
                val_size = int(np.ceil(splits["validation"] * len(season_folders)))
                val_set = other_folders[:val_size]
                train_set = other_folders[val_size:]
            else:
                if splits["test"] + splits["train"] < 1:
                    validation_size = 1 - splits["test"] - splits["train"]
                    if validation_size > splits["test"]:
                        round_func = np.floor
                    else:
                        round_func = np.ceil
                    val_size = int(round_func(validation_size * len(season_folders)))
                    val_set = other_folders[:val_size]
                    train_set = other_folders[val_size:]
                else:
                    train_set = other_folders

        sets = {
            "train": get_TC_seasons(season_list=train_set, **kwargs),
            "test": get_TC_seasons(season_list=test_set, **kwargs),
        }
        if val_set is not None:
            sets["validation"] = get_TC_seasons(season_list=val_set, **kwargs)
    elif test_strategy == "custom":
        sets = {}
        for key, item in splits.items():
            assert isinstance(
                item, list
            ), f"Invalid type for {key} in splits: {type(item)}. Expected list."
            assert np.all(
                [isinstance(value, int) for value in item]
            ), f"Invalid type for values in {key} in splits: {type(item)}. Expected int."
            assert np.all(
                [value in season_folders for value in item]
            ), f"Invalid values for {key} in splits: {item}. Season not found in datadir."
            assert key in ["train", "test", "validation"], "Unsupported key in splits."
            assert len(item) > 0, f"Empty list for {key} in splits."
            sets[key] = get_TC_seasons(
                season_list=item, datadir_path=datadir_path, **kwargs
            )

    else:
        raise ValueError(f"Unsupported test strategy {test_strategy}")

    data = {}
    for key, data_set in sets.items():
        if isinstance(data_set, dict):
            input_data = None
            target_data = None
            t_data = None
            lead_data = None

            if kwargs.get("base_intensity", True):
                intensity = None
                if kwargs.get("base_position", False):
                    position = None

            # Parallel processing
            if kwargs.get("parallel", True):
                for season, storms in data_set.items():
                    progress = kwargs.get("progress_indicator", True)
                    if progress:
                        print(f"Processing {season}", end="", flush=True)

                        def processor(storm, **kwargs):
                            if kwargs.get("verbose", False):
                                print(f"Processing {str(storm)}")
                            result = storm.ReAnal.serve(**kwargs)
                            # if result is not None:
                            #     for item in result:
                            #         print(item.shape)

                            if kwargs.get("base_position", False):
                                assert kwargs.get(
                                    "base_intensity", True
                                ), "Base position requires that the base intensity be requested. Set base_intensity to True."

                            if (
                                kwargs.get("base_intensity", True)
                                and result is not None
                                and result[0].size > 0
                            ):
                                _, _, t, leads = result
                                base_time = t - leads.astype("timedelta64[h]")

                                matches = base_time[:, None] == storm.timestamps
                                # Use numpy.nonzero to find the indices of the matches
                                indices = np.nonzero(matches)[1]

                                base_intensity = np.vstack(
                                    [
                                        storm.wind[indices].astype(int),
                                        storm.pressure[indices].astype(int),
                                    ]
                                ).T

                                if kwargs.get("base_position", False):
                                    base_position = np.vstack(
                                        [
                                            storm.track[indices, 0].astype(float),
                                            storm.track[indices, 1].astype(float),
                                        ]
                                    ).T

                                    result = (*result, base_intensity, base_position)
                                else:
                                    result = (*result, base_intensity)

                                if progress:
                                    print(".", end="", flush=True)

                                return result

                    n_jobs = kwargs.get("n_jobs", jl.cpu_count())
                    temp_data = jl.Parallel(n_jobs=n_jobs)(
                        jl.delayed(processor)(storm, **kwargs) for storm in storms
                    )
                    if progress:
                        print(" Done!", flush=True)

                    # Remove any empty entries
                    temp_data = [entry for entry in temp_data if entry is not None]

                    for result in temp_data:
                        if kwargs.get("base_intensity", True):
                            if kwargs.get("base_position", False):
                                (
                                    inputs,
                                    outputs,
                                    t,
                                    leads,
                                    base_intensity,
                                    base_position,
                                ) = result
                            else:
                                inputs, outputs, t, leads, base_intensity = result
                        else:
                            inputs, outputs, t, leads = result

                        if input_data is None:
                            input_data = inputs
                            target_data = outputs
                            t_data = t
                            lead_data = leads
                            if kwargs.get("base_intensity", True):
                                if kwargs.get("base_position", False):
                                    position = base_position
                                intensity = base_intensity
                        else:
                            input_data = da.vstack((input_data, inputs))
                            target_data = da.vstack((target_data, outputs))
                            t_data = np.hstack((t_data, t))
                            lead_data = np.hstack((lead_data, leads))
                            if kwargs.get("base_intensity", True):
                                intensity = da.vstack((intensity, base_intensity))
                                if kwargs.get("base_position", False):
                                    position = da.vstack((position, base_position))

            # Serial processing
            else:
                if kwargs.get("base_intensity", True):
                    raise NotImplementedError(
                        "Base intensity calculation not implemented for serial processing"
                    )
                raise NotImplementedError("Serial processing deprecated")
                for season, storms in data_set.items():
                    if kwargs.get("progress_indicator", True):
                        print(f"Processing {season}", end="", flush=True)
                    for idx, storm in enumerate(storms):
                        if kwargs.get("progress_indicator", True) and (idx // 10 == 0):
                            print(".", end="", flush=True)
                        inputs = None
                        outputs = None
                        try:
                            inputs, outputs, t, leads = storm.serve_ai_data(**kwargs)
                        except Exception as e:
                            if kwargs.get("debug", False):
                                print(f"Failed to process {str(storm)}")
                                print(f"Error: {e}")
                        if (inputs is not None) and (outputs is not None):
                            if input_data is None:
                                input_data = inputs
                                target_data = outputs
                                t_data = t
                                lead_data = leads
                            else:
                                input_data = da.vstack((input_data, inputs))
                                target_data = da.vstack((target_data, outputs))
                                t_data = np.hstack((t_data, t))
                                lead_data = np.hstack((lead_data, leads))
                    if kwargs.get("progress_indicator", True):
                        print(" Done!", flush=True)

            data[key] = {
                "inputs": input_data,
                "outputs": target_data,
                "time": t_data[:, None],
                "leadtime": lead_data[:, None],
            }
            if kwargs.get("base_intensity", True):
                data[key]["base_intensity"] = intensity
                if kwargs.get("base_position", False):
                    data[key]["base_position"] = position

    return sets, data


# %% Data Processing / Preprocessing Library
"""
This cell contains the following classes and functions:
    
"""


def read_hist_track_file(**kwargs):
    """
    Function to read storm track files that include all the historical data
    in a single file

    Parameters
    ----------
    tracks_path : STR, required
        Path to the folder holding the storm track in a single file.
        The default is f'{repo_path}/tracks/ibtracs'. By default the file
        is expected to be in csv format

    backend : FUNC, required
        file handler for loading the track file. The default is pandas'
        read_csv function.

    track_cols : track_cols object defined in constants.py, required
        Track columns object which contains data about the columns in the
        tracks file. Needs to at least be able to return the column names,
        column dtypes (excluding datetime types), and columns that should
        be parsed as dates.

    skip_rows : list, optional
        rows to skip when reading in data with pandas.read_csv
        Required when working with ibtracs, as there is a unit row after
        the header row, hence the default [1] value


    Returns
    -------
    track_object : type defined by backend. Default is pandas dataframe

    """
    tracks_path = kwargs.get(
        "tracks_path",
        f"/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/tracks/ibtracs/",
    )
    backend = kwargs.get("backend", pd.read_csv)
    track_cols = kwargs.get("track_cols", constants.ibtracs_cols)
    skip_rows = kwargs.get("skip_rows", [1])
    lon_mode = kwargs.get("lon_mode", 360)
    file_list = os.listdir(tracks_path)

    assert len(file_list) == 1, f"{tracks_path} has more than one file. Aborting."

    with open(f"{tracks_path}/{file_list[0]}") as handle:
        if backend == pd.read_csv:
            data = backend(
                handle,
                usecols=track_cols.get_colnames(),
                dtype=track_cols.get_dtypes(),
                skiprows=skip_rows,
                # parse_dates=track_cols.get_datetime_cols(),
                na_filter=False,  # Otherwise pandas interprets 'NA' as NaN
            )
            data["ISO_TIME"] = pd.to_datetime(data["ISO_TIME"])

        elif backend == "meteo_france":
            data = pd.read_csv(
                handle,
                usecols=track_cols.get_colnames(),
                dtype=track_cols.get_dtypes(),
                # parse_dates = {'date':[*track_cols.get_datetime_cols()]},
                sep=";",
            )
            datetime_cols = track_cols.get_datetime_cols()
            data["date"] = pd.to_datetime(
                data[datetime_cols]
                .copy()
                .rename(
                    columns={
                        "ANNEE": "year",
                        "MOIS": "month",
                        "JOUR": "day",
                        "HEURE_UTC": "hour",
                    }
                )
            )
        else:
            data = backend(handle)

    if lon_mode == 360:
        data["LON"] = data["LON"].apply(lambda x: x + 360 if x < 0 else x)
    elif lon_mode == 180:
        data["LON"] = data["LON"].apply(lambda x: x - 180 if x > 180 else x)
    else:
        raise ValueError(f"Unsupported lon_mode {lon_mode}")

    return data


def read_py_track_file(
    tracks_path=f"{repo_path}/tracks/sgstracks/", year=2000, backend=pd.read_csv
):
    """
    Function to read per-year separated storm track files

    Parameters
    ----------
    tracks_path : STR, required
        Path to the folder holding the storm tracks separated by year.
        The default is f'{repo_path}/tracks/'. By default the files are
        expected to be in csv format
    year : INT, required
        year of the tracks to load. The default is 2000.
    backend : FUNC, required
        file handler for loading the track file. The default is pandas'
        read_csv function.

    Returns
    -------
    track_object : type defined by backend. Default is pandas dataframe

    """
    # Retrieve the list of files from the path
    file_list = os.listdir(tracks_path)

    for idx, file in enumerate(file_list.copy()):
        if str(year) not in file:
            file_list.remove(file)

    assert (
        len(file_list) == 1
    ), f"{tracks_path} has more than one file for year {year}. Aborting."

    with open(f"{tracks_path}/{file_list[0]}") as handle:
        data = backend(handle)

    return data


# Functions to sanitize timestamp data
def sanitize_timestamps(timestamps, data, time_coord="time", **kwargs):
    # Sanitize timestamps because ibtracs includes unsual time steps,
    # e.g. 603781 (Katrina, 2005) includes 2005-08-25 22:30:00,
    # 2005-08-29 11:10:00, 2005-08-29 14:45:00

    leadtime_coord = kwargs.get("leadtime_coord", None)
    if leadtime_coord is not None:
        base_stamps = data[time_coord].values[:, None]
        lead_times = data[leadtime_coord].values
        valid_stamps = base_stamps + lead_times.astype("timedelta64[h]").flatten()
        ds_stamps = np.unique(np.hstack([base_stamps, valid_stamps]))
    else:
        ds_stamps = data[time_coord].values

    valid_steps = timestamps[np.isin(timestamps, ds_stamps)]

    return valid_steps


# def get_regrider(dataset: xr.Dataset, lat_coord: str, lon_coord: str, **kwargs):
#     # Generate lat and lot vectors
#     lat_vector, lon_vector = axis_generator(**kwargs)

#     assert (
#         lon_vector.shape <= dataset[lon_coord].shape
#     ), f"Longitude vector is too long. Expected <={dataset[lon_coord].shape} but got {lon_vector.shape}. Downscaling not yet supported."
#     assert (
#         lat_vector.shape <= dataset[lat_coord].shape
#     ), f"Latitude vector is too long. Expected <={dataset[lat_coord].shape} but got {lat_vector.shape}. Downscaling not yet supported."

#     # If either the lat or lon vectors are smaller than the dataset, regrid
#     if (
#         lon_vector.shape != dataset[lon_coord].shape
#         or lat_vector.shape != dataset[lat_coord].shape
#     ):
#         print("Making a regridder...")
#         # Generate empty array to cast data with
#         casting_array = xr.DataArray(
#             np.NaN,
#             dims=[lat_coord, lon_coord],
#             coords={lat_coord: lat_vector, lon_coord: lon_vector},
#         )
#         regridder = xe.Regridder(
#             dataset,
#             casting_array,
#             "bilinear",
#         )

#         return regridder(dataset)
#     else:
#         return None


# %% Class Definitions


# Class presenting tracks
class tc_track:
    def __str__(self):
        return f"TCBench_track_{self.uid}: {self.name}"

    def __repr__(self):
        return f"TCBench TC track Object; UID-{self.uid}  Name-{self.name}"

    def __init__(self, UID, NAME, track, timestamps, **kwargs) -> None:
        try:
            assert isinstance(
                UID, str
            ), f"Unique Identifier Error. {type(UID)} given when string was expected"
            assert isinstance(
                NAME, str
            ), f"Name Error. {type(NAME)} given when string was expected"
            assert isinstance(
                track, np.ndarray
            ), f"Track Error. {type(track)} given when numpy array was expected"
            assert isinstance(timestamps, np.ndarray) or isinstance(
                timestamps, pd.DataFrame
            ), f"Unsupported timestamp iterator {type(timestamps)}"
            assert np.issubdtype(
                timestamps.dtype, np.datetime64
            ), f"Unsupported timestamp type: {timestamps.dtype} - expected datetime64"

            self.uid = UID
            self.name = NAME
            self.track = (track,)[0]  # Weird workaround - look into fixing
            self.timestamps = timestamps
            self.wind = kwargs.get("wind", None)
            self.pressure = kwargs.get("pres", None)
            self.season = kwargs.get("storm_season", None)
            # Add alternate ID if available
            self.ALT_ID = kwargs.get("ALT_ID", None)
            self.datadir_path = kwargs.get(
                "datadir_path", os.path.join(os.getcwd(), "data")
            )
            # Check to see if file exists for UID, if not, create it
            self.filepath = os.path.join(
                self.datadir_path,
                # str(self.season),
            )

            # get the valid kwargs for the timeseries
            valid_kwargs = {
                key: kwargs.get(key, timeseries._timeseries__default_vals.get(key))
                for key in timeseries._timeseries__valid_kwargs
            }
            self.timeseries = timeseries(parent=self, **valid_kwargs)

            # get the valid kwargs for the AI data
            valid_kwargs = {
                key: kwargs.get(key, AI._AI__default_vals.get(key))
                for key in AI._AI__valid_kwargs
            }
            self.ai = AI(parent=self, **valid_kwargs)

            valid_kwargs = {
                key: kwargs.get(key, ReAnal._ReAnal__default_vals.get(key))
                for key in ReAnal._ReAnal__valid_kwargs
            }
            self.ReAnal = ReAnal(parent=self, **valid_kwargs)

            if not os.path.exists(self.filepath):
                if kwargs.get("verbose", False):
                    print("Filepath for processed data does not exist. Creating...")
                os.makedirs(self.filepath)
            else:
                # Get the filelist at the filepath
                file_list = os.listdir(self.filepath)

                # Filter out files that do not include the UID
                for idx, file in enumerate(file_list.copy()):
                    if self.uid not in file:
                        file_list.remove(file)

                # # For each type of data, load the data into the object
                # # Files follow <SID>.<TYPE>.<(S)ingle or (M)ultilevel>.nc naming convention
                # for file in file_list:
                #     # Get the variable name from the file name
                #     ds_type = file.split(".")[1]
                #     level_type = file.split(".")[2]
                #     # Load the data into the object
                #     setattr(
                #         self,
                #         ds_type + level_type + "ds",
                #         xr.open_dataset(self.filepath + file),
                #     )

        except Exception as e:
            print(
                f"Encountered and exception when instiating a tc_track object: \n{e}\n"
            )
            print(traceback.format_exc())

    def add_trackdata(
        self,
        var_dict: dict,
    ):
        """
        Function to add additional track data to the track object

        Parameters
        ----------
        var_dict : dict, required
            Dictionary containing the variable name as the key and the
            variable data as the value
        """
        for var, data in var_dict.items():
            self.__setattr__(var, data)

    def get_radmask(self, point, **kwargs):
        # read in parameters if submitted, otherwise use defaults
        radius = kwargs.get("radius", 500)
        grid = kwargs.get("grid", ll_gridder(**kwargs))
        distance_calculator = kwargs.get("distance_calculator", haversine)

        return (
            distance_calculator(
                point[0],
                point[1],
                grid[0],
                grid[1],
            )
            <= radius
        ).T[np.newaxis, :, :]

    def get_rectmask(self, point, **kwargs):
        # read in parameters if submitted, otherwise use defaults
        grid = kwargs.get("grid", ll_gridder(**kwargs))
        distance_calculator = kwargs.get("distance_calculator", haversine)
        circum_points = kwargs.get("circum_points", 4)

        distances = distance_calculator(
            point[0],
            point[1],
            grid[0],
            grid[1],
        ).T[np.newaxis, :, :]

        min_idx = np.unravel_index(distances.argmin(), distances.shape)

        output = np.zeros_like(distances)
        output[
            min_idx[0],
            min_idx[1] - circum_points : min_idx[1] + circum_points + 1,
            min_idx[2] - circum_points : min_idx[2] + circum_points + 1,
        ] = 1

        # TODO: fix so that the mask wraps around the grid if the circum_points exceed the grid size

        return output

    def get_mask(self, timestamp, **kwargs):
        # read in parameters if submitted, otherwise use defaults
        masktype = kwargs.get("masktype", "rect")

        if masktype == "rad":
            mask_getter = self.get_radmask
        elif masktype == "rect":
            mask_getter = self.get_rectmask
        else:
            raise ValueError(f"Unsupported mask type {masktype}")

        mask = None

        point = self.track[self.timestamps == timestamp][0]

        # Get the mask and flip the y axis to match the data
        temp_mask = mask_getter(point, **kwargs)[:, ::-1, :]

        # Tile the mask if the number of levels is greater than 1
        num_levels = kwargs.get("num_levels", 1)
        if num_levels > 1:
            temp_mask = np.tile(temp_mask, (num_levels, 1, 1))[np.newaxis, :, :, :]

        if mask is None:
            mask = temp_mask
        else:
            mask = np.vstack([mask, temp_mask])

        # Squeeze the mask to remove unnecessary dimensions. Flip y because reasons?
        mask = mask.squeeze()

        return mask

    def get_mask_series(self, timesteps, **kwargs):
        """
        This function returns a series of lat_lon masks corresponding
        to the track at the given timesteps.
        """
        # read in parameters if submitted, otherwise use defaults
        masktype = kwargs.get("masktype", "rect")

        if masktype == "rad":
            get_mask = self.get_radmask
        elif masktype == "rect":
            get_mask = self.get_rectmask
        else:
            raise ValueError(f"Unsupported mask type {masktype}")

        # Number of vertical levels
        num_levels = kwargs.get("num_levels", 1)
        # Number of leadtimes
        num_ldts = kwargs.get("num_leadtimes", None)

        mask = None
        for stamp in timesteps:
            point = self.track[self.timestamps == stamp][0]

            temp_mask = get_mask(point, **kwargs).astype(bool)[:, ::-1, :]

            # if num_levels > 1:
            #     temp_mask = np.tile(temp_mask, (num_levels, 1, 1))[np.newaxis, ...]

            # if num_ldts is not None:
            #     num_dims = 3 if num_levels > 1 else 2
            #     temp_mask = np.tile(
            #         temp_mask, (num_ldts, *np.ones(num_dims, dtype=int))
            #     )[np.newaxis, ...]

            if mask is None:
                mask = temp_mask
            else:
                mask = np.vstack([mask, temp_mask])

        return mask

    def process_data_collection(self, data_collection, **kwargs):
        # Function to add all variables from a data collection object
        # to the track object
        assert isinstance(data_collection, data_lib.Data_Collection) or isinstance(
            data_collection, data_lib.AI_Data_Collection
        ), f"Invalid data type {type(data_collection)}. Expected data_lib.data_collection or data_lib.AI_data_collection"
        try:
            if str(data_collection) == "TCBench_REANAL_DataCollection":
                # make a set of the years present in the tc_track timestamps
                years = sorted(
                    list(set(self.timestamps.astype("datetime64[Y]").astype(str)))
                )
                years = [int(year) for year in years]

                for var_type in data_collection.meta_dfs:
                    # Skip the var type if it's empty
                    if len(data_collection.meta_dfs[var_type].columns) == 0:
                        (f"Skipping {var_type} because it's empty")
                        continue

                    # Get the list of variables for the var type
                    var_list = data_collection.meta_dfs[var_type].index

                    assert (
                        len(var_list) > 0
                    ), f"No variables found in {var_type} - note that long names are required for specifying reanalysis data variables"

                    # Check if variables to load are specified
                    if kwargs.get("reanal_variables", None):
                        assert isinstance(
                            kwargs["reanal_variables"], list
                        ), f"Invalid type for variables: {type(kwargs['reanal_variables'])}. Expected list"
                        var_list = var_list[var_list.isin(kwargs["reanal_variables"])]

                    # try loading the ReAnalysis dataset
                    if hasattr(self, "ReAnal"):
                        try:
                            self.ReAnal.load()
                            print("ReAnalysis data loaded successfully")
                        except Exception as e:
                            if kwargs.get("verbose", False):
                                print(
                                    f"Failed to load ReAnalysis data for {self.uid}. Error: {e}. ",
                                    f"Proceeding with processing the data collection...",
                                )
                                traceback.print_exc()

                    if hasattr(self.ReAnal, "ds"):
                        # assume the if 5 variables are present, then the data is already loaded
                        if len(self.ReAnal.ds.data_vars) == 5:
                            print(
                                f"ReAnalysis data already loaded for {self.uid}. Skipping processing the data collection."
                            )
                            continue

                    # Get the metadata for the variables
                    var_meta = data_collection.meta_dfs[var_type].loc[var_list]

                    if len(var_list) == 0:
                        continue

                    var_dict = {}
                    try:
                        for var in var_list:
                            for year in years:
                                year = str(year)
                                # Assert each variable in the collection is available for each
                                # year the storm is active
                                assert (
                                    len(var_meta[year]) == var_meta[year].sum()
                                ), f"Missing variable {var} data for {year}"

                                dates = pd.to_datetime(self.timestamps)
                                dates = dates[dates.hour.isin([0, 6, 12, 18])]
                                dates = dates[dates.minute == 0].values

                                var_data, name_dict = data_collection.retrieve_ds(
                                    vars=[var], dates=dates, **kwargs
                                )
                                var_dict.update(name_dict)

                                # get coordinate names
                                (
                                    lat_coord,
                                    lon_coord,
                                    time_coord,
                                    level_coord,
                                    _,
                                ) = get_coord_vars(var_data)

                                valid_steps = sanitize_timestamps(
                                    self.timestamps, var_data
                                )
                                valid_steps = pd.to_datetime(valid_steps)

                                # Filter to only include 00, 06, 12, 18 time steps
                                valid_steps = valid_steps[
                                    valid_steps.hour.isin([0, 6, 12, 18])
                                ]
                                # Filter to make sure only 00 minutes are included
                                valid_steps = valid_steps[
                                    valid_steps.minute == 0
                                ].values

                                ds_type = kwargs.get("masktype", "rect")

                                if level_coord:
                                    num_levels = var_data[level_coord].shape[0]
                                else:
                                    num_levels = 1

                                attrs = var_data.attrs

                                masks = []
                                for timestamp in valid_steps:
                                    mask = self.get_mask(
                                        num_levels=num_levels,
                                        timestamp=timestamp,
                                        **kwargs,
                                    )
                                    masks.append(mask[None, ...])

                                mask = np.vstack(masks)

                                data = var_data.sel(time=valid_steps).where(mask)

                                setattr(self.ReAnal, "ds", data)

                                encoding = {}
                                for data_var in data.data_vars:
                                    encoding[data_var] = {
                                        "original_shape": data[data_var].shape,
                                        "_FillValue": data[data_var].encoding.get(
                                            "_FillValue", -32767
                                        ),
                                        "dtype": kwargs.get("dtype", np.int16),
                                        "add_offset": data[data_var]
                                        .mean()
                                        .compute()
                                        .values,
                                        "scale_factor": data[data_var]
                                        .std()
                                        .compute()
                                        .values
                                        / kwargs.get("scale_divisor", 2000),
                                    }

                                print(
                                    f"Saving temporary file for {self.uid} to disk...",
                                    flush=True,
                                )
                                # Check if the temporary directory, which is self.filepath + temp, exists
                                temp_dir = os.path.join(self.filepath, "temp")
                                if not os.path.exists(temp_dir):
                                    os.makedirs(temp_dir)

                                data.to_netcdf(
                                    os.path.join(
                                        temp_dir,
                                        f"temp_{np.where(self.timestamps == timestamp)[0][0]}.{data_var}.{self.uid}.{ds_type}.nc",
                                    ),
                                    mode="w",
                                    compute=True,
                                    encoding=encoding,
                                )

                                del data, var_data
                                gc.collect()
                    # Exception for
                    except FileNotFoundError as e:
                        print(f"File not found: {e}", flush=True)
                        traceback.print_exc()
                    except PermissionError as e:
                        print(f"Permission error: {e}", flush=True)
                        traceback.print_exc()
                    except OSError as e:
                        print(f"OS error: {e}", flush=True)
                        traceback.print_exc()
                    except Exception as e:
                        print(f"An unexpected error occurred: {e}", flush=True)
                        traceback.print_exc()
                    # Read in the temporary files and merge them into the final file
                    file_list = os.listdir(os.path.join(self.filepath, "temp"))
                    datavars = [*var_dict.values()]

                    # Filter out files that do not include the temp preix, variable names, and UID
                    for file in file_list.copy():

                        # remove if the file doesn't include any of the data variables
                        checker_list = []
                        for datavar in datavars:
                            checker = f".{datavar}."
                            if checker not in file:
                                checker_list.append(False)
                            else:
                                checker_list.append(True)

                        if not any(checker_list):
                            file_list.remove(file)
                            if kwargs.get("verbose", False):
                                print(
                                    f"Removing {file} from file list because it doesn't include {datavars}"
                                )
                        elif "temp" not in file:
                            file_list.remove(file)
                            if kwargs.get("verbose", False):
                                print(
                                    f"Removing {file} from file list because it doesn't include temp"
                                )
                        elif self.uid not in file:
                            file_list.remove(file)
                            if kwargs.get("verbose", False):
                                print(
                                    f"Removing {file} from file list because it doesn't include {self.uid}"
                                )

                    # If temporary files are present, merge them into the final file
                    if len(file_list) > 0:
                        # Load the temporary files into an xarray dataset
                        temp_ds = xr.open_mfdataset(
                            [
                                os.path.join(self.filepath, "temp", file)
                                for file in file_list
                            ],
                            # concat_dim=time_coord,
                            combine="by_coords",  # "nested",
                            parallel=True,
                        )

                        # Select the non-null data from the temporary dataset
                        temp_ds = temp_ds.where(
                            temp_ds.notnull().compute(), drop=True
                        )  # testing: deleted .compute() from notnull

                        # calculate the encoding for the final dataset
                        encoding = {}
                        for data_var in temp_ds.data_vars:
                            encoding[data_var] = {
                                "original_shape": temp_ds[data_var].shape,
                                "_FillValue": temp_ds[data_var].encoding.get(
                                    "_FillValue", -32767
                                ),
                                "dtype": kwargs.get("dtype", np.int16),
                                "add_offset": temp_ds[data_var].mean().compute().values,
                                "scale_factor": temp_ds[data_var].std().compute().values
                                / kwargs.get("scale_divisor", 2000),
                            }

                        print(f"Writing to file {self.ReAnal.filepath}...")
                        # if the UID dataset exists, merge the new data into it
                        if os.path.exists(self.ReAnal.filepath):
                            print(f"Merging into {self.uid}...")
                            # Load the UID dataset
                            final_ds = xr.open_dataset(self.ReAnal.filepath)

                            # Merge the new data into the final dataset
                            final_ds = xr.merge([final_ds, temp_ds], compat="override")

                            # Save the final dataset to disk
                            final_ds.to_netcdf(
                                self.ReAnal.filepath[:-3] + ".appended.nc",
                                mode="w",
                                compute=True,
                                encoding=encoding,
                            )

                            # Replace the old file
                            os.replace(
                                self.ReAnal.filepath[:-3] + ".appended.nc",
                                self.ReAnal.filepath,
                            )

                        else:
                            # Save the final dataset to disk
                            temp_ds.to_netcdf(
                                self.ReAnal.filepath,
                                mode="w",
                                compute=True,
                                encoding=encoding,
                            )

                        # Close the temporary dataset
                        temp_ds.close()
                        del temp_ds

                        # delete the temporary files
                        for file in file_list:
                            os.remove(os.path.join(self.filepath, "temp", file))
                    else:
                        print("No temporary files found. Skipping...")
                    # for var in var_list:
                    #     ##TODO: check if file exsits and if it does, check if the variable is already in the file
                    #     # if it is, skip it

                    #     print("Years", years, "Variable", var)
                    #     # Load the data for each variable
                    #     var_data, _ = data_collection.retrieve_ds(
                    #         vars=var, years=years, **kwargs
                    #     )

                    #     # get coordinate names
                    #     (
                    #         lat_coord,
                    #         lon_coord,
                    #         time_coord,
                    #         level_coord,
                    #         _,
                    #     ) = get_coord_vars(var_data)

                    #     print(f"Adding {var} to {self.uid}...")

                    #     # Sanitize timestamps because ibtracs includes unsual time steps
                    #     valid_steps = sanitize_timestamps(self.timestamps, var_data)

                    #     print(f"Processing time steps in parallel...")
                    #     # Loop through each time step in parallel
                    #     n_jobs = kwargs.get("n_jobs", jl.cpu_count())
                    #     jl.Parallel(n_jobs=n_jobs, verbose=2)(
                    #         jl.delayed(self.process_timestep)(
                    #             var_data.sel(time=t).copy(),
                    #             **{
                    #                 "lat_coord": lat_coord,
                    #                 "lon_coord": lon_coord,
                    #                 "time_coord": time_coord,
                    #                 "level_coord": level_coord,
                    #             },
                    #             timestamp=t,
                    #             **kwargs,
                    #         )
                    #         for t in valid_steps
                    #     )

                    #     # Garbage collect to free up memory
                    #     gc.collect()

                    #     # Read in the temporary files and merge them into the final file
                    #     file_list = os.listdir(os.path.join(self.filepath, "temp"))

                    #     datavar = list(var_data.data_vars)[0]

                    #     # Filter out files that do not include the temp preix, variable name, and UID
                    #     for file in file_list.copy():
                    #         if "." + datavar + "." not in file:
                    #             file_list.remove(file)
                    #             print(
                    #                 f"Removing {file} from file list because it doesn't include {datavar}"
                    #             )
                    #         elif "temp" not in file:
                    #             file_list.remove(file)
                    #             print(
                    #                 f"Removing {file} from file list because it doesn't include temp"
                    #             )
                    #         elif self.uid not in file:
                    #             file_list.remove(file)
                    #             print(
                    #                 f"Removing {file} from file list because it doesn't include {self.uid}"
                    #             )

                    #     print("after removal: ", file_list)

                    #     # If temporary files are present, merge them into the final file
                    #     if len(file_list) > 0:
                    #         # Load the temporary files into an xarray dataset
                    #         temp_ds = xr.open_mfdataset(
                    #             [
                    #                 os.path.join(self.filepath, "temp", file)
                    #                 for file in file_list
                    #             ],
                    #             concat_dim=time_coord,
                    #             combine="nested",
                    #             parallel=True,
                    #         )

                    #         # Select the non-null data from the temporary dataset
                    #         temp_ds = temp_ds.where(
                    #             temp_ds.notnull().compute(), drop=True
                    #         )

                    #         # calculate the encoding for the final dataset
                    #         encoding = {}
                    #         for data_var in temp_ds.data_vars:
                    #             encoding[data_var] = {
                    #                 "original_shape": temp_ds[data_var].shape,
                    #                 "_FillValue": temp_ds[data_var].encoding.get(
                    #                     "_FillValue", -32767
                    #                 ),
                    #                 "dtype": kwargs.get("dtype", np.int16),
                    #                 "add_offset": temp_ds[data_var]
                    #                 .mean()
                    #                 .compute()
                    #                 .values,
                    #                 "scale_factor": temp_ds[data_var]
                    #                 .std()
                    #                 .compute()
                    #                 .values
                    #                 / kwargs.get("scale_divisor", 2000),
                    #             }

                    #         # TODO 2024-07-18: Fix path handling

                    #         # if the UID dataset exists, merge the new data into it
                    #         if os.path.exists(self.ReAnal.filepath):
                    #             print(f"Merging {var} into {self.uid}...")
                    #             # Load the UID dataset
                    #             final_ds = xr.open_dataset(self.ReAnal.filepath)

                    #             # Merge the new data into the final dataset
                    #             final_ds = xr.merge([final_ds, temp_ds])

                    #             # Save the final dataset to disk
                    #             final_ds.to_netcdf(
                    #                 self.ReAnal.filepath[:-3] + ".appended.nc",
                    #                 mode="w",
                    #                 compute=True,
                    #                 encoding=encoding,
                    #             )

                    #             # Replace the old file
                    #             os.replace(
                    #                 self.ReAnal.filepath[:-3] + ".appended.nc",
                    #                 self.ReAnal.filepath,
                    #             )

                    #             # # Workaround to rewrite file with appended file
                    #             # # Overwrite the original file with the appended file
                    #             # subprocess.run(
                    #             #     [
                    #             #         "mv",
                    #             #         "-f",
                    #             #         f"{self.filepath}{self.uid}.{kwargs.get('masktype', 'rect')}.appended.nc",
                    #             #         f"{self.filepath}{self.uid}.{kwargs.get('masktype', 'rect')}.nc",
                    #             #     ]
                    #             # )
                    #         else:
                    #             # Save the final dataset to disk
                    #             temp_ds.to_netcdf(
                    #                 self.ReAnal.filepath,
                    #                 mode="w",
                    #                 compute=True,
                    #                 encoding=encoding,
                    #             )

                    #         # Close the temporary dataset
                    #         temp_ds.close()
                    #         del temp_ds

                    #         # delete the temporary files
                    #         for file in file_list:
                    #             os.remove(os.path.join(self.filepath, "temp", file))

                    #         # Garbage collect to free up memory
                    #         gc.collect()
            elif str(data_collection) == "TCBench_AI_DataCollection":
                # Check if the target dataset exists. If it does, and the
                # overwrite argument isn't in the kwargs, skip the processing
                if os.path.exists(
                    os.path.join(
                        self.filepath, f"{self.uid}.AI.{data_collection.ai_model}.nc"
                    )
                ) and (not kwargs.get("overwrite", False)):
                    print(
                        f"{self.uid} already processed for {data_collection.ai_model}. Skipping...",
                        flush=True,
                    )
                    return

                # AI models only have 3 hourly data available - filter the track timestamps
                # to only include 3 hourly data
                ds = data_collection.retrieve_ds(self.timestamps, **kwargs)

                (
                    lat_coord,
                    lon_coord,
                    time_coord,
                    level_coord,
                    leadtime_coord,
                ) = get_coord_vars(ds)

                vars_to_keep = kwargs.get(
                    "vars_to_keep",
                    {
                        "u10": [
                            None,
                            {
                                "units": "m s**-1",
                                "long_name": "10 metre U wind component",
                            },
                        ],
                        "v10": [
                            None,
                            {
                                "units": "m s**-1",
                                "long_name": "10 metre V wind component",
                            },
                        ],
                        "msl": [
                            None,
                            {
                                "units": "Pa",
                                "long_name": "Mean sea level pressure",
                                "standard_name": "air_pressure_at_sea_level",
                            },
                        ],
                        "z": [
                            500,
                            {
                                "units": "m**2 s**-2",
                                "long_name": "Geopotential",
                                "standard_name": "geopotential",
                            },
                        ],
                        "t": [
                            850,
                            {
                                "units": "K",
                                "long_name": "Temperature",
                                "standard_name": "air_temperature",
                            },
                        ],
                    },
                )

                da_list = []
                for var, [level, attrs] in vars_to_keep.items():
                    if level is not None:
                        # if isinstance(level, int) or isinstance(level, float):
                        #     level = [level]
                        # if len(level) == 1:
                        #     assert (
                        #         type(level[0]) == ds[level_coord].dtype
                        #         and level in ds[level_coord].values
                        #     ), f"Invalid level {level}. Make sure the correct level type and value are selected."
                        # elif len(level) > 1:
                        #     assert all(
                        #         [
                        #             (
                        #                 type(lv) == ds[level_coord].dtype
                        #                 and lv in ds[level_coord].values
                        #             )
                        #             for lv in level
                        #         ]
                        #     ), f"Invalid levels {level}. Make sure the correct level type and values are selected."

                        # TODO: Add support for multiple levels
                        da = ds[var].sel({level_coord: level})

                        # For each level, add a variable with the level appended to the variable name
                        # if len(level) > 1:
                        #     for lv in level:
                        #         da = ds[var].sel({level_coord: lv})
                        #         da = da.rename(f"{var}{lv}")
                        #         da = da.drop_vars(level_coord)
                        #         da.attrs = attrs
                        #         da_list.append(da)
                        # else:
                        da = ds[var].sel({level_coord: level})
                        da = da.rename(f"{var}{level}")
                        da = da.drop_vars(level_coord)
                        da.attrs = attrs
                    else:
                        da = ds[var]
                    da_list.append(da)
                temp_ds = xr.merge(da_list).chunk()

                # limit the number of leadtimes
                num_days = kwargs.get("num_days", 7)
                steps_per_day = kwargs.get("steps_per_day", 4)

                temp_ds = temp_ds.isel(
                    {
                        leadtime_coord: [
                            0,
                            1,
                            2,
                            *np.arange(3, num_days * steps_per_day + 1, 4),
                        ]
                    }
                )

                # num_levels = ds[level_coord].size
                num_leadtimes = temp_ds[leadtime_coord].size
                timesteps = temp_ds[time_coord].values

                # Get the mask for the track. For AI models, we have a set mask of +-30° in
                # latitude and longitude for 7 days. For 5 days, we set it to +-25°
                circum_points = kwargs.get("circum_points", 30 * 4)

                mask = self.get_mask_series(
                    timesteps=timesteps,
                    masktype="rect",
                    circum_points=circum_points,
                    # num_levels=num_levels,
                    # num_leadtimes=num_leadtimes,
                )

                # Create a mask data array
                mask_da = xr.DataArray(
                    mask,
                    dims=[time_coord, lat_coord, lon_coord],
                    coords={
                        time_coord: timesteps,
                        lat_coord: ds[lat_coord],
                        lon_coord: ds[lon_coord],
                    },
                )

                filtered_ds = temp_ds.where(mask_da, drop=True).chunk()
                filtered_ds.attrs.update({"AI_model": data_collection.ai_model})

                # # filter out nan values
                # filtered_ds = filtered_ds.where(~filtered_ds.isnull(), drop=True)

                if kwargs.get("reencode", False):
                    # calculate the encoding for the final dataset
                    print("encoding...", flush=True)
                    encoding = {}
                    for data_var in filtered_ds.data_vars:
                        encoding[data_var] = {
                            "original_shape": filtered_ds[data_var].shape,
                            "_FillValue": filtered_ds[data_var].encoding.get(
                                "_FillValue", -32767
                            ),
                            "dtype": kwargs.get("dtype", np.int16),
                            "add_offset": filtered_ds[data_var].mean().compute().values,
                            "scale_factor": filtered_ds[data_var].std().compute().values
                            / kwargs.get("scale_divisor", 1000),
                        }

                # Save the final dataset to disk
                print(f"Saving {str(self)} to {self.filepath}...", flush=True)
                filtered_ds.to_netcdf(
                    os.path.join(
                        self.filepath, f"{self.uid}.AI.{data_collection.ai_model}.nc"
                    ),
                    mode="w",
                    compute=True,
                    # encoding=encoding,
                )

                return
        except:
            print(
                f"Issue when processing {str(data_collection)} \nfor {str(self)}. Skipping..."
            )

            # Check if the error log exists
            log_path = os.path.join(self.filepath, "error.log")
            if not os.path.exists(log_path):
                # If it doesn't exist, create it
                open(log_path, "w").close()

            # Append the timestamp and storm to the error log
            with open(log_path, "a") as f:
                # retrieve the timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"Time: {timestamp} Storm: {self}\n")

            print(traceback.format_exc())

    def process_timestep(self, data, **kwargs):
        """
        Function to add a dataset to the track object. It expects
        a dataset served by the data collection object.


        The function first checks if a dataset does not exist in the
        object, and if it does not, the function checks if a dataset
        exists on disk. If it does, the function loads the dataset and
        then checks to see if the input data variables are already present
        in the dataset. If they are, the function skips the variable.
        If they are not, the variable is added to a list of variables to
        process.

        If the dataset exists in the object, the function checks to see
        whether the input data variables are already present in the
        dataset. If they are, the function skips the variable. If they
        are not, the variable is added to a list of variables to process.

        If the list of variables to process is not empty, the function
        loops through the list and adds the variables to the dataset.
        The dataset is then stored in the object and saved to disk.

        Parameters
        ----------
        data : xr.Dataset, required
            Dataset containing the data to add to the track object
            Coords:
                'lat' or 'Lat' must be in the latitude dimension name
                'lon' or 'Lon' must be in the longitude dimension
                'time', 'Time', or 't' must be found as the time dimension
                'lev', 'Lev', 'pressure', or 'Pressure' must be in the level dimension if the dataset is multilevel

        Returns
        -------
        None

        """

        # Determine if the data will be rectanglular (rect) or radial (rad), default rect
        ds_type = kwargs.get("masktype", "rect")

        # Get the coordinate names
        lon_coord = kwargs.get("lon_coord", None)
        lat_coord = kwargs.get("lat_coord", None)
        time_coord = kwargs.get("time_coord", None)
        level_coord = kwargs.get("level_coord", None)

        # Determine the number of levels
        if level_coord:
            num_levels = data[level_coord].shape[0]
        else:
            num_levels = 1

        # Get the timestamp from the kwargs
        timestamp = kwargs.get("timestamp", None)
        assert timestamp is not None, "Fatal error: timestamp not found in kwargs"

        var_list = None

        var_list = list(data.data_vars)

        if var_list is not None:
            data = data[var_list]

            attrs = data.attrs

            # Retrieve regridder if necessary
            regridder = None  # get_regrider(dataset=data, **kwargs)

            if regridder is not None:
                data = regridder(data)

            level_coord = kwargs.get("level_coord", None)
            if level_coord:
                num_levels = data[level_coord].shape[0]

            mask = self.get_mask(num_levels=num_levels, **kwargs)

            data = data.where(mask)
            data.attrs = attrs

            setattr(self.ReAnal, "ds", data)

            encoding = {}
            for data_var in data.data_vars:
                encoding[data_var] = {
                    "original_shape": data[data_var].shape,
                    "_FillValue": data[data_var].encoding.get("_FillValue", -32767),
                    "dtype": kwargs.get("dtype", np.int16),
                    "add_offset": data[data_var].mean().compute().values,
                    "scale_factor": data[data_var].std().compute().values
                    / kwargs.get("scale_divisor", 2000),
                }

            print(f"Saving temporary file for {self.uid} to disk...", flush=True)
            # Check if the temporary directory, which is self.filepath + temp, exists
            temp_dir = os.path.join(self.filepath, "temp")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            data.to_netcdf(
                os.path.join(
                    temp_dir,
                    f"temp_{np.where(self.timestamps == timestamp)[0][0]}.{data_var}.{self.uid}.{ds_type}.nc",
                ),
                mode="w",
                compute=True,
                encoding=encoding,
            )

    def load_data(self, **kwargs):
        """
        Function to load the data from storage

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        ds_type = kwargs.get("ds_type", "rect")

        # Check if the dataset doesn't exist in the object
        if not hasattr(self, f"{ds_type}_ds"):
            # Check if the dataset exists on disk
            if os.path.exists(self.filepath + f"{self.uid}.{ds_type}.nc"):
                # If it does, load the dataset
                # print("Loading dataset...")
                setattr(
                    self,
                    f"{ds_type}_ds",
                    xr.open_dataset(self.filepath + f"{self.uid}.{ds_type}.nc"),
                )
            else:
                if kwargs.get("verbose", False) or kwargs.get("debug", False):
                    print(f"Dataset {self.uid}.{ds_type}.nc not found on disk")
                    print(
                        'You can create the dataset by running the "process_data_collection" method'
                        " with a data collection object as an argument."
                    )

    def get_ground_truth(self, **kwargs):
        ground_truth = kwargs.get("ground_truth", ["wind", "pressure"])

        # check that the ground truth is a list, and if it is
        # that the list only includes elements from "WIND", "PRES",
        # and "TRACK"
        assert isinstance(
            ground_truth, list
        ), f"Invalid ground truth type {type(ground_truth)}. Expected list"
        assert all(
            [gt in ["wind", "pressure", "track"] for gt in ground_truth]
        ), f"Invalid ground truth elements {ground_truth}. Expected ['wind', 'pressure', 'track']"

        gt_arrays = []
        for gt in ground_truth:
            if gt == "wind":
                gt_arrays.append(self.wind)
            elif gt == "pressure":
                gt_arrays.append(self.pressure)
            elif gt == "track":
                gt_arrays.append(self.track[:, 0])
                gt_arrays.append(self.track[:, 1])

        gt_array = np.stack(gt_arrays, axis=1)
        gt_array = np.char.strip(gt_array.astype(str))

        # load the timestamps so we know which steps we're
        # interested in postprocessing
        timestamps = self.timestamps

        # if any row has a "" value, remove it
        timestamps = timestamps[~np.any(gt_array == "", axis=1)]
        gt_array = gt_array[~np.any(gt_array == "", axis=1)].astype(float)

        return gt_array, timestamps

    def historical_displacement(self, **kwargs):
        """
        Function to calculate the historical displacement of a storm
        for a given leadtime.

        Parameters
        ----------
        leadtime : int, required
            Leadtime in hours for which to calculate the historical displacement
            of the storm.

        Returns
        -------
        historical_displacement : np.ndarray
            Array containing the historical displacement of the storm
            for the given leadtime, in 100s of kilometers.
        lat_delta : np.ndarray
            Array containing the latitude delta of the storm
            for the given leadtime.
        lon_delta : np.ndarray
            Array containing the longitude delta of the storm
            for the given leadtime.

        """

        # Get the ground truth data
        positions, timestamps = self.get_ground_truth(ground_truth=["track"], **kwargs)
        timestamps = pd.to_datetime(timestamps)

        # Get the leadtime
        leadtime = kwargs.get("leadtime", 24)
        assert isinstance(
            leadtime, int
        ), f"Invalid leadtime type {type(leadtime)}. Expected int"

        # Filter out non-6-hourly timestamps
        valid_timestamps = timestamps[timestamps.hour.isin([0, 6, 12, 18])]

        target_timestamps = valid_timestamps + np.timedelta64(leadtime, "h")

        # Get the indices of the target timestamps that are valid
        valid_indices = np.where(np.isin(target_timestamps, timestamps))[0]

        # Get the initial/end timestamps that have a valid index associated with them
        initial_timestamps = valid_timestamps[valid_indices]
        end_timestamps = target_timestamps[valid_indices]

        # Get the initial/end positions
        initial_positions = positions[np.isin(timestamps, initial_timestamps)]
        end_positions = positions[np.isin(timestamps, end_timestamps)]

        distances = list_to_list_haversine(initial_positions, end_positions)

        if not kwargs.get("non_absolute", False):
            lat_delta = np.abs(end_positions[:, 0] - initial_positions[:, 0])[:, None]
            lon_delta = np.abs(end_positions[:, 1] - initial_positions[:, 1])[:, None]
        else:
            lat_delta = end_positions[:, 0] - initial_positions[:, 0]
            lon_delta = end_positions[:, 1] - initial_positions[:, 1]

        return distances, lat_delta, lon_delta

    def plot_track(self, **kwargs):
        """

        Parameters
        ----------
        figsize : tuple, optional
            Figure size.
            The default is (6, 2). Note that the tuple is fed inverted to the
            matplotlib figure function, i.e., default is fed as (2,6)

        height : float, optional
            Height at which to plot the track and points.
            The default is 0.0.

        point_size : int, optional
            Size of the points. The default is 150.

        step : int, optional
            Timesteps to skip when plotting points. The default is 6.

        radius : int, optional
            The number of degrees east/west/north/south of the track minima/maxima
            to plot. The default is 20.

        background_color : tuple, optional
            Background color. The default is matplotlibs tab:cyan with 0.25 alpha.

        land_color : tuple, optional
            Land color. The default is matplotlibs tab:green with 0.05 alpha.

        coastline_color: str, optional
            Coastline color. The default is "grey".

        view_angles : tuple, optional
            View angles. The default is (25, -90).

        track_color : str, optional
            Track color. The default is "black".

        point_color : tuple, optional
            Point color. The default is matplotlibs tab:orange.



        Sources:
        https://stackoverflow.com/questions/30223161/how-to-increase-the-size-of-an-axis-stretch-in-a-3d-plot
        """

        save_path = kwargs.get("save_path", None)
        width, height = kwargs.get("figsize", (10, 2.5))
        point_height = kwargs.get("point_height", 0)
        point_size = kwargs.get("point_size", 300)
        step = kwargs.get("step", 6)
        radius = kwargs.get("radius", 20)
        background_color = kwargs.get(
            "background_color",
            (*mpl.colors.to_rgb(mpl.colors.TABLEAU_COLORS["tab:cyan"]), 0.25),
        )
        land_color = kwargs.get(
            "land_color",
            (*mpl.colors.to_rgb(mpl.colors.TABLEAU_COLORS["tab:green"]), 0.05),
        )
        coastline_color = kwargs.get("coastline_color", "grey")
        view_angles = kwargs.get("view_angles", (25, -90))
        track_color = kwargs.get("track_color", "black")
        point_color = kwargs.get(
            "point_color", mpl.colors.to_rgb(mpl.colors.TABLEAU_COLORS["tab:orange"])
        )
        convert_to_180 = kwargs.get("convert_to_180", True)
        if convert_to_180:
            self.track[:, 1][self.track[:, 1] > 180] = (
                self.track[:, 1][self.track[:, 1] > 180] - 360
            )

        xlims = (self.track[:, 1].min() - radius, self.track[:, 1].max() + radius)
        ylims = (self.track[:, 0].min() - radius, self.track[:, 0].max() + radius)

        fig = plt.figure(figsize=(width, height), dpi=150)
        ax3d = fig.add_axes([0, 0, width, height], projection="3d")

        # Make an axes that we can use for mapping the data in 2d.
        proj_ax = plt.figure().add_axes(
            [0, 0, width, height], projection=ccrs.PlateCarree()
        )
        proj_ax.autoscale_view()

        # Add the geodata to the plot
        concat = lambda iterable: list(itertools.chain.from_iterable(iterable))

        target_projection = proj_ax.projection

        feature = cartopy.feature.NaturalEarthFeature("physical", "land", "110m")
        geoms = feature.geometries()

        proj_ax.set_xlim(xlims)
        proj_ax.set_ylim(ylims)

        # Use the convenience (private) method to get the extent as a shapely geometry.
        boundary = proj_ax._get_extent_geom()

        # Transform the geometries from PlateCarree into the desired projection.
        geoms = [
            target_projection.project_geometry(geom, feature.crs) for geom in geoms
        ]
        # Clip the geometries based on the extent of the map (because mpl3d can't do it for us)
        geoms = [boundary.intersection(geom) for geom in geoms]

        # Convert the geometries to paths so we can use them in matplotlib.
        paths = concat(geos_to_path(geom) for geom in geoms)
        polys = concat(path.to_polygons() for path in paths)
        lc = PolyCollection(
            polys,
            edgecolor=coastline_color,
            facecolor=land_color,
            closed=True,
        )
        ax3d.add_collection3d(lc, zs=0)

        plt.close(proj_ax.figure)

        ax3d.view_init(*view_angles)
        ax3d.set_xlim(xlims)
        ax3d.set_ylim(ylims)
        # ax3d.set_xlim(*proj_ax.get_xlim())
        # ax3d.set_ylim(*proj_ax.get_ylim())
        ax3d.set_zlim(0, 0.5)
        ax3d.set_box_aspect((width, height * 2, 0.5))
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.xaxis.pane.set_edgecolor((1, 1, 1, 0))
        ax3d.yaxis.pane.set_edgecolor((1, 1, 1, 0))
        ax3d.zaxis.set_pane_color(background_color)

        ax3d.grid(False)
        ax3d.zaxis.line.set_lw(0.0)
        ax3d.set_zticks([])

        ax3d.scatter3D(
            self.track[::step, 1],
            self.track[::step, 0],
            np.ones_like(self.track[::step, 0]) * point_height,
            c=point_color,
            s=point_size,
            marker=hurricane_symbol(),
            depthshade=0,
        )

        ax3d.plot3D(
            self.track[:, 1],
            self.track[:, 0],
            np.ones_like(self.track[:, 0]) * point_height,
            c=track_color,
            # alpha=0.5,
        )

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()


# Class representing the timeseries object within the TCBench framework
# This class becomes a subobject of the TC_track object
class timeseries:
    __valid_kwargs = [
        "timeseries_source",
    ]

    __default_vals = {
        "timeseries_source": "SHIPS_netcdfs",
    }

    def __str__(self):
        return f"Timeseries object for {self.__parent.uid}"

    def __init__(self, parent, **kwargs):
        self.__parent = parent
        source = kwargs.get("timeseries_source", "SHIPS_netcdfs")
        self.filepath = os.path.join(self.__parent.datadir_path, source)

    def load(self, **kwargs):
        # check if the UID or the ALT_ID is in the ATCF format
        if (
            self.__parent.uid[0:2].isalpha()
            and self.__parent.uid[0:2].isupper()
            and self.__parent.uid[2:6].isdigit()
        ):
            # if it is, save the ID into a variable
            atcf_id = self.__parent.uid
        elif (
            self.__parent.ALT_ID[0:2].isalpha()
            and self.__parent.ALT_ID[0:2].isupper()
            and self.__parent.ALT_ID[2:6].isdigit()
        ):
            # if it is, save the ID into a variable
            atcf_id = self.__parent.ALT_ID
        else:
            raise ValueError(
                f"Invalid ATCF ID format for storm {self.__parent.uid} with ALT_ID {self.__parent.ALT_ID}"
            )

        # extract the data folder from the kwargs
        data_dir = kwargs.get(
            "timeseries_dir",
            os.path.join(self.__parent.datadir_path, "SHIPS_netcdfs"),
        )

        # Check if the dataset doesn't exist in the object
        if not hasattr(self, "ds") or kwargs.get("reload", False):
            # Check if the dataset exists on disk
            ds_path = os.path.join(data_dir, f"{atcf_id}.nc")

            if os.path.exists(ds_path):
                # If it does, load the dataset
                # print("Loading dataset...")
                setattr(
                    self,
                    "ds",
                    xr.open_dataset(ds_path),
                )
            else:
                # Raise file not found error
                print(
                    f"Time series storm data for storm {self.__parent.uid} with ATCF_ID {atcf_id} not found in timeseries directory {data_dir}"
                )
                raise FileNotFoundError(f"{ds_path} not found")

    def serve(self, **kwargs):
        # Check if the dataset doesn't exist in the object
        if not hasattr(self, "ds"):
            # if it doesnt, try loading it
            self.load(**kwargs)
        pass

        leadtime_min = kwargs.get("leadtime_min", 0)
        leadtime_max = kwargs.get("leadtime_max", 0)

        coord_array = np.array(self.ds.coords)
        try:
            time_coord = False
            for idx, coord in enumerate(coord_array):
                if not time_coord:
                    time_coord = (
                        coord_array[idx]
                        if np.any(
                            [
                                valid_name == coord
                                for valid_name in constants.valid_coords["time"]
                            ]
                        )
                        else False
                    )
            if not time_coord:
                raise ValueError("Time coordinate not found in dataset")
        except:
            raise ValueError("Error finding Time coordinate in dataset")

        try:
            ldt_coord = False
            for idx, coord in enumerate(coord_array):
                if not ldt_coord:
                    ldt_coord = (
                        coord_array[idx]
                        if np.any(
                            [
                                valid_name in coord
                                for valid_name in constants.valid_coords["leadtime"]
                            ]
                        )
                        else False
                    )
            if not ldt_coord:
                raise ValueError("Leadtime coordinate not found in dataset")
        except:
            raise ValueError("Error finding Leadtime coordinate in dataset")

        other_coords = coord_array[~np.isin(coord_array, time_coord)]

        ds = self.ds.copy()
        if other_coords is not None:
            for coord in other_coords:
                if coord == ldt_coord:
                    ds = ds.sel({ldt_coord: slice(leadtime_min, leadtime_max)})
                ds = ds.where(~ds[coord].isnull(), drop=True)
                if kwargs.get("verbose", False):
                    print(f"Data shape: {ds}")

        variables = kwargs.get(
            "variables",
            constants.default_ships_vars,
        )

        if "LHRD" in variables:
            ds["LHRD"] = ds["SHDC"] * np.sin(np.radians(ds["lat"].astype(float)))

        if "VSHR" in variables:
            ds["VSHR"] = ds["VMAX"] * ds["SHDC"]

        try:
            ds = ds[variables]
        except:
            print(f"One or more variables missing from storm {self.uid}. Skipping...")
            return None, None

        data = None
        for var in variables:
            # Select only columns that have at least one non_nan
            temp_data = (
                ds[var].where(np.any(~ds[var].isnull(), axis=0), drop=True).values
            )
            if len(temp_data) == 0:
                continue
            if temp_data.ndim == 1:
                temp_data = temp_data[np.newaxis, ...]

            if data is None:
                data = temp_data
            else:
                data = np.hstack([data, temp_data])
        data = data.astype(float)

        # Boolean mask to remove nans
        bool_mask = np.any(np.isnan(data), axis=1)
        data = data[~bool_mask]

        timestamps = ds[time_coord].values[~bool_mask]

        if kwargs.get("verbose", False):
            print(
                f"A total of {len(variables)} variables were loaded for storm ",
                f"{self.uid} with shape {data.shape}. {bool_mask.sum()} nan rows",
                f" were removed.",
                sep="",
                flush=True,
            )

        return data, timestamps


# Class organizing the AI output data corresponding to a track object
# This class becomes a subobject of the TC_track object
class AI:
    __valid_kwargs = ["filepath", "ai_model", "cache_dir"]

    __default_vals = {"filepath": None, "ai_model": "panguweather", "cache_dir": None}

    def __str__(self):
        return f"AI data object for {self.__parent.uid}"

    def __init__(self, parent, **kwargs):
        self.__parent = parent
        self.model = kwargs.get("ai_model", "panguweather")

        if kwargs.get("filepath", None) is None:
            self.filepath = os.path.join(
                self.__parent.datadir_path,
                str(self.__parent.season),
                f"{self.__parent.uid}.AI.{self.model}.nc",
            )
        else:
            self.filepath = kwargs.get("filepath")

        if kwargs.get("cache_dir", None) is None:
            self.cache_dir = os.path.join(self.__parent.datadir_path, "cache")
        else:
            self.cache_dir = kwargs.get("cache_dir")

    def load(self, **kwargs):
        """
        Function to load the AI data from storage

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        if kwargs.get("verbose", False):
            print(f"Loading {self.model} forecast data for {self.__parent.uid}...")

        # Check if the dataset doesn't exist in the object
        if not hasattr(self, "ds"):
            if os.path.exists(self.filepath):
                # If it does, load the dataset
                # print("Loading dataset...")
                setattr(self, "ds", xr.open_dataset(self.filepath))
            else:
                if kwargs.get("verbose", False) or kwargs.get("debug", False):
                    print(
                        f"Dataset {self.__parent.uid}.AI.{self.model}.nc not found on disk"
                    )
                    print(
                        'You can create the dataset by running the "process_data_collection" method'
                        " with an AI data collection object as an argument."
                    )

    def serve(self, **kwargs):
        try:
            # check if the cache dir exists
            cache_dtype = kwargs.get("cache_dtype", float)
            # cache_dir = kwargs.get(
            #     "cache_dir", os.path.join(self.datadir_path, "cache")
            # )
            load_from_cache = kwargs.get("use_cached", True)
            # ai_model = kwargs.get("ai_model", "panguweather")

            if os.path.exists(self.cache_dir) and load_from_cache:
                try:
                    # check if the cache files exist. If they don't exist, continue
                    if not all(
                        [
                            os.path.exists(
                                os.path.join(
                                    self.cache_dir, f"{self.__parent.uid}_{file}.npy"
                                )
                            )
                            for file in ["X", "Y", "t", "leads"]
                        ]
                    ):
                        raise FileNotFoundError(
                            "Cache files not found. Rebuilding cache..."
                        )

                    # load the files into dask arrays
                    X = da.from_array(
                        np.load(
                            os.path.join(self.cache_dir, f"{self.__parent.uid}_X.npy"),
                            mmap_mode="r",
                        ),
                        chunks=(kwargs.get("chunk_size", 256), 5, 241, 241),
                    )
                    Y = da.from_array(
                        np.load(
                            os.path.join(self.cache_dir, f"{self.__parent.uid}_Y.npy"),
                            mmap_mode="r",
                        )
                    )
                    t = da.from_array(
                        np.load(
                            os.path.join(self.cache_dir, f"{self.__parent.uid}_t.npy"),
                            mmap_mode="r",
                        )
                    )
                    leads = da.from_array(
                        np.load(
                            os.path.join(
                                self.cache_dir, f"{self.__parent.uid}_leads.npy"
                            ),
                            mmap_mode="r",
                        )
                    )
                    if kwargs.get("verbose", False):
                        print(
                            f"Succesfully loaded cache files for {self.__parent.uid} from cache...",
                            flush=True,
                        )

                    return X, Y, t, leads
                except Exception as e:
                    if kwargs.get("debug", False):
                        print(
                            f"Error loading cache files for {self.__parent.uid}. Rebuilding cache..."
                        )
                        if kwargs.get("verbose", False):
                            print(e)
            else:
                if not os.path.exists(self.cache_dir):
                    os.makedirs(self.cache_dir)

            if kwargs.get("verbose", False) and kwargs.get("use_cached", True):
                print(
                    f"used_cached disabled or cache files not found for {self.__parent.uid}. Rebuilding cache..."
                )

            # Check if the dataset doesn't exist in the object

            if not hasattr(self, "ds"):
                # if it doesnt, try loading it
                self.load(**kwargs)

            assert hasattr(
                self, "ds"
            ), f"AI dataset not found for {self.__parent.uid} with model {self.model}"

            (
                _,
                _,
                time_coord,
                _,
                leadtime_coord,
            ) = get_coord_vars(self.ds)

            # Get ground truth and valid timestamps
            gt, stamps = self.__parent.get_ground_truth(**kwargs)

            ref_gt = gt.copy()
            ref_stamps = stamps.copy()

            valid_stamps = sanitize_timestamps(
                stamps,
                self.ds,
                time_coord,  # leadtime_coord=leadtime_coord
            )

            gt = gt[np.isin(stamps, valid_stamps)]
            stamps = stamps[np.isin(stamps, valid_stamps)]

            if len(gt) == 0:
                if kwargs.get("verbose", False) or kwargs.get("debug", False):
                    print(
                        f"No valid timestamps found for {self.__parent.uid} with model {self.model}"
                    )
                return

            # Get the AI data
            inputs = None
            targets = None
            outstamps = None
            out_leads = None
            for stamp in stamps:
                temp_ds = self.ds.sel({time_coord: stamp})
                temp_ds = temp_ds.where(~temp_ds.isnull(), drop=True)

                timedeltas = temp_ds[leadtime_coord].values.astype("timedelta64[h]")
                leadtimes = stamp + timedeltas

                # make a boolean index to see what leadtime data we have a
                # truth value for
                bool_idx = np.isin(leadtimes, ref_stamps)

                out_data = temp_ds.isel({leadtime_coord: bool_idx}).to_array().values
                out_data = np.moveaxis(out_data, 0, 1)
                out_targets = ref_gt[np.isin(ref_stamps, leadtimes[bool_idx])]
                # base_targets = gt[stamps == stamp]
                temp_stamps = leadtimes[bool_idx]
                temp_leads = temp_ds.isel({leadtime_coord: bool_idx})[
                    leadtime_coord
                ].values

                if outstamps is None:
                    try:
                        ##TODO: The following check appears to be an issue
                        ## with the way the data is being filtered when processing
                        ## the storm fields. My guess is that this is due to the
                        ## crossing of the 0° longitude line. This should be fixed
                        ## process_data_collection method.

                        if out_data.shape[-2:] != (
                            241,
                            241,
                        ):
                            raise ValueError(
                                f"Invalid shape for stamp {stamp}... Skipping..."
                            )
                        else:
                            inputs = out_data
                            targets = out_targets
                            outstamps = temp_stamps  # leadtimes[bool_idx]
                            # base_stamps = np.full_like(outstamps, stamp)
                            out_leads = temp_leads

                    except ValueError as e:
                        if kwargs.get("verbose", False):
                            print(f"Problem processing {stamp}... Skipping...")
                            print(e)
                else:
                    try:
                        if out_data.shape[-2:] != (241, 241):
                            raise ValueError(
                                f"Invalid shape for stamp {stamp}... Skipping..."
                            )
                        else:
                            inputs = np.vstack([inputs, out_data])
                            targets = np.vstack([targets, out_targets])
                            outstamps = np.hstack([outstamps, temp_stamps])
                            out_leads = np.hstack([out_leads, temp_leads])

                        # outstamps = np.hstack([outstamps, leadtimes[bool_idx]])
                        # out_leads = np.hstack(
                        #     [out_leads, temp_ds[leadtime_coord].values[bool_idx]]
                        # )

                    except ValueError as e:
                        if kwargs.get("verbose", False):
                            print(f"Problem processing {stamp}... Skipping...")
                            print(e)

            # save if the data is not empty
            if inputs is not None:
                # save the data to the cache
                np.save(
                    os.path.join(self.cache_dir, f"{self.__parent.uid}_X.npy"), inputs
                )
                np.save(
                    os.path.join(self.cache_dir, f"{self.__parent.uid}_Y.npy"), targets
                )
                np.save(
                    os.path.join(self.cache_dir, f"{self.__parent.uid}_t.npy"),
                    outstamps,
                )
                np.save(
                    os.path.join(self.cache_dir, f"{self.__parent.uid}_leads.npy"),
                    out_leads,
                )

                # load saved files into dask arrays
                X = da.from_array(
                    np.load(
                        os.path.join(self.cache_dir, f"{self.__parent.uid}_X.npy"),
                        mmap_mode="r",
                    ),
                    chunks=(kwargs.get("chunk_size", 256), 5, 241, 241),
                )
                Y = da.from_array(
                    np.load(
                        os.path.join(self.cache_dir, f"{self.__parent.uid}_Y.npy"),
                        mmap_mode="r",
                    )
                )
                t = da.from_array(
                    np.load(
                        os.path.join(self.cache_dir, f"{self.__parent.uid}_t.npy"),
                        mmap_mode="r",
                    )
                )
                leads = da.from_array(
                    np.load(
                        os.path.join(self.cache_dir, f"{self.__parent.uid}_leads.npy"),
                        mmap_mode="r",
                    )
                )
                return X, Y, t, leads
            else:
                # save empty arrays to the cache
                np.save(
                    os.path.join(self.cache_dir, f"{self.__parent.uid}_X.npy"),
                    np.array([]),
                )
                np.save(
                    os.path.join(self.cache_dir, f"{self.__parent.uid}_Y.npy"),
                    np.array([]),
                )
                np.save(
                    os.path.join(self.cache_dir, f"{self.__parent.uid}_t.npy"),
                    np.array([]),
                )
                np.save(
                    os.path.join(self.cache_dir, f"{self.__parent.uid}_leads.npy"),
                    np.array([]),
                )
                return np.array([]), np.array([]), np.array([]), np.array([])

        except Exception as e:
            if kwargs.get("verbose", False):
                print(
                    f"Error serving AI data for {self.__parent.uid} with model {self.model}",
                    flush=True,
                )
                print(e)
            return None

    def animate_var(self, var, **kwargs):
        figsize = kwargs.get("figsize", (5, 6))
        dpi = kwargs.get("dpi", 150)
        save_path = kwargs.get(
            "save_path",
            f"/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/dev/results/{self.__parent.uid}.{self.model}_{var}_animation.gif",
        )

        # Check if the dataset doesn't exist in the object
        if not hasattr(self, "ds"):
            if kwargs.get("verbose", False):
                print("Data not yet loaded - trying to load it now...")
            self.load(**kwargs)

        # Assert that the variable exists in the object data
        assert var in self.ds.data_vars, f"Variable {var} not found in dataset"

        # get sanitized timestamps
        valid_steps = sanitize_timestamps(self.__parent.timestamps, self.ds)

        # Create a projection with Cartopy
        projection = ccrs.PlateCarree()

        fig, ax = plt.subplots(
            figsize=figsize, dpi=dpi, subplot_kw={"projection": projection}
        )

        minn, maxx = (
            self.ds[var].min().values,
            self.ds[var].max().values,
        )

        data = self.ds[var]

        _, _, time_coord, _, leadtime_coord = get_coord_vars(data)

        plot_kwargs = kwargs.get("plot_kwargs", {})
        plot_kwargs["add_colorbar"] = False

        plot_data = data.isel({time_coord: 0, leadtime_coord: 0})
        plot_data = plot_data.where(plot_data.notnull(), drop=True)
        plot_object = plot_data.plot(ax=ax, transform=projection, **plot_kwargs)

        # Add Cartopy outlines or features
        ax.coastlines()  # Adds coastlines

        cbar = fig.colorbar(plot_object, orientation="horizontal", ax=ax, pad=0.1)
        textcolor = kwargs.get("textcolor", np.array([242, 240, 228]) / 255)
        cbar.ax.xaxis.set_tick_params(color=textcolor)

        cbar.set_label(
            f"{self.ds[var].attrs['longname']}, {self.ds[var].attrs['units']}",
            color=textcolor,
        )
        plt.setp(plt.getp(cbar.ax.axes, "xticklabels"), color=textcolor, fontsize=6)

        plot_facecolors(fig=fig, axes=ax, **kwargs)

        fig.suptitle(
            f"Tropical Cyclone {self.__parent.name.capitalize()} ({self.__parent.uid})",
            color=textcolor,
        )

        # Get the 0th time in YYYY-MM-DD from self.ds
        time = self.ds[time_coord].values[0].astype("datetime64[D]").astype(str)
        plt.tight_layout()

        # set ax title
        ax.set_title(
            f"{var} forecast ({time}) for leadtime +{data.isel({leadtime_coord: 0})[leadtime_coord].values.item()}h"
        )

        def update(frame):
            ax.clear()
            time = self.ds[time_coord].values[frame].astype("datetime64[D]").astype(str)
            plot_data = data.isel({time_coord: frame // 10, leadtime_coord: frame % 10})
            plot_data = plot_data.where(plot_data.notnull(), drop=True)
            plot_data.plot(ax=ax, transform=projection, **plot_kwargs)
            ax.coastlines()
            plot_facecolors(fig=fig, axes=ax)
            ax.set_title(
                f"{var} forecast ({time}) for leadtime +{data.isel({leadtime_coord: frame%10})[leadtime_coord].values.item()}h"
            )
            plt.tight_layout()

        ani = FuncAnimation(
            fig,
            update,
            np.arange(1, valid_steps.size, 1),
            blit=False,
            interval=800,
        )

        ani.save(save_path)


# Class representing the Reanalysis data corresponding to a track object
# This class becomes a subobject of the TC_track object
class ReAnal:
    __valid_kwargs = ["filepath", "source", "cache_dir", "masktype"]

    __default_vals = {
        "filepath": None,
        "source": "ERA5",
        "cache_dir": None,
        "masktype": "rect",
    }

    def __str__(self):
        return f"Reanalysis data object for {self.__parent.uid}"

    def __init__(self, parent, **kwargs):
        self.__parent = parent
        self.model = kwargs.get("source", "ERA5")

        if kwargs.get("filepath", None) is None:
            self.filepath = os.path.join(
                self.__parent.datadir_path,
                str(self.__parent.season),
                f"{self.__parent.uid}.ReAnal.{self.model}.nc",
            )
        else:
            self.filepath = kwargs.get("filepath")

        if kwargs.get("cache_dir", None) is None:
            self.cache_dir = os.path.join(self.__parent.datadir_path, "cache")
        else:
            self.cache_dir = kwargs.get("cache_dir")

    def load(self, **kwargs):
        """
        Function to load the Reanalysis data from storage

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        if kwargs.get("verbose", False):
            print(f"Loading {self.model} reanalysis data for {self.__parent.uid}...")

        # Check if the dataset doesn't exist in the object
        if not hasattr(self, "ds"):
            if os.path.exists(self.filepath):
                # If it does, load the dataset
                # print("Loading dataset...")
                setattr(self, "ds", xr.open_dataset(self.filepath))
            else:
                if kwargs.get("verbose", False) or kwargs.get("debug", False):
                    print(
                        f"Dataset {self.__parent.uid}.ReAnal.{self.model}.nc not found on disk"
                    )
                    print(
                        'You can create the dataset by running the "process_data_collection" method'
                        " with a Reanalysis data collection object as an argument."
                    )

    def serve(self, **kwargs):
        try:
            # check if the cache dir exists
            cache_dtype = kwargs.get("cache_dtype", float)
            # cache_dir = kwargs.get(
            #     "cache_dir", os.path.join(self.datadir_path, "cache")
            # )
            load_from_cache = kwargs.get("use_cached", True)
            # ai_model = kwargs.get("ai_model", "panguweather")

            if os.path.exists(self.cache_dir) and load_from_cache:
                try:
                    # check if the cache files exist. If they don't exist, continue
                    if not all(
                        [
                            os.path.exists(
                                os.path.join(
                                    self.cache_dir, f"{self.__parent.uid}_{file}.npy"
                                )
                            )
                            for file in ["IC-X", "IC-Y", "IC-t", "IC-leads"]
                        ]
                    ):
                        raise FileNotFoundError(
                            "Cache files not found. Rebuilding cache..."
                        )

                    # load the files into dask arrays
                    X = da.from_array(
                        np.load(
                            os.path.join(
                                self.cache_dir, f"{self.__parent.uid}_IC-X.npy"
                            ),
                            mmap_mode="r",
                        ),
                        chunks=(kwargs.get("chunk_size", 256), 5, 241, 241),
                    )
                    Y = da.from_array(
                        np.load(
                            os.path.join(
                                self.cache_dir, f"{self.__parent.uid}_IC-Y.npy"
                            ),
                            mmap_mode="r",
                        )
                    )
                    t = da.from_array(
                        np.load(
                            os.path.join(
                                self.cache_dir, f"{self.__parent.uid}_IC-t.npy"
                            ),
                            mmap_mode="r",
                        )
                    )
                    leads = da.from_array(
                        np.load(
                            os.path.join(
                                self.cache_dir, f"{self.__parent.uid}_IC-leads.npy"
                            ),
                            mmap_mode="r",
                        )
                    )
                    if kwargs.get("verbose", False):
                        print(
                            f"Succesfully loaded IC cache files for {self.__parent.uid} from cache...",
                            flush=True,
                        )

                    return X, Y, t, leads
                except Exception as e:
                    if kwargs.get("debug", False):
                        print(
                            f"Error loading IC cache files for {self.__parent.uid}. Rebuilding cache..."
                        )
                        if kwargs.get("verbose", False):
                            print(e)
            else:
                if not os.path.exists(self.cache_dir):
                    os.makedirs(self.cache_dir)

            if kwargs.get("verbose", False) and kwargs.get("use_cached", True):
                print(
                    f"used_cached disabled or IC cache files not found for {self.__parent.uid}. Rebuilding cache..."
                )

            # Check if the dataset doesn't exist in the object

            if not hasattr(self, "ds"):
                # if it doesnt, try loading it
                self.load(**kwargs)

            assert hasattr(
                self, "ds"
            ), f"ReAnal dataset not found for {self.__parent.uid} with model {self.model}"

            _, _, time_coord, _, _ = get_coord_vars(self.ds)

            # Get ground truth and valid timestamps
            gt, stamps = self.__parent.get_ground_truth(**kwargs)

            timedeltas = kwargs.get(
                "timedeltas", [6, 12, 24, 48, 72, 96, 120, 144, 168]
            )
            timedeltas = np.array(timedeltas).astype("timedelta64[h]")

            valid_stamps = sanitize_timestamps(stamps, self.ds, time_coord)

            # assert that timedeltas as a list, numpy array, or tuple, and length > 0
            assert (
                isinstance(timedeltas, (list, np.ndarray, tuple))
                and len(timedeltas) > 0
            ), "Invalid timedeltas argument"

            gt = gt[np.isin(stamps, valid_stamps)]
            stamps = stamps[np.isin(stamps, valid_stamps)]

            if len(gt) == 0:
                if kwargs.get("verbose", False) or kwargs.get("debug", False):
                    print(
                        f"No valid timestamps found for {self.__parent.uid} with model {self.model}"
                    )
                return

            # Get the AI data
            inputs = None
            targets = None
            outstamps = None
            out_leads = None
            for stamp in stamps:
                temp_ds = self.ds.sel({time_coord: stamp})
                temp_ds = temp_ds.where(~temp_ds.isnull(), drop=True)

                # timedeltas = temp_ds[leadtime_coord].values.astype("timedelta64[h]")
                leadtimes = stamp + timedeltas

                # make a boolean index to see what leadtime data we have a
                # truth value for
                bool_idx = np.isin(leadtimes, stamps)

                # read in the data as a numpy array
                out_data = temp_ds.to_array().values
                # because we need to copy the IC data across leadtimes, we tile
                # the ICS
                out_data = np.tile(out_data, (bool_idx.sum(), 1, 1, 1))
                out_targets = gt[np.isin(stamps, leadtimes[bool_idx])]

                # base_targets = gt[stamps == stamp]
                temp_stamps = leadtimes[bool_idx]
                temp_leads = timedeltas[bool_idx].astype(int)

                if outstamps is None:
                    try:
                        ##TODO: The following check appears to be and issue
                        ## with the way the data is being filtered when processing
                        ## the storm fields. My guess is that this is due to the
                        ## crossing of the 0° longitude line. This should be fixed
                        ## process_data_collection method.

                        if (
                            out_data.shape[-2:]
                            != (
                                241,
                                241,
                            )
                            or out_data.shape[-3] != 5
                        ):
                            raise ValueError(
                                f"Invalid shape for stamp {stamp}... Skipping..."
                            )
                        else:
                            inputs = out_data
                            targets = out_targets
                            outstamps = temp_stamps  # leadtimes[bool_idx]
                            # base_stamps = np.full_like(outstamps, stamp)
                            out_leads = temp_leads.astype(int)

                    except ValueError as e:
                        if kwargs.get("verbose", False):
                            print(f"Problem processing {stamp}... Skipping...")
                            print(e)
                else:
                    try:
                        if (out_data.shape[-2:] != (241, 241)) or (
                            out_data.shape[-3] != 5
                        ):
                            raise ValueError(
                                f"Invalid shape for stamp {stamp}... Skipping..."
                            )
                        else:
                            inputs = np.vstack([inputs, out_data])
                            targets = np.vstack([targets, out_targets])
                            outstamps = np.hstack([outstamps, temp_stamps])
                            out_leads = np.hstack([out_leads, temp_leads])

                        # outstamps = np.hstack([outstamps, leadtimes[bool_idx]])
                        # out_leads = np.hstack(
                        #     [out_leads, temp_ds[leadtime_coord].values[bool_idx]]
                        # )

                    except ValueError as e:
                        if kwargs.get("verbose", False):
                            print(f"Problem processing {stamp}... Skipping...")
                            print(e)

            # save if the data is not empty
            if inputs is not None:
                # save the data to the cache
                np.save(
                    os.path.join(self.cache_dir, f"{self.__parent.uid}_IC-X.npy"),
                    inputs,
                )
                np.save(
                    os.path.join(self.cache_dir, f"{self.__parent.uid}_IC-Y.npy"),
                    targets,
                )
                np.save(
                    os.path.join(self.cache_dir, f"{self.__parent.uid}_IC-t.npy"),
                    outstamps,
                )
                np.save(
                    os.path.join(self.cache_dir, f"{self.__parent.uid}_IC-leads.npy"),
                    out_leads,
                )

                # load saved files into dask arrays
                X = da.from_array(
                    np.load(
                        os.path.join(self.cache_dir, f"{self.__parent.uid}_IC-X.npy"),
                        mmap_mode="r",
                    ),
                    chunks=(kwargs.get("chunk_size", 256), 5, 241, 241),
                )
                Y = da.from_array(
                    np.load(
                        os.path.join(self.cache_dir, f"{self.__parent.uid}_IC-Y.npy"),
                        mmap_mode="r",
                    )
                )
                t = da.from_array(
                    np.load(
                        os.path.join(self.cache_dir, f"{self.__parent.uid}_IC-t.npy"),
                        mmap_mode="r",
                    )
                )
                leads = da.from_array(
                    np.load(
                        os.path.join(
                            self.cache_dir, f"{self.__parent.uid}_IC-leads.npy"
                        ),
                        mmap_mode="r",
                    )
                )
                return X, Y, t, leads
            else:
                # save empty arrays to the cache
                np.save(
                    os.path.join(self.cache_dir, f"{self.__parent.uid}_IC-X.npy"),
                    np.array([]),
                )
                np.save(
                    os.path.join(self.cache_dir, f"{self.__parent.uid}_IC-Y.npy"),
                    np.array([]),
                )
                np.save(
                    os.path.join(self.cache_dir, f"{self.__parent.uid}_IC-t.npy"),
                    np.array([]),
                )
                np.save(
                    os.path.join(self.cache_dir, f"{self.__parent.uid}_IC-leads.npy"),
                    np.array([]),
                )
                return np.array([]), np.array([]), np.array([]), np.array([])

        except Exception as e:
            if kwargs.get("verbose", False):
                print(
                    f"Error serving AI data for {self.__parent.uid} with model {self.model}"
                )
                print(traceback.format_exc())
                # print(e)
            return None

    def animate_2d(self, **kwargs):
        if not hasattr(self, "ds"):
            self.load()
        dpi = kwargs.get("dpi", 150)
        save_path = kwargs.get(
            "save_path",
            f"/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/data/{self.__parent.name}_animation.gif",
        )

        self.ds["wind_magnitude"] = np.sqrt(self.ds["10u"] ** 2 + self.ds["10v"] ** 2)
        self.ds["wind_direction"] = np.arctan2(self.ds["10v"], self.ds["10u"])

        self.ds["wind_magnitude"].attrs = {
            "long_name": "Wind magnitude",
            "units": "m/s",
        }

        self.ds["wind_direction"].attrs = {
            "long_name": "Wind direction",
            "units": "radians",
        }

        # Check how many variables are in the dataset
        if len(self.ds.data_vars) > 1:
            numvars = len(self.ds.data_vars)

        cmaps = [
            "seismic",
            "seismic",
            "cividis_r",
            "inferno",
            "cividis_r",
            "seismic",
            "seismic",
        ]

        # Make a figure with 1 column and numvar rows
        fig, axs = plt.subplots(
            1,
            numvars,
            dpi=dpi,
            figsize=(7 * numvars, 5),
            # subplot_kw={"projection": ccrs.PlateCarree()},
        )

        minn, maxx = (
            self.ds.min().to_array().values,
            self.ds.max().to_array().values,
        )

        # Make the first frame:
        for i in range(numvars):
            data = self.ds.isel(time=0)
            data = data.where(data.notnull(), drop=True)
            # extent = data.latitude.min.values,
            # ax.set_extent(extent, crs=ccrs.PlateCarree())
            # axs[i].coastlines()
            data[list(self.ds.data_vars)[i]].plot.imshow(
                cmap=cmaps[i],
                ax=axs[i],
                vmin=minn[i],
                vmax=maxx[i],
                transform=ccrs.PlateCarree(),
            )
            axs[i].set_title(list(self.ds.data_vars)[i])

        plt.tight_layout()

        def update(frame):
            for ax in axs:
                ax.clear()
            for i in range(numvars):
                data = self.ds.isel(time=frame)
                # axs[i].coastlines()
                data = data.where(data.notnull(), drop=True)
                data[list(self.ds.data_vars)[i]].plot.imshow(
                    cmap=cmaps[i],
                    ax=axs[i],
                    vmin=minn[i],
                    vmax=maxx[i],
                    add_colorbar=False,
                )

                # axs[i].set_title(list(self.ds.data_vars)[i])

        ani = FuncAnimation(
            fig,
            update,
            np.arange(1, len(self.ds.time), 1),
            blit=False,
            interval=400,
        )

        ani.save(save_path)

    def animate_data(self, var, **kwargs):
        figsize = kwargs.get("figsize", (4, 6))
        dpi = kwargs.get("dpi", 150)
        save_path = kwargs.get(
            "save_path",
            f"/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/data/{var}_animation.gif",
        )

        # Check if the dataset doesn't exist in the object
        if not hasattr(self, "ds"):
            print("Data not yet loaded - trying to load it now...")
            self.load(**kwargs)

        # # Assert that the variable exists in the object data
        # assert (
        #     var in self.__getattribute__(f"{kwargs.get('ds_type', 'rad')}_ds").data_vars
        # ), f"Variable {var} not found in dataset"

        # get sanitized timestamps
        valid_steps = sanitize_timestamps(self.__parent.timestamps, self.ds)

        fig, ax = plt.subplots(
            figsize=figsize,
            dpi=dpi,  # subplot_kw={"projection": "3d"}
        )

        minn, maxx = (
            self.__getattribute__(f"{kwargs.get('ds_type', 'rad')}_ds")[var]
            .min()
            .values,
            self.__getattribute__(f"{kwargs.get('ds_type', 'rad')}_ds")[var]
            .max()
            .values,
        )

        self.plot3D(var, valid_steps[0], minn=minn, maxx=maxx, fig=fig, ax=ax, **kwargs)

        def update(frame):
            ax.clear()
            self.plot3D(
                var,
                valid_steps[frame],
                add_colorbar=False,
                minn=minn,
                maxx=maxx,
                fig=fig,
                ax=ax,
                **kwargs,
            )

        ani = FuncAnimation(
            fig,
            update,
            np.arange(1, valid_steps.size, 1),
            blit=False,
            interval=400,
        )

        ani.save(save_path)

    # def plot3D(self, var, timestamp, **kwargs):
    #     """
    #     Function to plot 3D data on a map, showing the variable in 3D for the given timestamps

    #     Based off of:
    #     https://stackoverflow.com/questions/13570287/image-overlay-in-3d-plot
    #     https://stackoverflow.com/questions/23785408/3d-cartopy-similar-to-matplotlib-basemap
    #     https://stackoverflow.com/questions/48269014/contourf-in-3d-cartopy%5D
    #     Parameters
    #     ----------
    #     var : STR, required
    #         Variable to plot

    #     timestamp : STR, required
    #         Timestamp to plot

    #     Returns
    #     -------
    #     None

    #     """
    #     ignore_levels = kwargs.get("ignore_levels", None)
    #     facecolor = kwargs.get("facecolor", (0.3, 0.3, 0.3))
    #     text_color = kwargs.get("text_color", "white")
    #     view_angles = kwargs.get("view_angles", (25, -45))
    #     figsize = kwargs.get("figsize", (4, 6))
    #     dpi = kwargs.get("dpi", 150)
    #     cmap = kwargs.get("cmap", "seismic")
    #     fig = kwargs.get("fig", None)
    #     ax3d = kwargs.get("ax", None)
    #     add_colorbar = kwargs.get("add_colorbar", True)
    #     minn, maxx = kwargs.get("minn", None), kwargs.get("maxx", None)

    #     # Check if the dataset doesn't exist in the object
    #     if not hasattr(self, f"{kwargs.get('ds_type', 'rad')}_ds"):
    #         print("Data not yet loaded - trying to load it now...")
    #         self.load_data(**kwargs)

    #     alpha = kwargs.get("alpha", 0.3)
    #     data = getattr(self, f"{kwargs.get('ds_type', 'rad')}_ds")[var]

    #     if ignore_levels:
    #         data = data.sel(
    #             {
    #                 kwargs.get("level_coord", "level"): ~data[
    #                     kwargs.get("level_coord", "level")
    #                 ].isin(ignore_levels)
    #             }
    #         )

    #     if fig is None and ax3d is None:
    #         fig, ax3d = plt.subplots(
    #             figsize=figsize, dpi=dpi, subplot_kw={"projection": "3d"}
    #         )
    #     elif (fig is not None) == (ax3d is None):  # xor
    #         raise ValueError(
    #             "Both fig and ax must be provided if providing the figure or the axes"
    #         )

    #     fig.set_facecolor(facecolor)
    #     ax3d.set_facecolor(facecolor)
    #     lat_coord, lon_coord, time_coord, level_coord, _ = get_coord_vars(data)

    #     if minn is None and maxx is None:
    #         minn, maxx = (
    #             data.sel({time_coord: timestamp}).min().values,
    #             data.sel({time_coord: timestamp}).max().values,
    #         )
    #     elif (minn is not None) == (maxx is None):  # xor
    #         raise ValueError("Both minn and maxx must be provided if one is provided")

    #     if level_coord:
    #         for level in data[level_coord].values:
    #             z = int(level)

    #             level_data = data.sel({time_coord: timestamp, level_coord: level})
    #             level_data = level_data.where(~level_data.isnull(), drop=True)

    #             norm = mpl.colors.Normalize(vmin=minn, vmax=maxx)
    #             m = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    #             m.set_array([])
    #             fcolors = m.to_rgba(level_data.values)

    #             anomaly = np.abs(level_data.values - level_data.values.mean())
    #             anomaly = (anomaly / anomaly.max()) ** 1.5

    #             fcolors[..., 3] = anomaly

    #             X, Y = np.meshgrid(
    #                 level_data[lon_coord].values, level_data[lat_coord].values[::-1]
    #             )

    #             ax3d.plot_surface(
    #                 X,
    #                 Y,
    #                 np.ones(X.shape) * z,
    #                 facecolors=fcolors,
    #                 vmin=minn,
    #                 vmax=maxx,
    #                 shade=False,
    #             )

    #     ax3d.xaxis.pane.fill = False
    #     ax3d.yaxis.pane.fill = False
    #     ax3d.zaxis.pane.fill = False
    #     ax3d.xaxis.pane.set_edgecolor(facecolor)
    #     ax3d.yaxis.pane.set_edgecolor(facecolor)
    #     ax3d.zaxis.pane.set_edgecolor(facecolor)
    #     ax3d.xaxis.line.set_color(text_color)
    #     ax3d.yaxis.line.set_color(text_color)
    #     ax3d.zaxis.line.set_color(text_color)
    #     ax3d.tick_params(axis="z", colors=text_color, pad=-1, labelsize=6)
    #     ax3d.tick_params(
    #         axis="x",
    #         colors=text_color,
    #         pad=-5,
    #         labelrotation=view_angles[0],
    #         labelsize=6,
    #     )
    #     ax3d.tick_params(
    #         axis="y",
    #         colors=text_color,
    #         pad=-5,
    #         labelrotation=-view_angles[0],
    #         labelsize=6,
    #     )
    #     ax3d.view_init(*view_angles)
    #     ax3d.grid(False)
    #     if add_colorbar:
    #         cbar = fig.colorbar(m, ax=ax3d, orientation="horizontal", pad=0.1)
    #         cbar.ax.xaxis.set_tick_params(color=text_color)
    #         cbar.set_label(
    #             f"{data.attrs['long_name']}, {data.attrs['units']}", color=text_color
    #         )
    #         plt.setp(
    #             plt.getp(cbar.ax.axes, "xticklabels"), color=text_color, fontsize=6
    #         )
    #     ax3d.invert_zaxis()
    #     fig.tight_layout()
    #     plt.show()


# %% debugging with main
if __name__ == "__main__":
    import data_lib as dlib

    # dc = AI_Data_Collection(
    #     "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/AI-milton/panguweather"
    # )
    # print(dc.meta_dfs)

    dc = dlib.AI_Data_Collection(
        "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/TCBench_alpha/hugging/TCBench/neural_weather_models/panguweather"
    )

    data_dir = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/TCBench_alpha"
    seasons = get_TC_seasons(
        season_list=[*range(2023, 2024)],
        datadir_path=data_dir,
    )
    storms = seasons[2023]
    for storm in storms:
        if storm.uid == "2023326N10240":
            break

# === TCBench plotting helpers (evaluation CSV utilities) ===
import re

# Canonical metric name aliases used across legacy outputs
METRIC_ALIASES = {
    # Track geometry (CARTE)
    "Along_TE": ["ATE", "along_te", "AlongTrackError"],
    "Cross_TE": ["CTE", "cross_te", "CrossTrackError"],
    "DPE_cart": ["CDPE", "dpe_cart", "cart_dpe"],
    # Track displacement / CRPS
    "DPE_GCD": ["DPE", "gcd_dpe"],
    "CRPS_haversine": ["CRPS_track", "CRPS_DPE"],
    # Intensity (wind)
    "AE_wind": ["AE_vmax", "abserr_vmax", "AE_WIND"],
    "SE_wind": ["SE_vmax", "sqerr_vmax", "SE_WIND"],
    "CRPS_vmax": ["CRPS_wind", "CRPS_VMAX"],
    # Intensity (pressure)
    "AE_pressure": ["AE_pmin", "abserr_pmin", "AE_PRESSURE"],
    "SE_pressure": ["SE_pmin", "sqerr_pmin", "SE_PRESSURE"],
    "CRPS_pmin": ["CRPS_pressure", "CRPS_PMIN"],
}


def load_eval_csv(path: str) -> pd.DataFrame:
    """Load a results CSV, parse timestamps, and add a `lead_hours` column.
    Falls back to a 6‑hour multiple if Initial/Valid times are missing.
    """
    df = pd.read_csv(path, low_memory=False)
    for c in ("Initial Time", "Valid Time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if {"Initial Time", "Valid Time"}.issubset(df.columns):
        df["lead_hours"] = (
            df["Valid Time"] - df["Initial Time"]
        ).dt.total_seconds() / 3600.0
    else:
        # Best-effort fallback
        if {"SID", "Initial Time"}.issubset(df.columns):
            df["lead_hours"] = df.groupby(["SID", "Initial Time"]).cumcount() * 6.0
        else:
            df["lead_hours"] = np.arange(len(df)) * 6.0
    return df


def find_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column name from a list of candidates (case‑insensitive)."""
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def pick_metric_col(df: pd.DataFrame, base: str) -> str | None:
    """Return the column for a metric, favoring probabilistic means ("*_mean")
    with robust fallbacks for legacy aliases. Also handles the legacy DPE name.
    """
    cols = {c.lower(): c for c in df.columns}
    candidates: list[str] = [f"{base}_mean", base]
    for alias in METRIC_ALIASES.get(base, []):
        candidates.extend([f"{alias}_mean", alias])
    if base == "DPE_GCD":
        candidates.extend(["DPE_mean", "DPE"])  # legacy
    for cand in candidates:
        if cand and cand.lower() in cols:
            return cols[cand.lower()]
    return None


# --- Helper: Compute per-lead R² by merging SE_* from results with IBTrACS GT
def compute_r2_by_lead_from_results(
    results_df: pd.DataFrame,
    ibtracs_df: pd.DataFrame,
    variable: str,  # "wind" or "pressure"
    year: int | None = None,
    debug: bool = False,
) -> pd.Series:
    """
    R²(L) = 1 - mean(SE_variable at lead L) / Var(GT_variable at lead L)

    Supports both deterministic and probabilistic results files:
      • Prefer `SE_{variable}_mean` when present (ensemble-mean squared error)
      • Else use `SE_{variable}`
      • Else fall back to squaring `AE_{variable}_mean` or `AE_{variable}`

    Assumes:
      - results_df has: ["SID", "Initial Time", "Valid Time", <error cols>]
      - ibtracs_df has: ["SID", "ISO_TIME", "USA_WIND", "USA_PRES"]
    Returns a Series indexed by integer lead hours.
    """
    assert variable in {"wind", "pressure"}
    gt_col = "USA_WIND" if variable == "wind" else "USA_PRES"

    # ---- sanity checks for required ID/time columns
    need_core = {"SID", "Initial Time", "Valid Time"}
    if not need_core.issubset(results_df.columns):
        missing = need_core - set(results_df.columns)
        raise ValueError(f"results_df missing columns: {missing}")
    need_ib = {"SID", "ISO_TIME", "USA_WIND", "USA_PRES"}
    if not need_ib.issubset(ibtracs_df.columns):
        missing = need_ib - set(ibtracs_df.columns)
        raise ValueError(f"ibtracs_df missing columns: {missing}")

    # ---- choose SE column (probabilistic-first) with robust fallbacks
    # Try SE_*_mean, then SE_*, else derive from AE_* (or AE_*_mean)^2
    se_pick = pick_metric_col(results_df, f"SE_{variable}")
    ae_pick = None
    src = None

    if se_pick is not None:
        src = "SE_mean" if se_pick.lower().endswith("_mean") else "SE"
    else:
        ae_pick = pick_metric_col(results_df, f"AE_{variable}")
        if ae_pick is None:
            tried = [
                f"SE_{variable}_mean",
                f"SE_{variable}",
                f"AE_{variable}_mean",
                f"AE_{variable}",
            ]
            raise ValueError(
                f"No squared-error columns found for variable='{variable}'. Looked for: {tried}"
            )
        src = "AE_mean^2" if ae_pick.lower().endswith("_mean") else "AE^2"

    # ---- normalize results times & compute/prepare SE column
    m = results_df.copy()
    m["Initial Time"] = pd.to_datetime(m["Initial Time"], errors="coerce")
    m["Valid Time"] = pd.to_datetime(m["Valid Time"], errors="coerce")

    if se_pick is not None:
        se_col = se_pick
        m[se_col] = pd.to_numeric(m[se_col], errors="coerce")
    else:
        # derive SE from AE
        se_col = f"__SE_from_{ae_pick}"
        m[se_col] = pd.to_numeric(m[ae_pick], errors="coerce") ** 2

    before_drop = len(m)
    m = m.dropna(subset=["SID", "Initial Time", "Valid Time", se_col]).copy()
    if m.empty:
        if debug:
            print(f"[R2/{variable}] No usable rows after coercion/dropna.")
        return pd.Series(dtype=float)

    # integer lead (nearest hour)
    m["lead_hours"] = (
        ((m["Valid Time"] - m["Initial Time"]).dt.total_seconds() / 3600.0)
        .round()
        .astype(int)
    )

    # hour key for merge on valid time (naive, floored to hour → int64 hrs since epoch)
    vt = m["Valid Time"]
    try:
        # if tz-aware, convert to naive; else leave as is
        if getattr(vt.dt, "tz", None) is not None:
            vt = vt.dt.tz_convert(None)
    except Exception:
        pass
    m["valid_hr"] = vt.dt.floor("h").astype("int64") // 3_600_000_000_000

    if debug:
        print(f"[R2/{variable}] results: rows={before_drop:,} -> kept={len(m):,}")
        print(
            f"[R2/{variable}] using column '{se_col}' (source={src}) | dtype={m[se_col].dtype}"
        )

    # ---- IBTrACS: filter year, normalize time key, coerce GT
    ib = ibtracs_df.copy()
    ib["ISO_TIME"] = pd.to_datetime(ib["ISO_TIME"], errors="coerce")
    if year is not None:
        ib = ib[ib["ISO_TIME"].dt.year == year]
    ib[gt_col] = pd.to_numeric(ib[gt_col], errors="coerce")
    ib = ib.dropna(subset=["SID", "ISO_TIME", gt_col]).copy()

    iso = ib["ISO_TIME"]
    try:
        if getattr(iso.dt, "tz", None) is not None:
            iso = iso.dt.tz_convert(None)
    except Exception:
        pass
    ib["iso_hr"] = iso.dt.floor("h").astype("int64") // 3_600_000_000_000

    # ---- merge GT onto model rows by (SID, valid_hr)
    merged = m.merge(
        ib[["SID", "iso_hr", gt_col]].rename(columns={"iso_hr": "valid_hr"}),
        on=["SID", "valid_hr"],
        how="left",
    )
    use = merged.dropna(subset=[se_col, gt_col, "lead_hours"]).copy()
    if debug:
        total = len(merged)
        have_gt = merged[gt_col].notna().sum()
        pct = (have_gt / total * 100.0) if total else 0.0
        print(
            f"[R2/{variable}] merge: rows={total:,}; with GT={have_gt:,} ({pct:.1f}%)"
        )
        print(f"[R2/{variable}] usable rows: {len(use):,}")

    if use.empty:
        return pd.Series(dtype=float)

    # ---- per-lead MSE and GT variance (sample variance)
    g = use.groupby("lead_hours", dropna=False)
    mse = g[se_col].mean()
    var_gt = g[gt_col].var(ddof=1)

    # compute R² safely; drop non-finite and clip to <= 1
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - (mse / var_gt)
    r2 = r2.replace([np.inf, -np.inf], np.nan).dropna().clip(upper=1.0)
    r2.name = f"R2_{variable}"

    if debug:
        try:
            sample = list(zip(r2.index[:5], r2.values[:5]))
        except Exception:
            sample = []
        print(f"[R2/{variable}] produced {r2.size} R² points; sample: {sample}")

    return r2.sort_index()
