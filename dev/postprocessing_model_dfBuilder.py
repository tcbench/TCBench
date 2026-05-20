#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% Imports
# OS and IO
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from importlib import reload

# Backend Libraries
import xarray as xr

from utils import toolbox, constants
from utils.toolbox import *
from utils import data_lib as dlib
from utils import ML_functions as mlf
import torch
import dask

# import cartopy for coastlines
import cartopy.crs as ccrs
import cartopy.feature as cfeature


import warnings

# # Suppress all warnings
# warnings.filterwarnings("ignore")

# Optionally, if you want to suppress only Dask-specific warnings:
# import dask
warnings.filterwarnings("ignore", module="dask")


# %%

model_dir = os.path.join(os.curdir, "postproc_models")
results_dir = os.path.join(os.curdir, "outputs")

full_data = toolbox.read_hist_track_file(
    tracks_path=os.path.join(os.curdir, "data", "ibtracs")
)

year = 2023

# filter full data to 2023
full_data = full_data[full_data.ISO_TIME.dt.year == year]


# %%
if __name__ == "__main__":

    calc_device = torch.device("cpu")

    ai_model = "panguweather"
    magangle = True

    cache_dir = os.path.join(os.curdir, "data", "cache")

    AI_scaler = None
    fpath = os.path.join(cache_dir, "AI_scaler.pkl")

    if os.path.exists(fpath):
        print("Loading AI scaler from cache...", flush=True)
        with open(fpath, "rb") as f:
            AI_scaler = pickle.load(f)
    else:
        raise FileNotFoundError(f"AI scaler not found at {fpath}")

    base_scaler = None
    fpath = os.path.join(cache_dir, "base_scaler.pkl")

    if os.path.exists(fpath):
        print("Loading base scaler from cache...", flush=True)
        with open(fpath, "rb") as f:
            base_scaler = pickle.load(f)
    else:
        raise FileNotFoundError(f"Base scaler not found at {fpath}")

    target_scaler = None
    fpath = os.path.join(cache_dir, "target_scaler.pkl")

    if os.path.exists(fpath):
        print("Loading target scaler from cache...", flush=True)
        with open(fpath, "rb") as f:
            target_scaler = pickle.load(f)
    else:
        raise FileNotFoundError(f"Target scaler not found at {fpath}")

    mask_inputs = True
    mask_path = os.path.join(os.curdir, "data", "mask_dict.pkl")
    with open(mask_path, "rb") as f:
        mask_dict = pickle.load(f)["linear"]

    # Load the MLR model
    # Load the models to evaluate
    models_to_load = [
        {
            "filepath": os.path.join(model_dir, "Probabilistic_ANN_masked.pt"),
            "masked": True,
            "probabilistic": True,
            "tag": "Probabilistic ANN (M)",
            "results": {},
            "deep": False,
        },
        {
            "filepath": os.path.join(model_dir, "Probabilistic_MLR_masked.pt"),
            "masked": True,
            "probabilistic": True,
            "tag": "Probabilistic MLR (M)",
            "results": {},
            "deep": False,
        },
        {
            "filepath": os.path.join(model_dir, "Probabilistic_UNet_masked.pt"),
            "masked": True,
            "probabilistic": True,
            "tag": "Probabilistic UNet (M)",
            "results": [],
            "deep": True,
        },
    ]

    for model in models_to_load:
        model["model"] = torch.load(model["filepath"], map_location=calc_device)
        model["model"].eval()
    # %%
    fit = True  # Set to True to fit the models, False to load the models from disk
    if fit:
        plotting_dict = {}
        initial_times = []
        i = 0
        total = len(full_data.SID.unique())
        for SID in full_data.SID.unique():
            i += 1
            print(f"Processing {SID}; {i} of {total}", flush=True)
            plotting_dict[SID] = {}
            storm = full_data[full_data.SID == SID]
            track = toolbox.tc_track(
                UID=storm.SID.iloc[0],
                NAME=storm.NAME.iloc[0],
                track=storm[["LAT", "LON"]].to_numpy(),
                timestamps=storm.ISO_TIME.to_numpy(),
                ALT_ID=storm[
                    constants.ibtracs_cols._track_cols__metadata.get("ALT_ID")
                ].iloc[0],
                wind=storm[
                    constants.ibtracs_cols._track_cols__metadata.get("WIND")
                ].to_numpy(),
                pres=storm[
                    constants.ibtracs_cols._track_cols__metadata.get("PRES")
                ].to_numpy(),
                datadir_path=os.path.join(os.curdir, "data"),
                storm_season=storm.SEASON.iloc[0],
                ai_model=ai_model,
            )

            truth_int, truth_time = track.get_ground_truth()
            plotting_dict[SID]["IBTrACS"] = (truth_int, truth_time)

            try:
                ai_data, base_intensity, target_time, target_leadtime = track.ai.serve()
            except Exception as e:
                print(f"Error loading data for {SID}: {e}")
                continue

            try:
                if magangle:
                    mlf.uv_to_magAngle(data=ai_data, u_idx=0, v_idx=1)

                ai_data = ai_data.compute()
                base_intensity_data = base_intensity.compute()
                target_time = target_time.compute()
                target_leadtime = target_leadtime.compute()
                base_time = target_time - target_leadtime.astype("timedelta64[h]")
                forecast_time = target_time - target_leadtime.astype("timedelta64[h]")
            except Exception as e:
                print(f"Error computing data for {SID}: {e}")
                continue

            # get the track timestamps
            track_time = track.timestamps

            # find the index of the forecast time in the track time
            forecast_idx = np.searchsorted(track_time, forecast_time)
            forecast_positions = track.track[forecast_idx]
            forecast_positions = mlf.latlon_to_sincos(forecast_positions)

            print("Starting prediction loop...")
            for model in models_to_load:
                plotting_dict[SID][model["tag"]] = {}

            for mode in ["MLR", "deep"]:
                for leadtime in range(6, 121, 6):
                    # for leadtime in [6, 12, 18, 24, 48, 72, 96, 120]:
                    # print(f"Processing {SID} {leadtime} hour forecast")
                    bool_idx = target_leadtime == leadtime

                    if sum(bool_idx.astype(int)) == 0:
                        continue

                    temp_ai = ai_data[bool_idx]
                    temp_bi = base_intensity[bool_idx]
                    temp_targettime = target_time[bool_idx]
                    temp_ldt = np.full_like(temp_targettime, leadtime / 168).astype(
                        float
                    )
                    temp_pos = forecast_positions[bool_idx]
                    temp_ai = AI_scaler.transform(temp_ai)
                    temp_bi = base_scaler.transform(temp_bi)

                    temp_mask = mask_dict[leadtime]

                    if mask_inputs:
                        temp_ai = temp_ai * temp_mask

                    if mode == "MLR":
                        temp_maxima = temp_ai.max(axis=(-2, -1))
                        temp_minima = temp_ai.min(axis=(-2, -1))
                        temp_range = temp_maxima - temp_minima

                        temp_inputs = np.vstack(
                            [
                                temp_maxima[:, 0],  # Maximum wind magnitude
                                temp_minima[:, 2],  # Minimum mean sea level pressure
                                temp_range[:, 0],  # Range of wind magnitude
                                temp_range[:, 2],  # Range of mean sea level pressure
                                temp_minima[
                                    :, 3
                                ],  # Minimum geopotential height at 500 hPa
                                temp_range[:, 4],  # Range of temperature at 850 hPa
                                temp_ldt.squeeze(),  # Leadtime
                                temp_bi.T,  # Base intensity
                            ]
                        ).T

                        for model in models_to_load:

                            if not model["deep"]:
                                temp_x = torch.tensor(
                                    temp_inputs, dtype=torch.float32
                                ).to(calc_device)
                            else:
                                pass
                                # print("Deep model, skipping in MLR inputs mode...")

                            with torch.no_grad():
                                if not model["deep"]:
                                    temp_pred = model["model"](temp_x).numpy()
                                    if model["probabilistic"]:
                                        temp_means = temp_pred[:, [0, 2]]
                                        temp_sigma = np.abs(temp_pred[:, [1, 2]])
                                        temp_sigma = target_scaler.inverse_transform(
                                            temp_sigma
                                        )
                                        temp_means = target_scaler.inverse_transform(
                                            temp_means
                                        )
                                        temp_base_adder = base_scaler.inverse_transform(
                                            temp_bi
                                        )
                                        # temp_means = temp_means + temp_base_adder
                                        # temp_95_lower = temp_means - 1.96 * temp_sigma
                                        # temp_95_upper = temp_means + 1.96 * temp_sigma

                            if not model["deep"]:
                                plotting_dict[SID][model["tag"]][leadtime] = {
                                    "base_intensity": temp_base_adder,
                                    "mean_intensification": temp_means,
                                    "sigma_intensification": temp_sigma,
                                    "time": temp_targettime,
                                }

                    elif mode == "deep":
                        scalars = np.vstack(
                            [
                                temp_bi.T,
                                temp_pos.T,
                                np.full(temp_bi.shape[0], leadtime / 168).reshape(
                                    1, -1
                                ),
                            ]
                        ).T

                        for model in models_to_load:
                            if model["deep"]:
                                temp_x = torch.tensor(temp_ai, dtype=torch.float32).to(
                                    calc_device
                                )
                                temp_scalars = torch.tensor(
                                    scalars, dtype=torch.float32
                                ).to(calc_device)
                            else:
                                pass
                                # print("Linear model, skipping in deep inputs mode...")

                            with torch.no_grad():
                                if model["deep"]:
                                    temp_pred = model["model"](
                                        temp_x, temp_scalars
                                    ).numpy()
                                    if model["probabilistic"]:
                                        temp_means = temp_pred[:, [0, 2]]
                                        temp_sigma = np.abs(temp_pred[:, [1, 2]])
                                        temp_sigma = target_scaler.inverse_transform(
                                            temp_sigma
                                        )
                                        temp_means = target_scaler.inverse_transform(
                                            temp_means
                                        )
                                        temp_base_adder = base_scaler.inverse_transform(
                                            temp_bi
                                        )
                                        # temp_means = temp_means + temp_base_adder
                                        # temp_95_lower = temp_means - 1.96 * temp_sigma
                                        # temp_95_upper = temp_means + 1.96 * temp_sigma
                                    else:
                                        "Deterministic model, skipping for now..."
                                        continue
                            if model["deep"]:
                                plotting_dict[SID][model["tag"]][leadtime] = {
                                    "base_intensity": temp_base_adder,
                                    "mean_intensification": temp_means,
                                    "sigma_intensification": temp_sigma,
                                    "time": temp_targettime,
                                }

        np.save(f"{results_dir}temp_df_data_{year}.npy", plotting_dict)
    else:
        plotting_dict = np.load(
            f"{results_dir}temp_df_data_{year}.npy", allow_pickle=True
        ).item()

# %%
model_tags = None
model_dfs = {}
num_members = 50
for id, data in plotting_dict.items():
    if model_tags is None:
        model_tags = [key for key in data.keys() if key not in ["IBTrACS"]]
        if len(model_tags) == 0:
            model_tags = None
            continue
        for tag in model_tags:
            if tag in data:
                model_dfs[tag] = pd.DataFrame()
    for tag in model_tags:
        if tag in data:
            for leadtime, results in data[tag].items():
                df = pd.DataFrame()
                df["SID"] = [id] * len(results["time"])
                df["Initial Time"] = results["time"] - np.array(
                    [np.timedelta64(int(leadtime), "h")] * len(results["time"])
                )
                df["Valid Time"] = results["time"]
                df["Lead Time (h)"] = [leadtime] * len(results["time"])
                df["Base Intensity (knots)"] = results["base_intensity"][:, 0].squeeze()
                df["Base Intensity (hpa)"] = results["base_intensity"][:, 1].squeeze()

                # thing from which we will sample
                mu_kt = results["mean_intensification"][:, 0].squeeze()
                mu_hpa = results["mean_intensification"][:, 1].squeeze()
                sigma_kt = np.abs(results["sigma_intensification"][:, 0].squeeze())
                sigma_hpa = np.abs(results["sigma_intensification"][:, 1].squeeze())

                # populate the ensemble members
                for member in range(num_members):
                    ensemble_df = df.copy()
                    ensemble_df["intensification_kt"] = np.random.normal(
                        loc=mu_kt, scale=sigma_kt
                    )
                    ensemble_df["intensification_hpa"] = np.random.normal(
                        loc=mu_hpa, scale=sigma_hpa
                    )
                    ensemble_df["ensemble_idx"] = member
                    ensemble_df["wind max"] = (
                        ensemble_df["Base Intensity (knots)"]
                        + ensemble_df["intensification_kt"]
                    )
                    ensemble_df["pres min"] = (
                        ensemble_df["Base Intensity (hpa)"]
                        + ensemble_df["intensification_hpa"]
                    )

                    low_wind_mask = ensemble_df["wind max"] < 0

                    ensemble_df.loc[low_wind_mask, "intensification_kt"] = (
                        -ensemble_df.loc[low_wind_mask, "Base Intensity (knots)"]
                    )
                    ensemble_df.loc[low_wind_mask, "wind max"] = 0

                    model_dfs[tag] = pd.concat(
                        [model_dfs[tag], ensemble_df], ignore_index=True
                    )
                # model_dfs[tag] = pd.concat([model_dfs[tag], df], ignore_index=True)

# %%
for model_name, model_df in model_dfs.items():
    # sort by SID, Initial Time, Lead Time, ensemble_idx
    model_df.sort_values(
        by=["SID", "ensemble_idx", "Initial Time", "Lead Time (h)"], inplace=True
    )
    model_df.reset_index(drop=True, inplace=True)
    # drop the base intensity columns
    try:
        model_df.drop(
            columns=[
                "Base Intensity (knots)",
                "Base Intensity (hpa)",
                "intensification_kt",
                "intensification_hpa",
            ],
            inplace=True,
        )
    except Exception as e:
        print(f"Error dropping columns: {e}")
    model_df.to_csv(
        f"{results_dir}postprocessing_panguweather_0shot_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_{year}.csv",
        index=False,
    )


# %%
