# Climatology Baseline Models
# Date: 2024.05.24
# This file contains the metrics used for evaluating model performance.
# The metrics are implemented following the scikit-learn convention.
# Author: Milton Gomez

# From SKLearn:
# Functions named as ``*_score`` return a scalar value to maximize: the higher
# the better.

# Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
# the lower the better.

# %% Imports
# OS and IO
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import dask.array as da
import pandas as pd
from scipy.special import erf
import torch

# Backend Libraries
import joblib as jl

from utils import toolbox, constants, ML_functions as mlf
from utils import data_lib as dlib

# %% Reference Dictionaries
short_names = {
    "root_mean_squared_error": "RMSE",
    "mean_squared_error": "MSE",
    "mean_absolute_error": "MAE",
    "mean_absolute_percentage_error": "MAPE",
    "mean_squared_logarithmic_error": "MSLE",
    "r2_score": "R2",
    "explained_variance_score": "EV",
    "max_error": "ME",
    "mean_poisson_deviance": "MPD",
    "mean_gamma_deviance": "MGD",
    "mean_tweedie_deviance": "MTD",
    "continuous_ranked_probability_score": "CRPS",
}

units = {"wind": "kts", "pressure": "hPa"}


# %% Base Class
class metric(object):
    """Base class for all metrics."""

    def __init__(self, name, function, unit=None, long_name=None, return_labels=None):
        self.name = name
        self.unit = unit
        self.function = function
        self.long_name = long_name
        self.return_labels = return_labels

    def __repr__(self):
        return f"{self.long_name if self.long_name else ''} {self.__class__.__name__}({self.name}, {self.unit})"

    def __str__(self):
        return f"{self.name}"

    def __call__(self, reference, predictions, **kwargs):
        """Compute the metric between the reference and predictions.
        Parameters
        ----------
        reference : pandas dataframe with shape (n_samples, n_features)
            The reference values. Needs to have the SID, lat, lon columns.
        predictions : pandas dataframe with shape (n_samples, n_features)
            The predicted values.
        """
        return self.function(reference, predictions, **kwargs)


def CRPS_ML(y_pred, y_true, **kwargs):
    """Compute the Continuous Ranked Probability Score (CRPS).

    The CRPS is a probabilistic metric that evaluates the accuracy of a
    probabilistic forecast. It is defined as the integral of the squared
    difference between the cumulative distribution function (CDF) of the
    forecast and the CDF of the observations.

    Parameters
    ----------

    y_pred : array-like of shape (n_samples, 2*n_features)
        The predicted probability parameters for each sample. The odd columns
        should contain the mean (mu), and the even columns should contain the
        standard deviation (sigma).

    y_true : array-like of shape (n_samples,n_features)

    Returns
    -------
    crps : float or array-like
        The CRPS value mean, or array of individual CRPS values.

    """
    # taken from https://github.com/WillyChap/ARML_Probabilistic/blob/main/Coastal_Points/Testing_and_Utility_Notebooks/CRPS_Verify.ipynb
    # and work by Louis Poulain--Auzeau (https://github.com/louisPoulain)
    reduction = kwargs.get("reduction", "mean")
    mu = y_pred[:, ::2]
    sigma = y_pred[:, 1::2]

    # prevent negative sigmas
    sigma = torch.sqrt(sigma.pow(2))
    loc = (y_true - mu) / sigma
    pdf = torch.exp(-0.5 * loc.pow(2)) / torch.sqrt(
        2 * torch.from_numpy(np.array(np.pi))
    )
    cdf = 0.5 * (1.0 + torch.erf(loc / torch.sqrt(torch.tensor(2.0))))

    # compute CRPS for each (input, truth) pair
    crps = sigma * (
        loc * (2.0 * cdf - 1.0)
        + 2.0 * pdf
        - 1.0 / torch.from_numpy(np.array(np.sqrt(np.pi)))
    )

    return crps.mean() if reduction == "mean" else crps


# def CRPSLoss(y_pred, y_true, **kwargs):
#     reduction = kwargs.get("reduction", "mean")
#     loss = 0.0
#     indiv_losses = np.empty((y_true.shape))

#     for i in range(0, y_true.shape[1]):
#         l = CRPS_ML(y_pred[:, i * 2 : (i + 1) * 2], y_true[:, i], reduction=reduction)
#         loss += l
#         indiv_losses[:, i] = l.detach().cpu().numpy()

#     return loss / (y_true.shape[1] // 2), indiv_losses


def CRPS_np(mu, sigma, y, **kwargs):
    reduction = kwargs.get("reduction", "mean")
    sigma = np.sqrt(sigma**2)
    loc = (y - mu) / sigma
    pdf = np.exp(-0.5 * loc**2) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1.0 + erf(loc / np.sqrt(2)))
    crps = sigma * (loc * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / np.sqrt(np.pi))
    return crps.mean() if reduction == "mean" else crps


def _DPE(reference, predictions, **kwargs):
    """Compute the Direct Position Error (DPE) between the reference and predictions.
    Parameters
    ----------
    reference : pandas dataframe with shape (n_samples, n_features)
        The reference values. Needs to have the SID, lat, lon columns.

    predictions : pandas dataframe with shape (n_samples, n_features)
        The predicted values."""

    # Check if a columns dictionary for the reference DataFrame is provided
    ref_col_dict = kwargs.get("ref_columns", None)
    if ref_col_dict is None:
        # Default columns dictionary
        ref_cols = {
            "SID": "SID",
            "lat": "LAT",
            "lon": "LON",
            "initial_time": "ISO_TIME",
        }
    else:
        ref_cols = ref_col_dict

    # Check if a columns dictionary for the predictions DataFrame is provided
    pred_col_dict = kwargs.get("pred_columns", None)
    if pred_col_dict is None:
        # Default columns dictionary
        pred_cols = {
            "SID": "SID",
            "lat": "lat",
            "lon": "lon",
            "valid_time": "Valid Time",
            "init_time": "Initial Time",
        }
    else:
        pred_cols = pred_col_dict

    # Check if the reference DataFrame contains the required columns
    for key in ref_cols.values():
        if key not in reference.columns:
            raise ValueError(f"Missing column '{key}' in reference DataFrame.")

    # Check that the predictions DataFrame contains the required columns
    for key in pred_cols.values():
        if key not in predictions.columns:
            raise ValueError(f"Missing column '{key}' in predictions DataFrame.")

    def storm_processor(ref, preds, storm_id):
        # Get the reference and prediction data for the current storm ID
        ref_data = ref[reference[ref_cols["SID"]] == storm_id].copy()
        pred_data = preds[predictions[pred_cols["SID"]] == storm_id].copy()

        # check if the predictions are probabilistic by searching the columns
        # for the word "ensemble"
        probabilistic = False
        ensemble_dim = None
        for col in pred_data.columns:
            if "ensemble" in col:
                probabilistic = True
                ensemble_dim = col
                break

        if not probabilistic:
            ## TODO - why did this even happen with Pangu/FCnet?
            # if initial time and valid time combination is repeated in prediction,
            # remove the duplicates
            pred_data = pred_data.drop_duplicates(
                subset=[pred_cols["valid_time"], pred_cols["init_time"]], keep="first"
            )

        # make the initial time and valid time columns datetime
        pred_data[pred_cols["valid_time"]] = pd.to_datetime(
            pred_data[pred_cols["valid_time"]]
        )
        pred_data[pred_cols["init_time"]] = pd.to_datetime(
            pred_data[pred_cols["init_time"]]
        )

        # if pred_data is empty, return empty dataframe with additional dpe column
        if pred_data.empty:
            pred_data["DPE"] = np.nan
            return pred_data[["DPE"]]

        for initial_time in pred_data[pred_cols["init_time"]].unique():

            pred_data_vals = pred_data[
                pd.to_datetime(pred_data[pred_cols["init_time"]]) == initial_time
            ].copy()
            # convert the init and valid time  to datetime
            pred_data_vals[pred_cols["valid_time"]] = pd.to_datetime(
                pred_data_vals[pred_cols["valid_time"]]
            )
            pred_data_vals[pred_cols["init_time"]] = pd.to_datetime(
                pred_data_vals[pred_cols["init_time"]]
            )

            if not probabilistic:
                try:
                    # Build a small reference table with exact timestamp matches
                    ref_subset = ref_data[
                        [ref_cols["initial_time"], ref_cols["lat"], ref_cols["lon"]]
                    ].copy()
                    ref_subset = ref_subset.rename(
                        columns={
                            ref_cols["initial_time"]: "Valid Time",
                            ref_cols["lat"]: "ref_lat",
                            ref_cols["lon"]: "ref_lon",
                        }
                    )
                    ref_subset["Valid Time"] = pd.to_datetime(
                        ref_subset["Valid Time"]
                    )  # ensure datetime

                    # Prepare left table from this init-time slice, carrying original indices
                    left = pred_data_vals[
                        [pred_cols["valid_time"], pred_cols["lat"], pred_cols["lon"]]
                    ].copy()
                    left = left.rename(
                        columns={
                            pred_cols["valid_time"]: "Valid Time",
                            pred_cols["lat"]: "pred_lat",
                            pred_cols["lon"]: "pred_lon",
                        }
                    )
                    left["orig_idx"] = pred_data_vals.index

                    # Left-merge preserves the order of prediction rows
                    merged = left.merge(
                        ref_subset, on="Valid Time", how="left", sort=False
                    )

                    dist = toolbox.haversine(
                        merged["ref_lat"].to_numpy(),
                        merged["ref_lon"].to_numpy(),
                        merged["pred_lat"].to_numpy(),
                        merged["pred_lon"].to_numpy(),
                    )

                    # Write back using the original indices of the prediction rows
                    pred_data.loc[merged["orig_idx"].to_numpy(), "DPE"] = dist
                except Exception as e:
                    print(
                        f"Error processing storm {storm_id} at initial time {initial_time}: {e}"
                    )
                    continue
            else:
                # Probabilistic: compute per ensemble member with a time merge, then write back by original index
                # Build reference table once per init_time
                ref_subset = ref_data[
                    [ref_cols["initial_time"], ref_cols["lat"], ref_cols["lon"]]
                ].copy()
                ref_subset = ref_subset.rename(
                    columns={
                        ref_cols["initial_time"]: "Valid Time",
                        ref_cols["lat"]: "ref_lat",
                        ref_cols["lon"]: "ref_lon",
                    }
                )
                ref_subset["Valid Time"] = pd.to_datetime(
                    ref_subset["Valid Time"]
                )  # ensure datetime

                for member in pred_data_vals[ensemble_dim].unique():
                    pred_data_vals_ens = pred_data_vals[
                        pred_data_vals[ensemble_dim] == member
                    ].copy()

                    # Prepare left table from this member slice, carrying original indices
                    left = pred_data_vals_ens[
                        [pred_cols["valid_time"], pred_cols["lat"], pred_cols["lon"]]
                    ].copy()
                    left = left.rename(
                        columns={
                            pred_cols["valid_time"]: "Valid Time",
                            pred_cols["lat"]: "pred_lat",
                            pred_cols["lon"]: "pred_lon",
                        }
                    )
                    left["orig_idx"] = pred_data_vals_ens.index

                    # Left-merge preserves the order of prediction rows (order-invariant join on time)
                    merged = left.merge(
                        ref_subset, on="Valid Time", how="left", sort=False
                    )

                    dist = toolbox.haversine(
                        merged["ref_lat"].to_numpy(),
                        merged["ref_lon"].to_numpy(),
                        merged["pred_lat"].to_numpy(),
                        merged["pred_lon"].to_numpy(),
                    )

                    # Write back using original indices of these member rows only
                    pred_data.loc[merged["orig_idx"].to_numpy(), "DPE"] = dist

        return pred_data[["DPE"]]

        # temp_df_locator = (temp_df["SID"] == storm_id) & np.isin(temp_df["Valid Time"], pred_data_vals_ens[pred_cols["valid_time"]]) & (temp_df["Initial Time"] == initial_time) & (temp_df[ensemble_dim] == member)

        # temp_df.loc[temp_df_locator, "DPE"] = dist

    # Build list of storms that appear in predictions and in reference
    pred_sids = predictions[pred_cols["SID"]].unique()
    ref_sids = reference[ref_cols["SID"]].unique()
    storm_list = [sid for sid in pred_sids if sid in ref_sids]

    # process the storms with joblib parallel (preserving each subset's original indices)
    results = jl.Parallel(n_jobs=6, prefer="threads")(
        jl.delayed(storm_processor)(reference, predictions, storm_id)
        for storm_id in storm_list
    )

    # Concatenate without resetting the index to preserve original row indices
    if len(results):
        temp_df = pd.concat(results, axis=0)
    else:
        temp_df = pd.DataFrame(columns=["DPE"])

    # Create an output series aligned to the caller's predictions order
    out = pd.Series(index=predictions.index, dtype=float, name="DPE")
    overlap_idx = temp_df.index.intersection(predictions.index)
    if len(overlap_idx):
        out.loc[overlap_idx] = temp_df.loc[overlap_idx, "DPE"].to_numpy()

    return out.to_numpy()


def _SE(reference, predictions, **kwargs):
    """Compute the Squared Error (SE) between the reference and predictions
    for the wind and minimum pressure.

    Parameters
    ----------
    reference : pandas dataframe with shape (n_samples, n_features)
        The reference values. Needs to have the SID, lat, lon columns.

    predictions : pandas dataframe with shape (n_samples, n_features)
        The predicted values."""
    # Check if a columns dictionary for the reference DataFrame is provided
    ref_col_dict = kwargs.get("ref_columns", None)
    if ref_col_dict is None:
        # Default columns dictionary
        ref_cols = {
            "SID": "SID",
            "wind": "USA_WIND",
            "pressure": "USA_PRES",
            "initial_time": "ISO_TIME",
        }
    else:
        ref_cols = ref_col_dict

    # Check if a columns dictionary for the predictions DataFrame is provided
    pred_col_dict = kwargs.get("pred_columns", None)
    if pred_col_dict is None:
        # Default columns dictionary
        pred_cols = {
            "SID": "SID",
            "wind": "wind max",
            "pressure": "pressure min",
            "valid_time": "Valid Time",
            "init_time": "Initial Time",
        }
    else:
        pred_cols = pred_col_dict

    # Check if the reference DataFrame contains the required columns
    for key in ref_cols.values():
        if key not in reference.columns:
            raise ValueError(f"Missing column '{key}' in reference DataFrame.")

    # Check that the predictions DataFrame contains the required columns
    for key in pred_cols.values():
        if key not in predictions.columns:
            raise ValueError(f"Missing column '{key}' in predictions DataFrame.")

    def storm_processor(ref, preds, storm_id):
        # Get the reference and prediction data for the current storm ID
        ref_data = ref[reference[ref_cols["SID"]] == storm_id].copy()
        pred_data = preds[predictions[pred_cols["SID"]] == storm_id].copy()

        # convert the wind and pressure columns to numeric
        ref_data[ref_cols["wind"]] = pd.to_numeric(
            ref_data[ref_cols["wind"]], errors="coerce"
        )
        ref_data[ref_cols["pressure"]] = pd.to_numeric(
            ref_data[ref_cols["pressure"]], errors="coerce"
        )
        pred_data[pred_cols["wind"]] = pd.to_numeric(
            pred_data[pred_cols["wind"]], errors="coerce"
        )
        pred_data[pred_cols["pressure"]] = pd.to_numeric(
            pred_data[pred_cols["pressure"]], errors="coerce"
        )

        # check if the predictions are probabilistic by searching the columns
        # for the word "ensemble"
        probabilistic = False
        ensemble_dim = None
        for col in pred_data.columns:
            if "ensemble" in col:
                probabilistic = True
                ensemble_dim = col
                break

        if not probabilistic:
            # if initial time and valid time combination is repeated in prediction, remove the duplicates
            pred_data = pred_data.drop_duplicates(
                subset=[pred_cols["valid_time"], pred_cols["init_time"]], keep="first"
            )

        # make the initial time and valid time columns datetime
        pred_data[pred_cols["valid_time"]] = pd.to_datetime(
            pred_data[pred_cols["valid_time"]]
        )
        pred_data[pred_cols["init_time"]] = pd.to_datetime(
            pred_data[pred_cols["init_time"]]
        )

        # if pred_data is empty, return only metric columns
        pred_data["SE_wind"] = np.nan
        pred_data["SE_pressure"] = np.nan

        if pred_data.empty:
            return pred_data[["SE_wind", "SE_pressure"]]

        for initial_time in pred_data[pred_cols["init_time"]].unique():
            pred_data_vals = pred_data[
                pd.to_datetime(pred_data[pred_cols["init_time"]]) == initial_time
            ].copy()
            pred_data_vals[pred_cols["valid_time"]] = pd.to_datetime(
                pred_data_vals[pred_cols["valid_time"]]
            )
            pred_data_vals[pred_cols["init_time"]] = pd.to_datetime(
                pred_data_vals[pred_cols["init_time"]]
            )

            if not probabilistic:
                try:
                    # Reference (time, wind, pressure) table
                    ref_subset = ref_data[
                        [
                            ref_cols["initial_time"],
                            ref_cols["wind"],
                            ref_cols["pressure"],
                        ]
                    ].copy()
                    ref_subset = ref_subset.rename(
                        columns={
                            ref_cols["initial_time"]: "Valid Time",
                            ref_cols["wind"]: "ref_wind",
                            ref_cols["pressure"]: "ref_pres",
                        }
                    )
                    ref_subset["Valid Time"] = pd.to_datetime(
                        ref_subset["Valid Time"]
                    )  # ensure datetime

                    # Left table from this init-time slice, keep original indices
                    left = pred_data_vals[
                        [
                            pred_cols["valid_time"],
                            pred_cols["wind"],
                            pred_cols["pressure"],
                        ]
                    ].copy()
                    left = left.rename(
                        columns={
                            pred_cols["valid_time"]: "Valid Time",
                            pred_cols["wind"]: "pred_wind",
                            pred_cols["pressure"]: "pred_pres",
                        }
                    )
                    left["orig_idx"] = pred_data_vals.index

                    # Merge on time (order-invariant)
                    merged = left.merge(
                        ref_subset, on="Valid Time", how="left", sort=False
                    )

                    # Coerce to float and handle blanks
                    for col in ["pred_wind", "pred_pres", "ref_wind", "ref_pres"]:
                        if col in merged:
                            merged[col] = pd.to_numeric(merged[col], errors="coerce")

                    se_wind = (merged["pred_wind"] - merged["ref_wind"]) ** 2
                    se_pres = (merged["pred_pres"] - merged["ref_pres"]) ** 2

                    pred_data.loc[merged["orig_idx"].to_numpy(), "SE_wind"] = (
                        se_wind.to_numpy()
                    )
                    pred_data.loc[merged["orig_idx"].to_numpy(), "SE_pressure"] = (
                        se_pres.to_numpy()
                    )
                except Exception as e:
                    print(
                        f"Error processing storm {storm_id} at initial time {initial_time}: {e}"
                    )
                    continue
            else:
                # Probabilistic: per-member merge on time, then write back by original index
                # Build reference (time, wind, pressure) once per init_time
                ref_subset = ref_data[
                    [ref_cols["initial_time"], ref_cols["wind"], ref_cols["pressure"]]
                ].copy()
                ref_subset = ref_subset.rename(
                    columns={
                        ref_cols["initial_time"]: "Valid Time",
                        ref_cols["wind"]: "ref_wind",
                        ref_cols["pressure"]: "ref_pres",
                    }
                )
                ref_subset["Valid Time"] = pd.to_datetime(
                    ref_subset["Valid Time"]
                )  # ensure datetime

                for member in pred_data_vals[ensemble_dim].unique():
                    pred_data_vals_ens = pred_data_vals[
                        pred_data_vals[ensemble_dim] == member
                    ].copy()

                    # Left table from this member slice, carry original indices
                    left = pred_data_vals_ens[
                        [
                            pred_cols["valid_time"],
                            pred_cols["wind"],
                            pred_cols["pressure"],
                        ]
                    ].copy()
                    left = left.rename(
                        columns={
                            pred_cols["valid_time"]: "Valid Time",
                            pred_cols["wind"]: "pred_wind",
                            pred_cols["pressure"]: "pred_pres",
                        }
                    )
                    left["orig_idx"] = pred_data_vals_ens.index

                    merged = left.merge(
                        ref_subset, on="Valid Time", how="left", sort=False
                    )

                    # Coerce numerics (turn blanks like " " into NaN)
                    for col in ["pred_wind", "pred_pres", "ref_wind", "ref_pres"]:
                        if col in merged:
                            merged[col] = pd.to_numeric(merged[col], errors="coerce")

                    se_wind = (merged["pred_wind"] - merged["ref_wind"]) ** 2
                    se_pres = (merged["pred_pres"] - merged["ref_pres"]) ** 2

                    # Write back using original indices of these member rows only
                    pred_data.loc[merged["orig_idx"].to_numpy(), "SE_wind"] = (
                        se_wind.to_numpy()
                    )
                    pred_data.loc[merged["orig_idx"].to_numpy(), "SE_pressure"] = (
                        se_pres.to_numpy()
                    )

        return pred_data[["SE_wind", "SE_pressure"]]

    # Limit to storms present in both predictions and reference (keeps work bounded)
    pred_sids = predictions[pred_cols["SID"]].unique()
    ref_sids = reference[ref_cols["SID"]].unique()
    storm_list = [sid for sid in pred_sids if sid in ref_sids]

    results = jl.Parallel(n_jobs=6, prefer="threads")(
        jl.delayed(storm_processor)(reference, predictions, storm_id)
        for storm_id in storm_list
    )

    if len(results):
        temp_df = pd.concat(results, axis=0)  # preserve original indices
    else:
        temp_df = pd.DataFrame(columns=["SE_wind", "SE_pressure"])

    # Align to caller's order
    out = pd.DataFrame(
        index=predictions.index, columns=["SE_wind", "SE_pressure"], dtype=float
    )
    overlap = temp_df.index.intersection(predictions.index)
    if len(overlap):
        out.loc[overlap, ["SE_wind", "SE_pressure"]] = temp_df.loc[
            overlap, ["SE_wind", "SE_pressure"]
        ].to_numpy()

    return out.to_numpy()

    # temp_df = predictions.copy()
    # temp_df["SE_wind"] = np.nan
    # temp_df["SE_pressure"] = np.nan
    # for storm_id in reference.SID.unique():
    #     # Get the reference and prediction data for the current storm ID
    #     ref_data = reference[reference[ref_cols["SID"]] == storm_id].copy()
    #     pred_data = predictions[predictions[pred_cols["SID"]] == storm_id]

    #     for val_time in pred_data[pred_cols["valid_time"]].unique():
    #         # Get the reference and prediction data for the current valid time
    #         ref_data_val = ref_data[ref_data[ref_cols["initial_time"]] == val_time]
    #         pred_data_val = pred_data[pred_data[pred_cols["valid_time"]] == val_time]

    #         # if the reference or prediction values are empty strings, set them to NaN
    #         ref_data_val.loc[:, ref_cols["wind"]] = ref_data_val[
    #             ref_cols["wind"]
    #         ].replace(" ", np.nan)
    #         ref_data_val.loc[:, ref_cols["pressure"]] = ref_data_val[
    #             ref_cols["pressure"]
    #         ].replace(" ", np.nan)
    #         pred_data_val.loc[:, pred_cols["wind"]] = pred_data_val[
    #             pred_cols["wind"]
    #         ].replace(" ", np.nan)
    #         pred_data_val.loc[:, pred_cols["pressure"]] = pred_data_val[
    #             pred_cols["pressure"]
    #         ].replace(" ", np.nan)

    #         if len(ref_data_val) == 0 or len(pred_data_val) == 0:
    #             continue
    #         # skip if all values are NaN
    #         if (
    #             ref_data_val[ref_cols["wind"]].isna().all()
    #             or pred_data_val[pred_cols["wind"]].isna().all()
    #         ):
    #             continue

    #         se_wind = (
    #             ref_data_val[ref_cols["wind"]].to_numpy().astype(float)
    #             - pred_data_val[pred_cols["wind"]].to_numpy().astype(float)
    #         ) ** 2
    #         se_pressure = (
    #             ref_data_val[ref_cols["pressure"]].to_numpy().astype(float)
    #             - pred_data_val[pred_cols["pressure"]].to_numpy().astype(float)
    #         ) ** 2
    #         temp_df.loc[pred_data_val.index, "SE_wind"] = np.sqrt(se_wind)
    #         temp_df.loc[pred_data_val.index, "SE_pressure"] = np.sqrt(se_pressure)

    # return temp_df[["SE_wind", "SE_pressure"]]


def _AE(reference, predictions, **kwargs):
    """Compute the Absolute Error (MAE) between the reference and predictions
    for the wind and minimum pressure.

    Parameters
    ----------
    reference : pandas dataframe with shape (n_samples, n_features)
        The reference values. Needs to have the SID, lat, lon columns.

    predictions : pandas dataframe with shape (n_samples, n_features)
        The predicted values."""
    # Check if a columns dictionary for the reference DataFrame is provided
    ref_col_dict = kwargs.get("ref_columns", None)
    if ref_col_dict is None:
        # Default columns dictionary
        ref_cols = {
            "SID": "SID",
            "wind": "USA_WIND",
            "pressure": "USA_PRES",
            "initial_time": "ISO_TIME",
        }
    else:
        ref_cols = ref_col_dict

    # Check if a columns dictionary for the predictions DataFrame is provided
    pred_col_dict = kwargs.get("pred_columns", None)
    if pred_col_dict is None:
        # Default columns dictionary
        pred_cols = {
            "SID": "SID",
            "wind": "wind max",
            "pressure": "pressure min",
            "valid_time": "Valid Time",
            "init_time": "Initial Time",
        }
    else:
        pred_cols = pred_col_dict

    # Check if the reference DataFrame contains the required columns
    for key in ref_cols.values():
        if key not in reference.columns:
            raise ValueError(f"Missing column '{key}' in reference DataFrame.")

    # Check that the predictions DataFrame contains the required columns
    for key in pred_cols.values():
        if key not in predictions.columns:
            raise ValueError(f"Missing column '{key}' in predictions DataFrame.")

    def storm_processor(ref, preds, storm_id):
        # Get the reference and prediction data for the current storm ID
        ref_data = ref[reference[ref_cols["SID"]] == storm_id].copy()
        pred_data = preds[predictions[pred_cols["SID"]] == storm_id].copy()

        # convert the wind and pressure columns to numeric
        ref_data[ref_cols["wind"]] = pd.to_numeric(
            ref_data[ref_cols["wind"]], errors="coerce"
        )
        ref_data[ref_cols["pressure"]] = pd.to_numeric(
            ref_data[ref_cols["pressure"]], errors="coerce"
        )
        pred_data[pred_cols["wind"]] = pd.to_numeric(
            pred_data[pred_cols["wind"]], errors="coerce"
        )
        pred_data[pred_cols["pressure"]] = pd.to_numeric(
            pred_data[pred_cols["pressure"]], errors="coerce"
        )

        # check if the predictions are probabilistic by searching the columns
        # for the word "ensemble"
        probabilistic = False
        ensemble_dim = None
        for col in pred_data.columns:
            if "ensemble" in col:
                probabilistic = True
                ensemble_dim = col
                break

        if not probabilistic:
            pred_data = pred_data.drop_duplicates(
                subset=[pred_cols["valid_time"], pred_cols["init_time"]], keep="first"
            )

        pred_data[pred_cols["valid_time"]] = pd.to_datetime(
            pred_data[pred_cols["valid_time"]]
        )
        pred_data[pred_cols["init_time"]] = pd.to_datetime(
            pred_data[pred_cols["init_time"]]
        )

        pred_data["AE_wind"] = np.nan
        pred_data["AE_pressure"] = np.nan

        if pred_data.empty:
            return pred_data[["AE_wind", "AE_pressure"]]

        for initial_time in pred_data[pred_cols["init_time"]].unique():
            pred_data_vals = pred_data[
                pd.to_datetime(pred_data[pred_cols["init_time"]]) == initial_time
            ].copy()
            pred_data_vals[pred_cols["valid_time"]] = pd.to_datetime(
                pred_data_vals[pred_cols["valid_time"]]
            )
            pred_data_vals[pred_cols["init_time"]] = pd.to_datetime(
                pred_data_vals[pred_cols["init_time"]]
            )

            if not probabilistic:
                try:
                    # Reference (time, wind, pressure)
                    ref_subset = ref_data[
                        [
                            ref_cols["initial_time"],
                            ref_cols["wind"],
                            ref_cols["pressure"],
                        ]
                    ].copy()
                    ref_subset = ref_subset.rename(
                        columns={
                            ref_cols["initial_time"]: "Valid Time",
                            ref_cols["wind"]: "ref_wind",
                            ref_cols["pressure"]: "ref_pres",
                        }
                    )
                    ref_subset["Valid Time"] = pd.to_datetime(
                        ref_subset["Valid Time"]
                    )  # ensure datetime

                    # Left table keeps original indices
                    left = pred_data_vals[
                        [
                            pred_cols["valid_time"],
                            pred_cols["wind"],
                            pred_cols["pressure"],
                        ]
                    ].copy()
                    left = left.rename(
                        columns={
                            pred_cols["valid_time"]: "Valid Time",
                            pred_cols["wind"]: "pred_wind",
                            pred_cols["pressure"]: "pred_pres",
                        }
                    )
                    left["orig_idx"] = pred_data_vals.index

                    merged = left.merge(
                        ref_subset, on="Valid Time", how="left", sort=False
                    )

                    for col in ["pred_wind", "pred_pres", "ref_wind", "ref_pres"]:
                        if col in merged:
                            merged[col] = pd.to_numeric(merged[col], errors="coerce")

                    if (
                        merged["pred_wind"].isna().all()
                        or merged["pred_pres"].isna().all()
                    ):
                        continue
                    if (
                        merged["ref_wind"].isna().all()
                        or merged["ref_pres"].isna().all()
                    ):
                        continue

                    ae_wind = (merged["pred_wind"] - merged["ref_wind"]).abs()
                    ae_pres = (merged["pred_pres"] - merged["ref_pres"]).abs()

                    pred_data.loc[merged["orig_idx"].to_numpy(), "AE_wind"] = (
                        ae_wind.to_numpy()
                    )
                    pred_data.loc[merged["orig_idx"].to_numpy(), "AE_pressure"] = (
                        ae_pres.to_numpy()
                    )
                except Exception as e:
                    print(
                        f"Error processing storm {storm_id} at initial time {initial_time}: {e}"
                    )
                    continue
            else:
                # Probabilistic: compute per ensemble member with a time merge, then write back by original index
                # Build reference (time, wind, pressure) once per init_time
                ref_subset = ref_data[
                    [ref_cols["initial_time"], ref_cols["wind"], ref_cols["pressure"]]
                ].copy()
                ref_subset = ref_subset.rename(
                    columns={
                        ref_cols["initial_time"]: "Valid Time",
                        ref_cols["wind"]: "ref_wind",
                        ref_cols["pressure"]: "ref_pres",
                    }
                )
                ref_subset["Valid Time"] = pd.to_datetime(
                    ref_subset["Valid Time"]
                )  # ensure datetime

                for member in pred_data_vals[ensemble_dim].unique():
                    pred_data_vals_ens = pred_data_vals[
                        pred_data_vals[ensemble_dim] == member
                    ].copy()

                    # Prepare left table from this member slice, carrying original indices
                    left = pred_data_vals_ens[
                        [
                            pred_cols["valid_time"],
                            pred_cols["wind"],
                            pred_cols["pressure"],
                        ]
                    ].copy()
                    left = left.rename(
                        columns={
                            pred_cols["valid_time"]: "Valid Time",
                            pred_cols["wind"]: "pred_wind",
                            pred_cols["pressure"]: "pred_pres",
                        }
                    )
                    left["orig_idx"] = pred_data_vals_ens.index

                    merged = left.merge(
                        ref_subset, on="Valid Time", how="left", sort=False
                    )

                    # numeric coercion
                    for col in ["pred_wind", "pred_pres", "ref_wind", "ref_pres"]:
                        if col in merged:
                            merged[col] = pd.to_numeric(merged[col], errors="coerce")

                    # skip if all wind and pressure values are NaN
                    if (
                        merged["pred_wind"].isna().all()
                        or merged["pred_pres"].isna().all()
                    ):
                        continue
                    if (
                        merged["ref_wind"].isna().all()
                        or merged["ref_pres"].isna().all()
                    ):
                        continue

                    ae_wind = (merged["pred_wind"] - merged["ref_wind"]).abs()
                    ae_pres = (merged["pred_pres"] - merged["ref_pres"]).abs()

                    # Write back for this member using the original indices
                    pred_data.loc[merged["orig_idx"].to_numpy(), "AE_wind"] = (
                        ae_wind.to_numpy()
                    )
                    pred_data.loc[merged["orig_idx"].to_numpy(), "AE_pressure"] = (
                        ae_pres.to_numpy()
                    )

        return pred_data[["AE_wind", "AE_pressure"]]

    pred_sids = predictions[pred_cols["SID"]].unique()
    ref_sids = reference[ref_cols["SID"]].unique()
    storm_list = [sid for sid in pred_sids if sid in ref_sids]

    results = jl.Parallel(n_jobs=6, prefer="threads")(
        jl.delayed(storm_processor)(reference, predictions, storm_id)
        for storm_id in storm_list
    )

    if len(results):
        temp_df = pd.concat(results, axis=0)
    else:
        temp_df = pd.DataFrame(columns=["AE_wind", "AE_pressure"])

    out = pd.DataFrame(
        index=predictions.index, columns=["AE_wind", "AE_pressure"], dtype=float
    )
    overlap = temp_df.index.intersection(predictions.index)
    if len(overlap):
        out.loc[overlap, ["AE_wind", "AE_pressure"]] = temp_df.loc[
            overlap, ["AE_wind", "AE_pressure"]
        ].to_numpy()

    return out.to_numpy()

    # temp_df = predictions.copy()
    # temp_df["AE_wind"] = np.nan
    # temp_df["AE_pressure"] = np.nan
    # for storm_id in reference.SID.unique():
    #     # Get the reference and prediction data for the current storm ID
    #     ref_data = reference[reference[ref_cols["SID"]] == storm_id]
    #     pred_data = predictions[predictions[pred_cols["SID"]] == storm_id]

    #     for val_time in pred_data[pred_cols["valid_time"]].unique():
    #         # Get the reference and prediction data for the current valid time
    #         ref_data_val = ref_data[ref_data[ref_cols["initial_time"]] == val_time]
    #         pred_data_val = pred_data[pred_data[pred_cols["valid_time"]] == val_time]

    #         # if the reference or prediction values are empty strings, set them to NaN
    #         ref_data_val.loc[:, ref_cols["wind"]] = ref_data_val[
    #             ref_cols["wind"]
    #         ].replace(" ", np.nan)
    #         ref_data_val.loc[:, ref_cols["pressure"]] = ref_data_val[
    #             ref_cols["pressure"]
    #         ].replace(" ", np.nan)
    #         pred_data_val.loc[:, pred_cols["wind"]] = pred_data_val[
    #             pred_cols["wind"]
    #         ].replace(" ", np.nan)
    #         pred_data_val.loc[:, pred_cols["pressure"]] = pred_data_val[
    #             pred_cols["pressure"]
    #         ].replace(" ", np.nan)

    #         if len(ref_data_val) == 0 or len(pred_data_val) == 0:
    #             continue
    #         # skip if all values are NaN
    #         if (
    #             ref_data_val[ref_cols["wind"]].isna().all()
    #             or pred_data_val[pred_cols["wind"]].isna().all()
    #         ):
    #             continue

    #         # if the reference or prediction values are empty strings, set them to NaN
    #         ref_data_val[ref_cols["wind"]] = ref_data_val[ref_cols["wind"]].replace(
    #             "", np.nan
    #         )
    #         ref_data_val[ref_cols["pressure"]] = ref_data_val[
    #             ref_cols["pressure"]
    #         ].replace("", np.nan)
    #         pred_data_val[pred_cols["wind"]] = pred_data_val[pred_cols["wind"]].replace(
    #             "", np.nan
    #         )
    #         pred_data_val[pred_cols["pressure"]] = pred_data_val[
    #             pred_cols["pressure"]
    #         ].replace("", np.nan)

    #         ae_wind = ref_data_val[ref_cols["wind"]].to_numpy().astype(
    #             float
    #         ) - pred_data_val[pred_cols["wind"]].to_numpy().astype(float)
    #         ae_pressure = ref_data_val[ref_cols["pressure"]].to_numpy().astype(
    #             float
    #         ) - pred_data_val[pred_cols["pressure"]].to_numpy().astype(float)
    #         temp_df.loc[pred_data_val.index, "AE_wind"] = np.abs(ae_wind)
    #         temp_df.loc[pred_data_val.index, "AE_pressure"] = np.abs(ae_pressure)

    # return temp_df[["AE_wind", "AE_pressure"]]


def _CARTE(reference, predictions, **kwargs):
    """Cartesian Along-Track / Cross-Track / Direct Positional Error.

    For each prediction at (SID, Valid Time), project the predicted point onto the
    IBTrACS reference segment defined by [Valid Time - 6h, Valid Time] and compute:
      - Along_TE: distance from the projection point to the segment endpoint at Valid Time
      - Cross_TE: perpendicular distance from prediction to the segment
      - DPE_cart: sqrt(Along_TE^2 + Cross_TE^2) in the same Cartesian metric

    Notes
    -----
    * Uses toolbox.latlon_2_cartesian to get a local Cartesian embedding (x,y).
    * Works for both deterministic and probabilistic predictions (per-member),
      writing results back by original row index. Deterministic duplicates for a
      given (SID, init, valid) are de-duplicated like DPE/AE/SE.
    * Returns an array aligned to `predictions.index` with columns ordered as
      ["Along_TE", "Cross_TE", "DPE_cart"].
    """
    # Column maps
    ref_col_dict = kwargs.get("ref_columns", None)
    if ref_col_dict is None:
        ref_cols = {
            "SID": "SID",
            "lat": "LAT",
            "lon": "LON",
            "initial_time": "ISO_TIME",
        }
    else:
        ref_cols = ref_col_dict

    pred_col_dict = kwargs.get("pred_columns", None)
    if pred_col_dict is None:
        pred_cols = {
            "SID": "SID",
            "lat": "lat",
            "lon": "lon",
            "valid_time": "Valid Time",
            "init_time": "Initial Time",
        }
    else:
        pred_cols = pred_col_dict

    # Column existence checks
    for key in ref_cols.values():
        if key not in reference.columns:
            raise ValueError(f"Missing column '{key}' in reference DataFrame.")
    for key in pred_cols.values():
        if key not in predictions.columns:
            raise ValueError(f"Missing column '{key}' in predictions DataFrame.")

    # Work on copies; ensure datetime
    ref = reference.copy()
    preds = predictions.copy()
    ref[ref_cols["initial_time"]] = pd.to_datetime(ref[ref_cols["initial_time"]])
    preds[pred_cols["valid_time"]] = pd.to_datetime(preds[pred_cols["valid_time"]])
    preds[pred_cols["init_time"]] = pd.to_datetime(preds[pred_cols["init_time"]])

    # Output scaffold aligned to caller
    out = pd.DataFrame(
        index=preds.index, columns=["Along_TE", "Cross_TE", "DPE_cart"], dtype=float
    )

    # Helper: numerically safe, NaN-aware, segment-projection (not infinite line)
    def compute_errors_cart(ref_start, ref_end, pred_slice):
        # Convert to Cartesian (float64, safe for NaNs)
        rsx, rsy = toolbox.latlon_2_cartesian(
            ref_start[ref_cols["lat"]].to_numpy(dtype=float),
            ref_start[ref_cols["lon"]].to_numpy(dtype=float),
        )
        rex, rey = toolbox.latlon_2_cartesian(
            ref_end[ref_cols["lat"]].to_numpy(dtype=float),
            ref_end[ref_cols["lon"]].to_numpy(dtype=float),
        )
        px, py = toolbox.latlon_2_cartesian(
            pred_slice[pred_cols["lat"]].to_numpy(dtype=float),
            pred_slice[pred_cols["lon"]].to_numpy(dtype=float),
        )

        # Ensure ndarray float64
        rsx = np.asarray(rsx, dtype=np.float64)
        rsy = np.asarray(rsy, dtype=np.float64)
        rex = np.asarray(rex, dtype=np.float64)
        rey = np.asarray(rey, dtype=np.float64)
        px = np.asarray(px, dtype=np.float64)
        py = np.asarray(py, dtype=np.float64)

        n = px.shape[0]
        # Default outputs
        ate = np.full(n, np.nan, dtype=np.float64)
        cte = np.full(n, np.nan, dtype=np.float64)
        cdpe = np.full(n, np.nan, dtype=np.float64)

        # Most slices have a single ref_start/ref_end row; handle that as scalars
        if rsx.size == 1 and rex.size == 1:
            Sx = float(rsx.reshape(-1)[0])
            Sy = float(rsy.reshape(-1)[0])
            Ex = float(rex.reshape(-1)[0])
            Ey = float(rey.reshape(-1)[0])
            dx = Ex - Sx
            dy = Ey - Sy
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                seg_norm2 = dx * dx + dy * dy
            if not np.isfinite(seg_norm2) or seg_norm2 <= 1e-12 or seg_norm2 > 1e20:
                return ate, cte, cdpe
            # projection parameter per point
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                t = ((px - Sx) * dx + (py - Sy) * dy) / seg_norm2
            t = np.clip(t, 0.0, 1.0)
            projx = Sx + t * dx
            projy = Sy + t * dy
            cte = np.sqrt((px - projx) ** 2 + (py - projy) ** 2)
            ate = np.sqrt((Ex - projx) ** 2 + (Ey - projy) ** 2)
            cdpe = np.sqrt(ate**2 + cte**2)
            return ate, cte, cdpe

        # General vectorized path (rare) — lengths must match
        # Broadcast to common shape via indexing mask of the segment array
        dx = rex - rsx
        dy = rey - rsy
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            seg_norm2 = dx * dx + dy * dy
        invalid = ~np.isfinite(seg_norm2) | (seg_norm2 <= 1e-12) | (seg_norm2 > 1e20)
        vm = ~invalid
        if not np.any(vm):
            return ate, cte, cdpe
        # For this path, require px/py to have same length as segment arrays
        if px.shape[0] != dx.shape[0]:
            # Fallback: compute pointwise using the first valid segment entry
            k = int(np.flatnonzero(vm)[0])
            Sx = float(rsx[k])
            Sy = float(rsy[k])
            Ex = float(rex[k])
            Ey = float(rey[k])
            dxk = Ex - Sx
            dyk = Ey - Sy
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                segn2 = dxk * dxk + dyk * dyk
            if not np.isfinite(segn2) or segn2 <= 1e-12 or segn2 > 1e20:
                return ate, cte, cdpe
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                t = ((px - Sx) * dxk + (py - Sy) * dyk) / segn2
            t = np.clip(t, 0.0, 1.0)
            projx = Sx + t * dxk
            projy = Sy + t * dyk
            cte = np.sqrt((px - projx) ** 2 + (py - projy) ** 2)
            ate = np.sqrt((Ex - projx) ** 2 + (Ey - projy) ** 2)
            cdpe = np.sqrt(ate**2 + cte**2)
            return ate, cte, cdpe

        # Fully vectorized case (matching lengths)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            t = ((px - rsx) * dx + (py - rsy) * dy) / seg_norm2
        t = np.clip(t, 0.0, 1.0)
        projx = rsx + t * dx
        projy = rsy + t * dy
        cte = np.sqrt((px - projx) ** 2 + (py - projy) ** 2)
        ate = np.sqrt((rex - projx) ** 2 + (rey - projy) ** 2)
        cdpe = np.sqrt(ate**2 + cte**2)
        return ate, cte, cdpe

    # Iterate storms present in both
    pred_sids = preds[pred_cols["SID"]].unique()
    ref_sids = ref[ref_cols["SID"]].unique()
    storm_list = [sid for sid in pred_sids if sid in ref_sids]

    def process_one(storm_id):
        ref_data = ref[ref[ref_cols["SID"]] == storm_id]
        pred_data = preds[preds[pred_cols["SID"]] == storm_id].copy()

        # Probabilistic detection
        probabilistic = False
        ensemble_dim = None
        for c in pred_data.columns:
            if "ensemble" in c:
                probabilistic = True
                ensemble_dim = c
                break
        if not probabilistic:
            pred_data = pred_data.drop_duplicates(
                subset=[pred_cols["valid_time"], pred_cols["init_time"]], keep="first"
            )

        # Pre-allocate
        pred_data["Along_TE"] = np.nan
        pred_data["Cross_TE"] = np.nan
        pred_data["DPE_cart"] = np.nan

        # Loop by valid time (and per member if probabilistic)
        for val_time in pred_data[pred_cols["valid_time"]].dropna().unique():
            ref_end = ref_data[ref_data[ref_cols["initial_time"]] == val_time]
            ref_start = ref_data[
                ref_data[ref_cols["initial_time"]]
                == (pd.to_datetime(val_time) - pd.Timedelta(hours=6))
            ]
            if ref_start.empty or ref_end.empty:
                continue
            # Drop any rows with non-finite coords in the segment endpoints
            if (
                not np.isfinite(ref_start[ref_cols["lat"]]).all()
                or not np.isfinite(ref_start[ref_cols["lon"]]).all()
            ):
                continue
            if (
                not np.isfinite(ref_end[ref_cols["lat"]]).all()
                or not np.isfinite(ref_end[ref_cols["lon"]]).all()
            ):
                continue

            slice_all = pred_data[pred_data[pred_cols["valid_time"]] == val_time]
            if not probabilistic:
                ate, cte, cdpe = compute_errors_cart(ref_start, ref_end, slice_all)
                pred_data.loc[slice_all.index, ["Along_TE", "Cross_TE", "DPE_cart"]] = (
                    np.vstack([ate, cte, cdpe]).T
                )
            else:
                for member in slice_all[ensemble_dim].dropna().unique():
                    mslice = slice_all[slice_all[ensemble_dim] == member]
                    if mslice.empty:
                        continue
                    ate, cte, cdpe = compute_errors_cart(ref_start, ref_end, mslice)
                    pred_data.loc[
                        mslice.index, ["Along_TE", "Cross_TE", "DPE_cart"]
                    ] = np.vstack([ate, cte, cdpe]).T

        # Robust: if columns were not set for some rows, reindex creates them with NaN instead of raising
        return pred_data.reindex(columns=["Along_TE", "Cross_TE", "DPE_cart"])

    # Parallel per storm
    results = jl.Parallel(n_jobs=6, prefer="threads")(
        jl.delayed(process_one)(sid) for sid in storm_list
    )
    temp_df = (
        pd.concat(results, axis=0)
        if len(results)
        else pd.DataFrame(columns=["Along_TE", "Cross_TE", "DPE_cart"])
    )

    # Align to caller order
    overlap = temp_df.index.intersection(preds.index)
    if len(overlap):
        out.loc[overlap, ["Along_TE", "Cross_TE", "DPE_cart"]] = temp_df.loc[
            overlap, ["Along_TE", "Cross_TE", "DPE_cart"]
        ].to_numpy()

    return out.to_numpy()


def _FCRPS(reference, predictions, **kwargs):
    """Compute the Fair CRPS between the reference and predictions using the kernel
    representation of the CRPS. Wind max and pressure min are assumed targets.

    Reference:
    Leutbecher, M. (2019). Ensemble size: How suboptimal is less than infinity?,
    QJ Roy. Meteor. Soc., 145, 107–128.


    Parameters
    ----------
    reference : pandas dataframe with shape (n_samples, n_features)
        The reference values. Needs to have the SID and columns with the variables
        to evaluate - default is `USA_WIND` and `USA_PRES`.

    predictions : pandas dataframe with shape (n_samples, n_features)
        The predicted values.
    """
    # Column maps
    ref_col_dict = kwargs.get("ref_columns", None)
    if ref_col_dict is None:
        ref_cols = {
            "SID": "SID",
            "wind": "USA_WIND",
            "pressure": "USA_PRES",
            "initial_time": "ISO_TIME",
        }
    else:
        ref_cols = ref_col_dict

    pred_col_dict = kwargs.get("pred_columns", None)
    if pred_col_dict is None:
        pred_cols = {
            "SID": "SID",
            "wind": "wind max",
            "pressure": "pressure min",
            "valid_time": "Valid Time",
            "init_time": "Initial Time",
            "ensemble_idx": "ensemble_idx",
        }
    else:
        pred_cols = pred_col_dict

    # Column existence checks
    for key in ref_cols.values():
        if key not in reference.columns:
            raise ValueError(f"Missing column '{key}' in reference DataFrame.")
    for key in pred_cols.values():
        if key not in predictions.columns:
            raise ValueError(f"Missing column '{key}' in predictions DataFrame.")

    # Ensure numeric types
    reference = reference.copy()
    predictions = predictions.copy()
    reference.loc[:, ref_cols["wind"]] = pd.to_numeric(
        reference[ref_cols["wind"]], errors="coerce"
    )
    reference.loc[:, ref_cols["pressure"]] = pd.to_numeric(
        reference[ref_cols["pressure"]], errors="coerce"
    )
    predictions.loc[:, pred_cols["wind"]] = pd.to_numeric(
        predictions[pred_cols["wind"]], errors="coerce"
    )
    predictions.loc[:, pred_cols["pressure"]] = pd.to_numeric(
        predictions[pred_cols["pressure"]], errors="coerce"
    )

    predictions.loc[:, pred_cols["valid_time"]] = pd.to_datetime(
        predictions[pred_cols["valid_time"]]
    )
    predictions.loc[:, pred_cols["init_time"]] = pd.to_datetime(
        predictions[pred_cols["init_time"]]
    )
    reference.loc[:, ref_cols["initial_time"]] = pd.to_datetime(
        reference[ref_cols["initial_time"]]
    )

    # Output aligned to caller's order
    out = pd.DataFrame(
        index=predictions.index, columns=["CRPS_vmax", "CRPS_pmin"], dtype=float
    )

    # Iterate storms present in both
    pred_sids = predictions[pred_cols["SID"]].unique()
    ref_sids = reference[ref_cols["SID"]].unique()
    storm_list = [sid for sid in pred_sids if sid in ref_sids]

    for storm_id in storm_list:
        ref_data = reference[reference[ref_cols["SID"]] == storm_id]
        pred_data = predictions[predictions[pred_cols["SID"]] == storm_id]

        for val_time in pred_data[pred_cols["valid_time"]].dropna().unique():
            ref_data_val = ref_data[ref_data[ref_cols["initial_time"]] == val_time]
            pred_data_val = pred_data[pred_data[pred_cols["valid_time"]] == val_time]
            if ref_data_val.empty or pred_data_val.empty:
                continue

            for init_time in pred_data_val[pred_cols["init_time"]].dropna().unique():
                ensemble_vals = pred_data_val[
                    pred_data_val[pred_cols["init_time"]] == init_time
                ]
                num_members = len(ensemble_vals)
                if num_members < 2:
                    continue

                ref_wind = ref_data_val[ref_cols["wind"]].to_numpy().astype(np.float32)
                ref_pres = (
                    ref_data_val[ref_cols["pressure"]].to_numpy().astype(np.float32)
                )

                ens_wind = ensemble_vals[pred_cols["wind"]].to_numpy()
                ens_pres = ensemble_vals[pred_cols["pressure"]].to_numpy()

                # CRPS for wind
                diff_wind = np.abs(ref_wind - ens_wind)
                diff_matrix_wind = np.abs(ens_wind[:, None] - ens_wind[None, :])
                crps_wind = diff_wind.mean() - diff_matrix_wind.sum() / (
                    2 * num_members * (num_members - 1)
                )

                # CRPS for pressure
                diff_pres = np.abs(ref_pres - ens_pres)
                diff_matrix_pres = np.abs(ens_pres[:, None] - ens_pres[None, :])
                crps_pres = diff_pres.mean() - diff_matrix_pres.sum() / (
                    2 * num_members * (num_members - 1)
                )

                # Assign same CRPS values to all member rows for this (SID, valid, init)
                out.loc[ensemble_vals.index, "CRPS_vmax"] = crps_wind
                out.loc[ensemble_vals.index, "CRPS_pmin"] = crps_pres

    return out.to_numpy()


def _HCRPS(reference, predictions, **kwargs):
    """Compute the Haversinial, Fair CRPS between the reference and predictions using the
    kernel representation of the CRPS. lat and lon are required in the predictions.
    Haversinial distance is used instead of the Euclidean distance to calculate the CRPS.

    Reference:
    Leutbecher, M. (2019). Ensemble size: How suboptimal is less than infinity?,
    QJ Roy. Meteor. Soc., 145, 107–128.

    Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction,
    and estimation. Journal of the American statistical Association, 102(477), 359-378.

    Parameters
    ----------
    reference : pandas dataframe with shape (n_samples, n_features)
        The reference values. Needs to have the SID and columns with the variables
        to evaluate - default is `LAT` and `LON`.

    predictions : pandas dataframe with shape (n_samples, n_features)
        The predicted values.
    """
    # Column maps
    ref_col_dict = kwargs.get("ref_columns", None)
    if ref_col_dict is None:
        ref_cols = {
            "SID": "SID",
            "lat": "LAT",
            "lon": "LON",
            "initial_time": "ISO_TIME",
        }
    else:
        ref_cols = ref_col_dict

    pred_col_dict = kwargs.get("pred_columns", None)
    if pred_col_dict is None:
        pred_cols = {
            "SID": "SID",
            "lat": "lat",
            "lon": "lon",
            "valid_time": "Valid Time",
            "init_time": "Initial Time",
            "ensemble_idx": "ensemble_idx",
        }
    else:
        pred_cols = pred_col_dict

    # Column existence checks
    for key in ref_cols.values():
        if key not in reference.columns:
            raise ValueError(f"Missing column '{key}' in reference DataFrame.")
    for key in pred_cols.values():
        if key not in predictions.columns:
            raise ValueError(f"Missing column '{key}' in predictions DataFrame.")

    # Ensure numeric and datetime types
    reference = reference.copy()
    predictions = predictions.copy()

    reference.loc[:, ref_cols["lat"]] = pd.to_numeric(
        reference[ref_cols["lat"]], errors="coerce"
    )
    reference.loc[:, ref_cols["lon"]] = pd.to_numeric(
        reference[ref_cols["lon"]], errors="coerce"
    )
    predictions.loc[:, pred_cols["lat"]] = pd.to_numeric(
        predictions[pred_cols["lat"]], errors="coerce"
    )
    predictions.loc[:, pred_cols["lon"]] = pd.to_numeric(
        predictions[pred_cols["lon"]], errors="coerce"
    )

    reference.loc[:, ref_cols["initial_time"]] = pd.to_datetime(
        reference[ref_cols["initial_time"]]
    )
    predictions.loc[:, pred_cols["valid_time"]] = pd.to_datetime(
        predictions[pred_cols["valid_time"]]
    )
    predictions.loc[:, pred_cols["init_time"]] = pd.to_datetime(
        predictions[pred_cols["init_time"]]
    )

    # Output aligned to caller's row order
    out = pd.Series(index=predictions.index, dtype=float, name="CRPS_haversine")

    # Iterate storms present in both
    pred_sids = predictions[pred_cols["SID"]].unique()
    ref_sids = reference[ref_cols["SID"]].unique()
    storm_list = [sid for sid in pred_sids if sid in ref_sids]

    for storm_id in storm_list:
        ref_data = reference[reference[ref_cols["SID"]] == storm_id]
        pred_data = predictions[predictions[pred_cols["SID"]] == storm_id]
        if pred_data.empty:
            continue

        for val_time in pred_data[pred_cols["valid_time"]].dropna().unique():
            # Reference at this valid time (IBTrACS uses initial_time timestamps)
            ref_data_val = ref_data[ref_data[ref_cols["initial_time"]] == val_time]
            pred_data_val = pred_data[pred_data[pred_cols["valid_time"]] == val_time]
            if ref_data_val.empty or pred_data_val.empty:
                continue

            for init_time in pred_data_val[pred_cols["init_time"]].dropna().unique():
                ensemble_vals = pred_data_val[
                    pred_data_val[pred_cols["init_time"]] == init_time
                ]
                num_members = len(ensemble_vals)
                if num_members < 2:
                    # Fair CRPS requires at least 2 members to avoid division by zero
                    continue

                # Reference lat/lon (take as arrays; broadcasting works)
                ref_lat = ref_data_val[ref_cols["lat"]].to_numpy().astype(np.float32)
                ref_lon = ref_data_val[ref_cols["lon"]].to_numpy().astype(np.float32)

                ens_lat = ensemble_vals[pred_cols["lat"]].to_numpy()
                ens_lon = ensemble_vals[pred_cols["lon"]].to_numpy()

                # Track-to-truth distances for each member
                diff_track = toolbox.haversine(ref_lat, ref_lon, ens_lat, ens_lon)

                # Pairwise distances among ensemble members
                diff_matrix = toolbox.haversine(
                    ens_lat[:, None],
                    ens_lon[:, None],
                    ens_lat[None, :],
                    ens_lon[None, :],
                )

                haver_crps = diff_track.mean() - diff_matrix.sum() / (
                    2 * num_members * (num_members - 1)
                )

                # Assign the same CRPS value to all member rows for this (SID, valid, init)
                out.loc[ensemble_vals.index] = haver_crps

    return out.to_numpy()


FCRPS = metric(
    name="FCRPS",
    function=_FCRPS,
    long_name="Fair CRPS",
    return_labels=["CRPS_vmax", "CRPS_pmin"],
)

HCRPS = metric(
    name="HCRPS",
    function=_HCRPS,
    long_name="Haversinial CRPS",
    return_labels=["CRPS_haversine"],
)

DPE = metric(
    name="DPE",
    function=_DPE,
    long_name="Direct Position Error (Great Circle)",
    return_labels=["DPE_GCD"],
)

CARTE = metric(
    name="CARTE",
    function=_CARTE,
    long_name="Cartesian Cross-Track / Along-Track / Direct Positional Error",
    return_labels=["Along_TE", "Cross_TE", "DPE_cart"],
)

SE = metric(
    name="SE",
    function=_SE,
    unit="kts",
    long_name="Squared Error",
    return_labels=["SE_wind", "SE_pressure"],
)

AE = metric(
    name="AE",
    function=_AE,
    long_name="Absolute Error",
    return_labels=["AE_wind", "AE_pressure"],
)

# Required by evaluate_tracks to name outputs and to compute *_mean/_std for ensembles
try:
    DPE.return_labels = ["DPE_GCD"]
except Exception:
    pass
try:
    AE.return_labels = ["AE_wind", "AE_pressure"]
except Exception:
    pass
try:
    SE.return_labels = ["SE_wind", "SE_pressure"]
except Exception:
    pass
try:
    FCRPS.return_labels = ["CRPS_vmax", "CRPS_pmin"]
except Exception:
    pass
try:
    HCRPS.return_labels = ["CRPS_haversine"]
except Exception:
    pass

# %%
