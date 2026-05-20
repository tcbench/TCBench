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


# # %% Utilities
# def _check_regression(y_true, y_pred):
#     """Check that the regression inputs are of the correct form."""
#     raise NotImplementedError
#     assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape."
#     # assert y_true.ndim == 1, "y_true and y_pred must be 1D arrays."
#     # assert y_pred.ndim == 1, "y_true and y_pred must be 1D arrays."
#     return y_true, y_pred


# def _check_classification(y_true, y_pred):
#     """Check that the classification inputs are of the correct form."""
#     raise NotImplementedError
#     assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape."
#     # assert y_true.ndim == 1, "y_true and y_pred must be 1D arrays."
#     # assert y_pred.ndim == 1, "y_true and y_pred must be 1D arrays."
#     return y_true, y_pred


# # %% Metrics


# # def CRPS(y_true, y_pred):
# #     """Compute the Continuous Ranked Probability Score (CRPS).

# #     The CRPS is a probabilistic metric that evaluates the accuracy of a
# #     probabilistic forecast. It is defined as the integral of the squared
# #     difference between the cumulative distribution function (CDF) of the
# #     forecast and the CDF of the observations.

# #     Parameters
# #     ----------
# #     y_true : array-like of shape (n_samples,)
# #         The true target values.

# #     y_pred : array-like of shape (n_samples, n_classes)
# #         The predicted probabilities for each class.

# #     Returns
# #     -------
# #     crps : float
# #         The CRPS value.
# #     """
# #     # Check that the inputs are of the correct form
# #     y_true, y_pred = _check_classification(y_true, y_pred)

# #     # Compute the CRPS
# #     raise NotImplementedError
# #     return crps


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


# def CRPSNumpy(mu_pred, sigma_pred, y_true, **kwargs):
#     reduction = kwargs.get("reduction", "mean")
#     indiv_losses = np.empty((y_true.shape))

#     for i in range(0, y_true.shape[1]):
#         l = CRPS_np(mu_pred[:, i], sigma_pred[:, i], y_true[:, i], reduction=reduction)
#         indiv_losses[:, i] = l

#     return indiv_losses


# def summarize_performance(y_true, y_pred, y_baseline, metrics: list, **kwargs):
#     """Summarize the performance of the model and the baseline.

#     Parameters
#     ----------
#     y_true : array-like of shape (n_samples,)
#         The true target values.

#     y_pred : array-like of shape (n_samples,)
#         The predicted target values.

#     y_baseline : array-like of shape (n_samples,)
#         The baseline target values.

#     metrics : list of functions
#         The list of metrics to compute.

#     Returns
#     -------
#     performance : dict
#         A dictionary containing the performance metrics.
#     """
#     # # Check that the inputs are of the correct form
#     # y_true, y_pred = _check_regression(y_true, y_pred)
#     # y_true, y_baseline = _check_regression(y_true, y_baseline)

#     # Assert that the predictions have the shape (n_samples, n_features)
#     assert y_pred.ndim == 2, "y_pred must have shape (n_samples, n_features)"

#     y_labels = kwargs.get("y_labels", {0: "Wind", 1: "Pressure"})

#     # Compute the performance metrics
#     performance = {}
#     for metric in metrics:
#         for i in range(y_pred.shape[1]):
#             y_true_i = y_true[:, i]
#             y_pred_i = y_pred[:, i]
#             y_baseline_i = y_baseline[:, i]

#             if metric.__name__ in short_names.keys():
#                 metric_name = short_names[metric.__name__]
#             else:
#                 metric_name = metric.__name__

#             performance[f"{metric_name}_{y_labels[i]}"] = metric(y_true_i, y_pred_i)
#             performance[f"{metric_name}_{y_labels[i]}_baseline"] = metric(
#                 y_true_i, y_baseline_i
#             )

#             if kwargs.get("skill", True):
#                 performance[f"{metric_name}_{y_labels[i]}_skill"] = (
#                     1
#                     - performance[f"{metric_name}_{y_labels[i]}"]
#                     / performance[f"{metric_name}_{y_labels[i]}_baseline"]
#                 )

#     return performance


# # %%
# def plot_performance(metrics: dict, ax, **kwargs):
#     """Plot the performance metrics.

#     Parameters
#     ----------
#     metrics : dict
#         A dictionary containing the performance metrics.

#     ax : matplotlib.axes.Axes
#         The axes to plot the metrics on.

#     Returns
#     -------
#     None
#     """
#     if kwargs.get("skill", True):
#         assert isinstance(
#             ax, np.ndarray
#         ), "ax should be a numpy array if skill=True (default)"

#         ax1 = ax[0]
#         ax2 = ax[1]
#     else:
#         ax1 = ax

#     model_name = kwargs.get("model_name", "Model")
#     baseline_name = kwargs.get("baseline_name", "Baseline")

#     # Generate a list of unique metrics
#     metric_names = []
#     for metric in metrics.keys():
#         if not ("skill" in metric or "baseline" in metric):
#             metric_names.append(metric)
#     metric_names = np.unique(metric_names)

#     # Define the colors for the bars
#     colors = kwargs.get("colors", plt.cm.tab20.colors)

#     ax1_labels = []
#     # Plot the performance metrics
#     for i, metric in enumerate(metric_names):
#         model_metric = metrics[metric]
#         baseline_metric = metrics[f"{metric}_baseline"]
#         var = metric.split("_")[-1].lower()
#         unit = units.get(var, "")
#         ax1.bar(
#             [i * 3],
#             [model_metric],
#             color=colors[i % len(colors)],
#             label=f"{model_name} {metric} ({unit})",
#             hatch=kwargs.get("model_hatch", None),
#         )

#         ax1.bar(
#             [i * 3 + 1],
#             [baseline_metric],
#             color=colors[i % len(colors)],
#             label=f"{baseline_name} {metric} ({unit})",
#             hatch=kwargs.get("baseline_hatch", "//"),
#         )

#         ax1.set_ylabel("Score")
#         ax1.set_title("Performance Metrics")

#         ax1_labels += [f"{model_name} metric", f"{baseline_name} {metric}", ""]

#         if kwargs.get("skill", True):
#             ax2.bar(
#                 [i],
#                 [metrics[f"{metric}_skill"]],
#                 color=colors[i % len(colors)],
#                 label=f"{metric} Skill Score",
#                 hatch=kwargs.get("skill_hatch", None),
#             )
#             ax2.set_ylabel("Skill Score (1 - model/baseline)")
#             ax2.set_title("Skill Scores")

#     ax1.set_xticks(range(len(metric_names) * 3))
#     ax1.set_xticklabels([""] * len(ax1_labels))
#     ax1.legend(loc="lower right", framealpha=0.5)

#     if kwargs.get("skill", True):
#         ax2.set_xticks(range(len(metric_names)))
#         ax2.set_xticklabels([""] * len(metric_names))
#         ax2.legend(loc="lower right", framealpha=0.5)


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
            return pred_data

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
                    val_times = pd.to_datetime(
                        pred_data_vals[pred_cols["valid_time"]]
                    ).to_numpy()
                    ref_times = ref_data[ref_cols["initial_time"]].to_numpy()
                    bool_mask = np.isin(ref_times, val_times)
                    ref_times = ref_times[bool_mask]

                    pred_data_vals.loc[:, "ref_lat"] = np.nan
                    pred_data_vals.loc[:, "ref_lon"] = np.nan

                    locator = np.isin(
                        pd.to_datetime(
                            pred_data_vals[pred_cols["valid_time"]]
                        ).to_numpy(),
                        ref_times,
                    )

                    pred_data_vals.loc[locator, "ref_lat"] = ref_data[
                        ref_cols["lat"]
                    ].to_numpy()[bool_mask]
                    pred_data_vals.loc[locator, "ref_lon"] = ref_data[
                        ref_cols["lon"]
                    ].to_numpy()[bool_mask]
                    dist = toolbox.haversine(
                        pred_data_vals["ref_lat"].to_numpy(),
                        pred_data_vals["ref_lon"].to_numpy(),
                        pred_data_vals[pred_cols["lat"]].to_numpy(),
                        pred_data_vals[pred_cols["lon"]].to_numpy(),
                    )
                    pred_data_locator = (
                        (pred_data[pred_cols["SID"]] == storm_id)
                        & np.isin(
                            pred_data[pred_cols["valid_time"]],
                            pred_data_vals[pred_cols["valid_time"]],
                        )
                        & (pred_data[pred_cols["init_time"]] == initial_time)
                    )
                    pred_data.loc[pred_data_locator, "DPE"] = dist
                except Exception as e:
                    print(
                        f"Error processing storm {storm_id} at initial time {initial_time}: {e}"
                    )
                    continue
            else:
                for member in pred_data_vals[ensemble_dim].unique():
                    pred_data_vals_ens = pred_data_vals[
                        pred_data_vals[ensemble_dim] == member
                    ].copy()
                    val_times = pd.to_datetime(
                        pred_data_vals_ens[pred_cols["valid_time"]]
                    ).to_numpy()
                    ref_times = ref_data[ref_cols["initial_time"]].to_numpy()
                    bool_mask = np.isin(ref_times, val_times)
                    ref_times = ref_times[bool_mask]

                    pred_data_vals_ens.loc[:, "ref_lat"] = np.nan
                    pred_data_vals_ens.loc[:, "ref_lon"] = np.nan

                    locator = np.isin(
                        pd.to_datetime(
                            pred_data_vals_ens[pred_cols["valid_time"]]
                        ).to_numpy(),
                        ref_times,
                    )

                    pred_data_vals_ens.loc[locator, "ref_lat"] = ref_data[
                        ref_cols["lat"]
                    ].to_numpy()[bool_mask]
                    pred_data_vals_ens.loc[locator, "ref_lon"] = ref_data[
                        ref_cols["lon"]
                    ].to_numpy()[bool_mask]

                    dist = toolbox.haversine(
                        pred_data_vals_ens["ref_lat"].to_numpy(),
                        pred_data_vals_ens["ref_lon"].to_numpy(),
                        pred_data_vals_ens[pred_cols["lat"]].to_numpy(),
                        pred_data_vals_ens[pred_cols["lon"]].to_numpy(),
                    )

                    pred_data_locator = (
                        (pred_data[pred_cols["SID"]] == storm_id)
                        & np.isin(
                            pred_data[pred_cols["valid_time"]],
                            pred_data_vals_ens[pred_cols["valid_time"]],
                        )
                        & (pred_data[ensemble_dim] == member)
                        & (pred_data[pred_cols["init_time"]] == initial_time)
                    )
                    pred_data.loc[pred_data_locator, "DPE"] = dist

        return pred_data

        # temp_df_locator = (temp_df["SID"] == storm_id) & np.isin(temp_df["Valid Time"], pred_data_vals_ens[pred_cols["valid_time"]]) & (temp_df["Initial Time"] == initial_time) & (temp_df[ensemble_dim] == member)

        # temp_df.loc[temp_df_locator, "DPE"] = dist

    # process the storms with joblib parallel
    results = jl.Parallel(n_jobs=6)(
        jl.delayed(storm_processor)(reference, predictions, storm_id)
        for storm_id in reference.SID.unique()
    )

    # Combine the results into a single DataFrame
    temp_df = pd.concat(results, ignore_index=True)

    # for val_time in pred_data[pred_cols["valid_time"]].unique():
    #     # Get the reference and prediction data for the current valid time
    #     ref_data_val = ref_data[ref_data[ref_cols["initial_time"]] == val_time]
    #     pred_data_val = pred_data[pred_data[pred_cols["valid_time"]] == val_time]

    #     # if the reference or prediction values are empty strings, set them to NaN
    #     ref_data_val.loc[:, ref_cols["lat"]] = ref_data_val[
    #         ref_cols["lat"]
    #     ].replace(" ", np.nan)
    #     ref_data_val.loc[:, ref_cols["lon"]] = ref_data_val[
    #         ref_cols["lon"]
    #     ].replace(" ", np.nan)
    #     pred_data_val.loc[:, pred_cols["lat"]] = pred_data_val[
    #         pred_cols["lat"]
    #     ].replace(" ", np.nan)
    #     pred_data_val.loc[:, pred_cols["lon"]] = pred_data_val[
    #         pred_cols["lon"]
    #     ].replace(" ", np.nan)

    #     if len(ref_data_val) == 0 or len(pred_data_val) == 0:
    #         continue

    #     # skip if all values are NaN
    #     if (
    #         ref_data_val[ref_cols["lat"]].isna().all()
    #         or pred_data_val[pred_cols["lat"]].isna().all()
    #     ):
    #         continue

    # distances = toolbox.haversine(
    #     ref_data_val[ref_cols["lat"]].to_numpy(),
    #     ref_data_val[ref_cols["lon"]].to_numpy(),
    #     pred_data_val[pred_cols["lat"]].to_numpy(),
    #     pred_data_val[pred_cols["lon"]].to_numpy(),
    # )

    # locator = (temp_df["SID"] == storm_id) & (temp_df["Valid Time"] == val_time)
    # temp_df.loc[locator, "DPE"] = distances

    #     pred_data.loc[pred_data_val.index, "DPE"] = distances

    # # if all pred_data["DPE"] are NaN, continue
    # if pred_data["DPE"].isna().all():
    #     continue

    # locator = (temp_df["SID"] == storm_id) & (temp_df["Valid Time"] == val_time)
    return temp_df["DPE"].to_numpy()


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
        pred_data["SE_wind"] = np.nan
        pred_data["SE_pressure"] = np.nan
        if pred_data.empty:
            return pred_data

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
                    val_times = pd.to_datetime(
                        pred_data_vals[pred_cols["valid_time"]]
                    ).to_numpy()
                    ref_times = ref_data[ref_cols["initial_time"]].to_numpy()
                    bool_mask = np.isin(ref_times, val_times)
                    ref_times = ref_times[bool_mask]

                    pred_data_vals.loc[:, "ref_wind"] = np.nan
                    pred_data_vals.loc[:, "ref_pres"] = np.nan

                    locator = np.isin(
                        pd.to_datetime(
                            pred_data_vals[pred_cols["valid_time"]]
                        ).to_numpy(),
                        ref_times,
                    )
                    pred_data_vals.loc[locator, "ref_wind"] = ref_data[
                        ref_cols["wind"]
                    ].to_numpy()[bool_mask]
                    pred_data_vals.loc[locator, "ref_pres"] = ref_data[
                        ref_cols["pressure"]
                    ].to_numpy()[bool_mask]
                    # if the reference or prediction values are empty strings, set them to NaN
                    pred_data_vals.loc[:, pred_cols["wind"]] = pred_data_vals[
                        pred_cols["wind"]
                    ].replace(" ", np.nan)
                    pred_data_vals.loc[:, pred_cols["pressure"]] = pred_data_vals[
                        pred_cols["pressure"]
                    ].replace(" ", np.nan)
                    pred_data_vals.loc[:, "ref_wind"] = pred_data_vals[
                        "ref_wind"
                    ].replace(" ", np.nan)
                    pred_data_vals.loc[:, "ref_pres"] = pred_data_vals[
                        "ref_pres"
                    ].replace(" ", np.nan)
                    pred_data_locator = (
                        (pred_data[pred_cols["SID"]] == storm_id)
                        & np.isin(
                            pred_data[pred_cols["valid_time"]],
                            pred_data_vals[pred_cols["valid_time"]],
                        )
                        & (pred_data[pred_cols["init_time"]] == initial_time)
                    )
                    pred_data.loc[pred_data_locator, "SE_wind"] = np.nan
                    pred_data.loc[pred_data_locator, "SE_pressure"] = np.nan
                    # skip if all wind and pressure values are NaN
                    if (
                        pred_data_vals[pred_cols["wind"]].isna().all()
                        or pred_data_vals[pred_cols["pressure"]].isna().all()
                    ):
                        continue
                    se_wind = (
                        pred_data_vals[pred_cols["wind"]].to_numpy().astype(float)
                        - pred_data_vals["ref_wind"].to_numpy().astype(float)
                    ) ** 2
                    se_pressure = (
                        pred_data_vals[pred_cols["pressure"]].to_numpy().astype(float)
                        - pred_data_vals["ref_pres"].to_numpy().astype(float)
                    ) ** 2
                    pred_data.loc[pred_data_locator, "SE_wind"] = se_wind
                    pred_data.loc[pred_data_locator, "SE_pressure"] = se_pressure
                #     pred_data_vals.loc[:,"ref_lat"] = np.nan
                #     pred_data_vals.loc[:,"ref_lon"] = np.nan

                #     locator = np.isin(pd.to_datetime(pred_data_vals[pred_cols["valid_time"]]).to_numpy(), ref_times)

                #     pred_data_vals.loc[locator, "ref_lat"] = ref_data[ref_cols["lat"]].to_numpy()[bool_mask]
                #     pred_data_vals.loc[locator, "ref_lon"] = ref_data[ref_cols["lon"]].to_numpy()[bool_mask]
                #     dist = toolbox.haversine(
                #         pred_data_vals["ref_lat"].to_numpy(),
                #         pred_data_vals["ref_lon"].to_numpy(),
                #         pred_data_vals[pred_cols["lat"]].to_numpy(),
                #         pred_data_vals[pred_cols["lon"]].to_numpy(),
                #     )
                #     pred_data_locator = (pred_data[pred_cols["SID"]] == storm_id) & np.isin(pred_data[pred_cols["valid_time"]], pred_data_vals[pred_cols["valid_time"]]) & (pred_data[pred_cols["init_time"]] == initial_time)
                #     pred_data.loc[pred_data_locator, "DPE"] = dist
                except Exception as e:
                    print(
                        f"Error processing storm {storm_id} at initial time {initial_time}: {e}"
                    )
                    continue
            else:
                for member in pred_data_vals[ensemble_dim].unique():
                    pred_data_vals_ens = pred_data_vals[
                        pred_data_vals[ensemble_dim] == member
                    ].copy()
                    val_times = pd.to_datetime(
                        pred_data_vals_ens[pred_cols["valid_time"]]
                    ).to_numpy()
                    ref_times = ref_data[ref_cols["initial_time"]].to_numpy()
                    bool_mask = np.isin(ref_times, val_times)
                    ref_times = ref_times[bool_mask]

                    pred_data_vals_ens.loc[:, "ref_wind"] = np.nan
                    pred_data_vals_ens.loc[:, "ref_pres"] = np.nan

                    locator = np.isin(
                        pd.to_datetime(
                            pred_data_vals_ens[pred_cols["valid_time"]]
                        ).to_numpy(),
                        ref_times,
                    )

                    pred_data_vals_ens.loc[locator, "ref_wind"] = ref_data[
                        ref_cols["wind"]
                    ].to_numpy()[bool_mask]
                    pred_data_vals_ens.loc[locator, "ref_pres"] = ref_data[
                        ref_cols["pressure"]
                    ].to_numpy()[bool_mask]

                    # if the reference or prediction values are empty strings, set them to NaN
                    pred_data_vals_ens.loc[:, pred_cols["wind"]] = pred_data_vals_ens[
                        pred_cols["wind"]
                    ].replace(" ", np.nan)
                    pred_data_vals_ens.loc[:, pred_cols["pressure"]] = (
                        pred_data_vals_ens[pred_cols["pressure"]].replace(" ", np.nan)
                    )
                    pred_data_vals_ens.loc[:, "ref_wind"] = pred_data_vals_ens[
                        "ref_wind"
                    ].replace(" ", np.nan)
                    pred_data_vals_ens.loc[:, "ref_pres"] = pred_data_vals_ens[
                        "ref_pres"
                    ].replace(" ", np.nan)

                    pred_data_locator = (
                        (pred_data[pred_cols["SID"]] == storm_id)
                        & np.isin(
                            pred_data[pred_cols["valid_time"]],
                            pred_data_vals_ens[pred_cols["valid_time"]],
                        )
                        & (pred_data[ensemble_dim] == member)
                        & (pred_data[pred_cols["init_time"]] == initial_time)
                    )
                    pred_data.loc[pred_data_locator, "SE_wind"] = np.nan
                    pred_data.loc[pred_data_locator, "SE_pressure"] = np.nan

                    # skip if all wind and pressure values are NaN
                    if (
                        pred_data_vals_ens[pred_cols["wind"]].isna().all()
                        or pred_data_vals_ens[pred_cols["pressure"]].isna().all()
                    ):
                        continue

                    se_wind = (
                        pred_data_vals_ens[pred_cols["wind"]].to_numpy().astype(float)
                        - pred_data_vals_ens["ref_wind"].to_numpy().astype(float)
                    ) ** 2
                    se_pressure = (
                        pred_data_vals_ens[pred_cols["pressure"]]
                        .to_numpy()
                        .astype(float)
                        - pred_data_vals_ens["ref_pres"].to_numpy().astype(float)
                    ) ** 2

                    pred_data.loc[pred_data_locator, "SE_wind"] = se_wind
                    pred_data.loc[pred_data_locator, "SE_pressure"] = se_pressure

        return pred_data

    # process the storms with joblib parallel
    results = jl.Parallel(n_jobs=6)(
        jl.delayed(storm_processor)(reference, predictions, storm_id)
        for storm_id in reference.SID.unique()
    )

    # Combine the results into a single DataFrame
    temp_df = pd.concat(results, ignore_index=True)

    return temp_df[["SE_wind", "SE_pressure"]].to_numpy()

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
        pred_data["AE_wind"] = np.nan
        pred_data["AE_pressure"] = np.nan

        if pred_data.empty:
            return pred_data

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
                    val_times = pd.to_datetime(
                        pred_data_vals[pred_cols["valid_time"]]
                    ).to_numpy()
                    ref_times = ref_data[ref_cols["initial_time"]].to_numpy()
                    bool_mask = np.isin(ref_times, val_times)
                    ref_times = ref_times[bool_mask]

                    pred_data_vals.loc[:, "ref_wind"] = np.nan
                    pred_data_vals.loc[:, "ref_pres"] = np.nan

                    locator = np.isin(
                        pd.to_datetime(
                            pred_data_vals[pred_cols["valid_time"]]
                        ).to_numpy(),
                        ref_times,
                    )
                    pred_data_vals.loc[locator, "ref_wind"] = ref_data[
                        ref_cols["wind"]
                    ].to_numpy()[bool_mask]
                    pred_data_vals.loc[locator, "ref_pres"] = ref_data[
                        ref_cols["pressure"]
                    ].to_numpy()[bool_mask]
                    # if the reference or prediction values are empty strings, set them to NaN
                    pred_data_vals.loc[:, pred_cols["wind"]] = pred_data_vals[
                        pred_cols["wind"]
                    ].replace(" ", np.nan)
                    pred_data_vals.loc[:, pred_cols["pressure"]] = pred_data_vals[
                        pred_cols["pressure"]
                    ].replace(" ", np.nan)
                    pred_data_vals.loc[:, "ref_wind"] = pred_data_vals[
                        "ref_wind"
                    ].replace(" ", np.nan)
                    pred_data_vals.loc[:, "ref_pres"] = pred_data_vals[
                        "ref_pres"
                    ].replace(" ", np.nan)
                    pred_data_locator = (
                        (pred_data[pred_cols["SID"]] == storm_id)
                        & np.isin(
                            pred_data[pred_cols["valid_time"]],
                            pred_data_vals[pred_cols["valid_time"]],
                        )
                        & (pred_data[pred_cols["init_time"]] == initial_time)
                    )
                    pred_data.loc[pred_data_locator, "AE_wind"] = np.nan
                    pred_data.loc[pred_data_locator, "AE_pressure"] = np.nan
                    # skip if all wind and pressure values are NaN
                    if (
                        pred_data_vals[pred_cols["wind"]].isna().all()
                        or pred_data_vals[pred_cols["pressure"]].isna().all()
                    ):
                        continue
                    ae_wind = np.abs(
                        pred_data_vals[pred_cols["wind"]].to_numpy().astype(float)
                        - pred_data_vals["ref_wind"].to_numpy().astype(float)
                    )
                    ae_pressure = np.abs(
                        pred_data_vals[pred_cols["pressure"]].to_numpy().astype(float)
                        - pred_data_vals["ref_pres"].to_numpy().astype(float)
                    )
                    pred_data.loc[pred_data_locator, "AE_wind"] = ae_wind
                    pred_data.loc[pred_data_locator, "AE_pressure"] = ae_pressure
                #     pred_data_vals.loc[:,"ref_lat"] = np.nan
                #     pred_data_vals.loc[:,"ref_lon"] = np.nan

                #     locator = np.isin(pd.to_datetime(pred_data_vals[pred_cols["valid_time"]]).to_numpy(), ref_times)

                #     pred_data_vals.loc[locator, "ref_lat"] = ref_data[ref_cols["lat"]].to_numpy()[bool_mask]
                #     pred_data_vals.loc[locator, "ref_lon"] = ref_data[ref_cols["lon"]].to_numpy()[bool_mask]
                #     dist = toolbox.haversine(
                #         pred_data_vals["ref_lat"].to_numpy(),
                #         pred_data_vals["ref_lon"].to_numpy(),
                #         pred_data_vals[pred_cols["lat"]].to_numpy(),
                #         pred_data_vals[pred_cols["lon"]].to_numpy(),
                #     )
                #     pred_data_locator = (pred_data[pred_cols["SID"]] == storm_id) & np.isin(pred_data[pred_cols["valid_time"]], pred_data_vals[pred_cols["valid_time"]]) & (pred_data[pred_cols["init_time"]] == initial_time)
                #     pred_data.loc[pred_data_locator, "DPE"] = dist
                except Exception as e:
                    print(
                        f"Error processing storm {storm_id} at initial time {initial_time}: {e}"
                    )
                    continue
            else:
                for member in pred_data_vals[ensemble_dim].unique():
                    pred_data_vals_ens = pred_data_vals[
                        pred_data_vals[ensemble_dim] == member
                    ].copy()
                    val_times = pd.to_datetime(
                        pred_data_vals_ens[pred_cols["valid_time"]]
                    ).to_numpy()
                    ref_times = ref_data[ref_cols["initial_time"]].to_numpy()
                    bool_mask = np.isin(ref_times, val_times)
                    ref_times = ref_times[bool_mask]

                    pred_data_vals_ens.loc[:, "ref_wind"] = np.nan
                    pred_data_vals_ens.loc[:, "ref_pres"] = np.nan

                    locator = np.isin(
                        pd.to_datetime(
                            pred_data_vals_ens[pred_cols["valid_time"]]
                        ).to_numpy(),
                        ref_times,
                    )

                    pred_data_vals_ens.loc[locator, "ref_wind"] = ref_data[
                        ref_cols["wind"]
                    ].to_numpy()[bool_mask]
                    pred_data_vals_ens.loc[locator, "ref_pres"] = ref_data[
                        ref_cols["pressure"]
                    ].to_numpy()[bool_mask]

                    # if the reference or prediction values are empty strings, set them to NaN
                    pred_data_vals_ens.loc[:, pred_cols["wind"]] = pred_data_vals_ens[
                        pred_cols["wind"]
                    ].replace(" ", np.nan)
                    pred_data_vals_ens.loc[:, pred_cols["pressure"]] = (
                        pred_data_vals_ens[pred_cols["pressure"]].replace(" ", np.nan)
                    )
                    pred_data_vals_ens.loc[:, "ref_wind"] = pred_data_vals_ens[
                        "ref_wind"
                    ].replace(" ", np.nan)
                    pred_data_vals_ens.loc[:, "ref_pres"] = pred_data_vals_ens[
                        "ref_pres"
                    ].replace(" ", np.nan)

                    pred_data_locator = (
                        (pred_data[pred_cols["SID"]] == storm_id)
                        & np.isin(
                            pred_data[pred_cols["valid_time"]],
                            pred_data_vals_ens[pred_cols["valid_time"]],
                        )
                        & (pred_data[ensemble_dim] == member)
                        & (pred_data[pred_cols["init_time"]] == initial_time)
                    )
                    pred_data.loc[pred_data_locator, "AE_wind"] = np.nan
                    pred_data.loc[pred_data_locator, "AE_pressure"] = np.nan

                    # skip if all wind and pressure values are NaN
                    if (
                        pred_data_vals_ens[pred_cols["wind"]].isna().all()
                        or pred_data_vals_ens[pred_cols["pressure"]].isna().all()
                    ):
                        continue

                    ae_wind = np.abs(
                        pred_data_vals_ens[pred_cols["wind"]].to_numpy().astype(float)
                        - pred_data_vals_ens["ref_wind"].to_numpy().astype(float)
                    )
                    ae_pressure = np.abs(
                        pred_data_vals_ens[pred_cols["pressure"]]
                        .to_numpy()
                        .astype(float)
                        - pred_data_vals_ens["ref_pres"].to_numpy().astype(float)
                    )

                    pred_data.loc[pred_data_locator, "AE_wind"] = ae_wind
                    pred_data.loc[pred_data_locator, "AE_pressure"] = ae_pressure

        return pred_data

    # process the storms with joblib parallel
    results = jl.Parallel(n_jobs=6)(
        jl.delayed(storm_processor)(reference, predictions, storm_id)
        for storm_id in reference.SID.unique()
    )

    # Combine the results into a single DataFrame
    temp_df = pd.concat(results, ignore_index=True)

    return temp_df[["AE_wind", "AE_pressure"]].to_numpy()

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
    """Compute the Cartesian Cross-Track / Along-Track / Direct Positional error
    between the reference and predictions projected onto a cartesian plane.

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

    # convert lat and lon to cartesian coordinates from scratch
    ref_x, ref_y = toolbox.latlon_2_cartesian(
        reference[ref_cols["lat"]].to_numpy(),
        reference[ref_cols["lon"]].to_numpy(),
    )
    pred_x, pred_y = toolbox.latlon_2_cartesian(
        predictions[pred_cols["lat"]].to_numpy(),
        predictions[pred_cols["lon"]].to_numpy(),
    )

    # add to respective dataframes
    reference["x"] = ref_x
    reference["y"] = ref_y
    predictions["x"] = pred_x
    predictions["y"] = pred_y

    temp_df = predictions.copy()
    temp_df["ATE"] = np.nan
    temp_df["CTE"] = np.nan
    for storm_id in reference.SID.unique():
        # Get the reference and prediction data for the current storm ID
        ref_data = reference[reference[ref_cols["SID"]] == storm_id]
        pred_data = predictions[predictions[pred_cols["SID"]] == storm_id]

        for val_time in pred_data[pred_cols["valid_time"]].unique():
            # Get the reference data for the valid time and one time step before
            ref_start = ref_data[
                ref_data[ref_cols["initial_time"]]
                == pd.to_datetime(val_time) - pd.Timedelta(hours=6)
            ]
            ref_end = ref_data[
                ref_data[ref_cols["initial_time"]] == pd.to_datetime(val_time)
            ]
            pred_data_val = pred_data[pred_data[pred_cols["valid_time"]] == val_time]
            if len(ref_start) == 0 or len(ref_end) == 0 or len(pred_data_val) == 0:
                continue

            dx_ = ref_end["x"].to_numpy() - ref_start["x"].to_numpy()
            dy_ = ref_end["y"].to_numpy() - ref_start["y"].to_numpy()
            norm = dx_**2 + dy_**2

            if norm == 0:
                projx = ref_start["x"].to_numpy()
                projy = ref_start["y"].to_numpy()
            else:
                t = (
                    (pred_data_val["x"].to_numpy() - ref_start["x"].to_numpy()) * dx_
                    + (pred_data_val["y"].to_numpy() - ref_start["y"].to_numpy()) * dy_
                ) / norm
                projx = ref_start["x"].to_numpy() + t * dx_
                projy = ref_start["y"].to_numpy() + t * dy_

            # calculate the cross track error
            dx = np.sqrt(
                (pred_data_val["x"].to_numpy() - projx) ** 2
                + (pred_data_val["y"].to_numpy() - projy) ** 2
            )
            dt = np.sqrt(
                (ref_end["x"].to_numpy() - projx) ** 2
                + (ref_end["y"].to_numpy() - projy) ** 2
            )

            # if the shape of dt and dx are not the same, then we need to
            # tile dt
            if dt.shape != dx.shape:
                dt = np.tile(dt, (dx.shape[0],))

            CDPE = np.sqrt(dx**2 + dt**2)

            pred_data.loc[pred_data_val.index, "ATE"] = dt
            pred_data.loc[pred_data_val.index, "CTE"] = dx
            pred_data.loc[pred_data_val.index, "CDPE"] = CDPE

        temp_df.loc[pred_data.index, "ATE"] = pred_data["ATE"]
        temp_df.loc[pred_data.index, "CTE"] = pred_data["CTE"]
        temp_df.loc[pred_data.index, "CDPE"] = pred_data["CDPE"]

    return temp_df["ATE", "CTE", "CDPE"].to_numpy()


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
            "ensemble_idx": "ensemble_idx",
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

    # convert the wind and pressure columns to numeric
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

    sample_idx = predictions[pred_cols["ensemble_idx"]].unique()[0]
    CRPS = predictions[predictions[pred_cols["ensemble_idx"]] == sample_idx].copy()
    # keep only the SID, valid time, and init time columns
    CRPS = CRPS[[pred_cols["SID"], pred_cols["valid_time"], pred_cols["init_time"]]]
    # add the target columns
    CRPS["CRPS_vmax"] = np.nan
    CRPS["CRPS_pmin"] = np.nan

    for storm_id in reference.SID.unique():
        # Get the reference and prediction data for the current storm ID
        ref_data = reference[reference[ref_cols["SID"]] == storm_id]
        pred_data = predictions[predictions[pred_cols["SID"]] == storm_id]

        for val_time in pred_data[pred_cols["valid_time"]].unique():
            # Get the reference and prediction data for the current valid time
            ref_data_val = ref_data[ref_data[ref_cols["initial_time"]] == val_time]
            pred_data_val = pred_data[pred_data[pred_cols["valid_time"]] == val_time]

            # if the reference or prediction values are empty strings, set them to NaN
            ref_data_val.loc[:, ref_cols["wind"]] = ref_data_val[
                ref_cols["wind"]
            ].replace(" ", np.nan)
            ref_data_val.loc[:, ref_cols["pressure"]] = ref_data_val[
                ref_cols["pressure"]
            ].replace(" ", np.nan)
            pred_data_val.loc[:, pred_cols["wind"]] = pred_data_val[
                pred_cols["wind"]
            ].replace(" ", np.nan)
            pred_data_val.loc[:, pred_cols["pressure"]] = pred_data_val[
                pred_cols["pressure"]
            ].replace(" ", np.nan)

            if len(ref_data_val) == 0 or len(pred_data_val) == 0:
                continue
            # skip if all values are NaN
            if (
                ref_data_val[ref_cols["wind"]].isna().all()
                or pred_data_val[pred_cols["wind"]].isna().all()
            ):
                continue

            for init_time in pred_data_val[pred_cols["init_time"]].unique():
                ensemble_vals = pred_data_val[
                    pred_data_val[pred_cols["init_time"]] == init_time
                ]
                num_members = len(ensemble_vals)
                if len(ensemble_vals) == 0:
                    continue

                ref_wind = ref_data_val[ref_cols["wind"]].to_numpy().astype(np.float32)
                ref_pressure = (
                    ref_data_val[ref_cols["pressure"]].to_numpy().astype(np.float32)
                )

                diff_wind = np.abs(
                    ref_wind - ensemble_vals[pred_cols["wind"]].to_numpy()
                )
                diff_matrix_wind = np.abs(
                    ensemble_vals[pred_cols["wind"]].to_numpy()[:, None]
                    - ensemble_vals[pred_cols["wind"]].to_numpy()[None, :]
                )

                diff_pressure = np.abs(
                    ref_pressure - ensemble_vals[pred_cols["pressure"]].to_numpy()
                )
                diff_matrix_pressure = np.abs(
                    ensemble_vals[pred_cols["pressure"]].to_numpy()[:, None]
                    - ensemble_vals[pred_cols["pressure"]].to_numpy()[None, :]
                )

                # CRPS for wind
                crps_wind = (
                    diff_wind.mean()
                    - 1 / (2 * num_members * (num_members - 1)) * diff_matrix_wind.sum()
                )
                # CRPS for pressure
                crps_pressure = (
                    diff_pressure.mean()
                    - 1
                    / (2 * num_members * (num_members - 1))
                    * diff_matrix_pressure.sum()
                )

                locator = (
                    (CRPS[pred_cols["SID"]] == storm_id)
                    & (CRPS[pred_cols["valid_time"]] == val_time)
                    & (CRPS[pred_cols["init_time"]] == init_time)
                )

                CRPS.loc[
                    locator,
                    "CRPS_vmax",
                ] = crps_wind
                CRPS.loc[
                    locator,
                    "CRPS_pmin",
                ] = crps_pressure

    return CRPS[["CRPS_vmax", "CRPS_pmin"]].to_numpy()


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
            "ensemble_idx": "ensemble_idx",
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

    sample_idx = predictions[pred_cols["ensemble_idx"]].unique()[0]
    CRPS = predictions[predictions[pred_cols["ensemble_idx"]] == sample_idx].copy()
    # keep only the SID, valid time, and init time columns
    CRPS = CRPS[[pred_cols["SID"], pred_cols["valid_time"], pred_cols["init_time"]]]
    # add the target column
    CRPS["CRPS_haversine"] = np.nan

    for storm_id in reference.SID.unique():
        # Get the reference and prediction data for the current storm ID
        ref_data = reference[reference[ref_cols["SID"]] == storm_id]
        pred_data = predictions[predictions[pred_cols["SID"]] == storm_id]

        for val_time in pred_data[pred_cols["valid_time"]].unique():
            # Get the reference and prediction data for the current valid time
            ref_data_val = ref_data[ref_data[ref_cols["initial_time"]] == val_time]
            pred_data_val = pred_data[pred_data[pred_cols["valid_time"]] == val_time]

            # if the reference or prediction values are empty strings, set them to NaN
            ref_data_val.loc[:, ref_cols["lat"]] = ref_data_val[
                ref_cols["lat"]
            ].replace(" ", np.nan)
            ref_data_val.loc[:, ref_cols["lon"]] = ref_data_val[
                ref_cols["lon"]
            ].replace(" ", np.nan)
            pred_data_val.loc[:, pred_cols["lat"]] = pred_data_val[
                pred_cols["lat"]
            ].replace(" ", np.nan)
            pred_data_val.loc[:, pred_cols["lon"]] = pred_data_val[
                pred_cols["lon"]
            ].replace(" ", np.nan)

            if len(ref_data_val) == 0 or len(pred_data_val) == 0:
                continue

            # skip if all values are NaN
            if (
                ref_data_val[ref_cols["lat"]].isna().all()
                or pred_data_val[pred_cols["lat"]].isna().all()
            ):
                continue

            for init_time in pred_data_val[pred_cols["init_time"]].unique():
                ensemble_vals = pred_data_val[
                    pred_data_val[pred_cols["init_time"]] == init_time
                ]
                num_members = len(ensemble_vals)
                if len(ensemble_vals) == 0:
                    continue

                ref_lat = ref_data_val[ref_cols["lat"]].to_numpy().astype(np.float32)
                ref_lon = ref_data_val[ref_cols["lon"]].to_numpy().astype(np.float32)

                diff_track = toolbox.haversine(
                    ref_lat,
                    ref_lon,
                    ensemble_vals[pred_cols["lat"]].to_numpy(),
                    ensemble_vals[pred_cols["lon"]].to_numpy(),
                )
                diff_matrix = toolbox.haversine(
                    ensemble_vals[pred_cols["lat"]].to_numpy()[:, None],
                    ensemble_vals[pred_cols["lon"]].to_numpy()[:, None],
                    ensemble_vals[pred_cols["lat"]].to_numpy()[None, :],
                    ensemble_vals[pred_cols["lon"]].to_numpy()[None, :],
                )

                denom = 2 * num_members * (num_members - 1)
                if denom == 0:
                    continue

                haver_crps = (
                    diff_track.mean()
                    - 1 / (2 * num_members * (num_members - 1)) * diff_matrix.sum()
                )

                locator = (
                    (CRPS[pred_cols["SID"]] == storm_id)
                    & (CRPS[pred_cols["valid_time"]] == val_time)
                    & (CRPS[pred_cols["init_time"]] == init_time)
                )

                CRPS.loc[
                    locator,
                    "CRPS_haversine",
                ] = haver_crps

    return CRPS[["CRPS_haversine"]].to_numpy()


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

# %%
