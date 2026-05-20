# %% Imports
# OS and IO
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

import argparse

from utils import toolbox, constants
from utils.toolbox import *
from utils import data_lib as dlib
import metrics

# %% Argument parser
parser = argparse.ArgumentParser(description="Evaluate tracks including RI metrics")
parser.add_argument(
    "--ibtracs_folder",
    type=str,
    default=os.path.join(os.curdir, "data", "ibtracs"),
    help="Folder containing IBTrACS CSV track file",
)
parser.add_argument(
    "--results_folder",
    type=str,
    default=os.path.join(os.curdir, "outputs"),
    help="Folder containing one or more track CSV files to evaluate",
)
parser.add_argument(
    "--RI_thresh",
    type=float,
    default=30,
    help="Threshold for rapid intensification (RI) in knots per window (default: 34 knots/24h)",
)
parser.add_argument(
    "--RI_window",
    type=int,
    default=24,
    help="Time window for rapid intensification (RI) in hours",
)
parser.add_argument(
    "--keep_intensification",
    action="store_true",
    help="Keep track of intensification float value",
)

# %% Read IBTrACS data
ibtracs = toolbox.read_hist_track_file(tracks_path=args.ibtracs_folder)
ibtracs = ibtracs[
    (ibtracs["ISO_TIME"].dt.year >= 1980) & (ibtracs["ISO_TIME"].dt.year <= 2023)
]
ibtracs.loc[:, "USA_WIND"] = (
    ibtracs["USA_WIND"].str.strip().replace("", np.nan).astype(float)
)
ibtracs.loc[:, "USA_PRES"] = (
    ibtracs["USA_PRES"].str.strip().replace("", np.nan).astype(float)
)

# %%
# Create a reference dataframe for the initial times and SIDs
ibtracs_time_ref = ibtracs[["ISO_TIME", "SID", "BASIN", "LAT", "LON"]].copy()

# get the SID values
SIDs = ibtracs_time_ref[np.isin(ibtracs_time_ref["ISO_TIME"].dt.hour, [0, 6, 12, 18])][
    "SID"
].values[:, None]
basins = ibtracs_time_ref[
    np.isin(ibtracs_time_ref["ISO_TIME"].dt.hour, [0, 6, 12, 18])
]["BASIN"].values[:, None]

# get the initial time values
start_times = ibtracs_time_ref[
    np.isin(ibtracs_time_ref["ISO_TIME"].dt.hour, [0, 6, 12, 18])
]["ISO_TIME"].values[:, None]

# calculate the desired forecast times
deltas = np.arange(6, 120 + 6, 6).astype("timedelta64[h]")

# tile the SIDs and start times to get the forecast shape (n_samples, n_forecasts)
SIDs = np.tile(SIDs, len(deltas))
start_times = np.tile(start_times, len(deltas))
basins = np.tile(basins, len(deltas))

# Calculate the target times
target_times = start_times + deltas

ibtracs_clim = pd.DataFrame(
    {
        "SID": SIDs.flatten(),
        "Initial Time": start_times.flatten(),
        "Valid Time": target_times.flatten(),
        "Basin": basins.flatten(),
    }
)
# %%
# Make a dataframe with the start time for each storm
start_df = ibtracs.groupby("SID")["ISO_TIME"].min().reset_index()
ibtracs_clim = ibtracs_clim.merge(start_df, on="SID", how="left").rename(
    columns={"ISO_TIME": "Storm Start"}
)

# Fetch max wind and location at initial time
ibtracs_clim = ibtracs_clim.merge(
    ibtracs[["SID", "ISO_TIME", "USA_WIND", "LAT", "LON", "USA_PRES"]].rename(
        columns={
            "USA_WIND": "Init Vmax",
            "USA_PRES": "Init Pmin",
            "ISO_TIME": "Initial Time",
            "LAT": "Init Lat",
            "LON": "Init Lon",
        }
    ),
    on=["SID", "Initial Time"],
    how="left",
)

# fetch max wind and location and valid time
ibtracs_clim = ibtracs_clim.merge(
    ibtracs[["SID", "ISO_TIME", "USA_WIND", "LAT", "LON", "USA_PRES"]].rename(
        columns={
            "USA_WIND": "Valid Vmax",
            "USA_PRES": "Valid Pmin",
            "ISO_TIME": "Valid Time",
            "LAT": "Valid Lat",
            "LON": "Valid Lon",
        }
    ),
    on=["SID", "Valid Time"],
    how="left",
)

# Calculate storm lifetime, in hours
ibtracs_clim["lifetime"] = (
    (ibtracs_clim["Valid Time"] - ibtracs_clim["Storm Start"]).dt.total_seconds() / 3600
).astype(int)
ibtracs_clim["lead_time"] = (
    (ibtracs_clim["Valid Time"] - ibtracs_clim["Initial Time"]).dt.total_seconds()
    / 3600
).astype(int)
# Calculate Intensification
ibtracs_clim["intensification_kt"] = ibtracs_clim["Valid Vmax"].astype(
    float
) - ibtracs_clim["Init Vmax"].astype(float)
ibtracs_clim["intensification_hpa"] = ibtracs_clim["Init Pmin"].astype(
    float
) - ibtracs_clim["Valid Pmin"].astype(float)
ibtracs_clim["lat_change"] = ibtracs_clim["Valid Lat"] - ibtracs_clim["Init Lat"]
ibtracs_clim["lon_change"] = ibtracs_clim["Valid Lon"] - ibtracs_clim["Init Lon"]

# remove rows where intensification, lat_change, and lon_change are all missing
ibtracs_clim = ibtracs_clim.dropna(
    subset=["intensification_kt", "lat_change", "lon_change", "intensification_hpa"],
    how="all",
)

# drop unecessary columns
ibtracs_clim = ibtracs_clim.drop(
    columns=[
        "Init Vmax",
        "Init Pmin",
        "Init Lat",
        "Init Lon",
        "Valid Vmax",
        "Valid Pmin",
        "Valid Lat",
        "Valid Lon",
    ]
)
# %%
eval_folder = os.path.join(os.curdir, "outputs")
ibtracs_clim.to_csv(f"{eval_folder}/ibtracs_clim.csv", index=False)

# %%
