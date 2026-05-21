# %%
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import huracanpy

import argparse
import sys

# %%
parser = argparse.ArgumentParser(description="Match detected tracks with ibtracs data")
parser.add_argument(
    "--model", type=str, default="2023_aifs", help="Model name (e.g. 2023_aifs)"
)
parser.add_argument(
    "--ibtracs_path",
    type=str,
    default=None,
    help="Path to ibtracs csv file (e.g. data/ibtracs/ibtracs.ALL.list.v04r01.csv)",
)
parser.add_argument(
    "--output_path",
    type=str,
    default=None,
    help="Path to save the output csv file (e.g. outputs/2023_aifs.csv)",
)
args = parser.parse_args()

# %%
model = args.model

file_dir = str(Path.cwd() / "data" / model)
# %%
# Get the list of files in the directory
days = Path.iterdir(Path(file_dir))
# %%
# Load ibtracs
if args.ibtracs_path is not None:
    ibtracs_path = args.ibtracs_path
else:
    ibtracs_path = str(Path.cwd() / "data" / "ibtracs" / "ibtracs.ALL.list.v04r01.csv")

ibtracs = pd.read_csv(ibtracs_path, keep_default_na=False)

# remove the first row
ibtracs = ibtracs.iloc[1:]

# Keep only the SID, ATCF_ID, lat, lon, ISO_TIME, USA_WIND and USA_PRESSURE columns
ibtracs = ibtracs[
    ["SID", "USA_ATCF_ID", "LAT", "LON", "ISO_TIME", "SEASON", "USA_WIND", "USA_PRES"]
]

# filter out rows with non-numeric values in the SEASON column
# conver lat, lon to float
ibtracs["LAT"] = ibtracs["LAT"].astype(float)
ibtracs["LON"] = ibtracs["LON"].astype(float)

# if the longtitude is less than 0, add 360
ibtracs.loc[ibtracs["LON"] < 0, "LON"] += 360

ibtracs = ibtracs[pd.to_numeric(ibtracs["SEASON"], errors="coerce").notnull()]
# turn season to int
ibtracs["SEASON"] = ibtracs["SEASON"].astype(int)
ibtracs = ibtracs[ibtracs["SEASON"] > 2020]
# turn ISO_TIME to datetime
ibtracs["ISO_TIME"] = pd.to_datetime(ibtracs["ISO_TIME"])
# select year 2023
ibtracs = ibtracs[ibtracs["ISO_TIME"].dt.year == 2023]


# %%
# Create an empty DataFrame with the desired columns
columns = [
    "SID",
    "Initial Time",
    "Valid Time",
    "ensemble_idx",
    "wind max",
    "pressure min",
    "lat",
    "lon",
]
df = pd.DataFrame(columns=columns)
ib = None
# %%
for file in days:
    if "aifs" in model:
        init_date = file.name.split("_")[1][5:]
    else:
        init_date = file.name.split("_")[1]
    init_datetime = pd.to_datetime(init_date)

    try:
        detected = huracanpy.load(str(file))
    except Exception as e:
        print(f"Error loading detected data: {e}", f"Skipping file: {file}", sep="\n")
        continue

    # detected is a netcdf file; flip the lat values if model is "aifs"
    if "aifs" in model:
        # multiply lat values by -1
        detected["lat"] = detected["lat"] * -1

    if ib is None:
        try:
            ib = huracanpy._data.load(
                filename=ibtracs, source="csv", load_function=lambda x: x
            )
        except Exception as e:
            print(f"Error loading ibtracs data: {e}")
            raise e
        ib = ib.rename({"sid": "track_id"})

    match = huracanpy.assess.match([ib, detected], ["ibtracs", "detected"])

    # iterate through the match df rows
    for index, row in match.iterrows():
        # Get the values for the current row
        sid = row["id_ibtracs"]
        initial_time = init_datetime
        temp_track = detected.where(detected.track_id == row["id_detected"], drop=True)
        detected_wind = temp_track.wind10.values
        detected_pressure = temp_track.slp.values
        detected_lat = temp_track.lat.values
        # switch lat sign for all values
        # detected_lat = np.where(detected_lat > 0, detected_lat, -detected_lat)
        detected_lon = temp_track.lon.values
        detected_time = temp_track.time.values

        # Create a new DataFrame with the current values
        temp_df = pd.DataFrame(
            {
                "SID": sid,
                "Initial Time": initial_time,
                "Valid Time": detected_time,
                # "ensemble_idx": None because deterministic
                "wind max": detected_wind,
                "pressure min": detected_pressure,
                "lat": detected_lat,
                "lon": detected_lon,
            }
        )

        if df.empty:
            df = temp_df
        else:
            # Concatenate the new DataFrame with the existing one
            df = pd.concat([df, temp_df], ignore_index=True)

# %%
# convert wind to knots
df["wind max"] = df["wind max"] * 1.94384
# convert pressure to hPa
df["pressure min"] = df["pressure min"] / 100

# %%

if args.output_path is not None:
    output_path = args.output_path
else:
    output_dir = str(Path.cwd() / "outputs")
    Path.mkdir(Path(output_dir), exist_ok=True, parents=True)
    output_path = str(Path(output_dir) / f"{model}.csv")

# Save the DataFrame to a CSV file
df.to_csv(output_path, index=False)


# %%
