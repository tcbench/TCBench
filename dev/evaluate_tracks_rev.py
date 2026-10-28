# %% Imports
# OS and IO
import os
import sys
from typing import List, Optional, Dict

_DEV_FOLDER = "/home/mgomezd1/repos/iop_tcbench/dev"
if _DEV_FOLDER not in sys.path:
    sys.path.insert(0, _DEV_FOLDER)
os.chdir(_DEV_FOLDER)

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import yaml


from utils import toolbox
from utils.toolbox import *
from utils import data_lib as dlib
import metrics_rev as metrics
import re

# %% Constants
# Evaluation configuration (paths, year, basin, lead times, thresholds),
# editable in a YAML file
CONFIG_PATH = os.path.join(_DEV_FOLDER, "evaluate_tracks_config.yaml")

# Columns to keep for the baseline forecasts
_BASELINE_COLS = [
    "sid",
    "initial_time",
    "valid_time",
    "wind_max",
    "pressure_min",
    "lat",
    "lon",
]


# %% Helper Functions
def _to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _to_numeric(s: pd.Series) -> pd.Series:
    # Robust numeric coercion (handles strings with blanks)
    if s.dtype == object:
        s = s.astype(str).str.strip().replace({"": np.nan})
    return pd.to_numeric(s, errors="coerce")


def _deduplicate_keys(df: pd.DataFrame, key_cols: list[str], **kwargs) -> pd.DataFrame:
    """
    Deduplicates a DataFrame based on specified key columns.
    Keeps the first occurrence of each unique key combination.
    """
    before_cleaning = len(df)
    out_df = (
        df.copy().drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)
    )
    after_cleaning = len(out_df)
    if before_cleaning != after_cleaning:
        model = kwargs.get("model", None)
        if model is not None:
            print(
                f"[INFO] Deduplicated {before_cleaning - after_cleaning} rows for model '{model}' based on keys: {key_cols}"
            )
        else:
            print(
                f"[INFO] Deduplicated {before_cleaning - after_cleaning} rows based on keys: {key_cols}"
            )
    return out_df


def _candidate_processing(candidate_paths: list[str]) -> list[str]:
    """
    Purges candidate paths that are not valid CSVs, do not contain the required columns,
    or match the results naming pattern. Returns a list of valid candidate paths.
    """

    # Constants
    _EXCLUDE_RE = re.compile(r"_20(?:2[0-4])")
    _REQUIRED_COLS = {"sid", "initial_time", "valid_time"}

    valid_candidates = []
    for path in candidate_paths:
        if not path.endswith(".csv") or _EXCLUDE_RE.search(path):
            continue
        try:
            cols = pd.read_csv(path, nrows=0).columns  # header only
        except Exception:
            continue
        if _REQUIRED_COLS.issubset(cols):
            valid_candidates.append(path)
    return valid_candidates


def _col_typing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures consistent column types for evaluation.
    """
    df = df.copy()
    # Ensure datetime columns are in datetime format
    for col in ["initial_time", "valid_time"]:
        if col in df.columns:
            df[col] = _to_datetime(df[col])

    # Ensure numeric columns are in numeric format
    for col in ["wind_max", "pressure_min", "lat", "lon"]:
        if col in df.columns:
            df[col] = _to_numeric(df[col])

    # Ensure boolean columns are in float format
    if "RI" in df.columns:
        df["RI"] = df["RI"].astype(float)

    return df


def _baseline_prep(ibtracs: pd.DataFrame) -> pd.DataFrame:
    """Filtering to the times of interest ( 00, 06, 12, 18 UTC) and ensuring
    numeric types for the relevant columns, plus dropping duplicates and empty
    rows.
    """
    ib = ibtracs[
        ["SID", "ISO_TIME", "BASIN", "USA_WIND", "USA_PRES", "LAT", "LON"]
    ].copy()
    ib["ISO_TIME"] = _to_datetime(ib["ISO_TIME"])
    for col in ["USA_WIND", "USA_PRES", "LAT", "LON"]:
        ib[col] = _to_numeric(ib[col])
    ib = ib[
        ib["ISO_TIME"].dt.hour.isin([0, 6, 12, 18]) & (ib["ISO_TIME"].dt.minute == 0)
    ]
    ib = ib.dropna(subset=["USA_WIND", "USA_PRES", "LAT", "LON"], how="all")
    return ib.drop_duplicates(subset=["SID", "ISO_TIME"]).reset_index(drop=True)


def _expand_leads(inits: pd.DataFrame, max_lead_h: int) -> pd.DataFrame:
    """Repeat each initial-time row for leads 6..max_lead_h h, adding lead_hours
    and valid_time."""
    inits = inits.reset_index(drop=True)
    leads = np.arange(6, max_lead_h + 1, 6)
    cases = inits.loc[inits.index.repeat(len(leads))].reset_index(drop=True)
    cases["lead_hours"] = np.tile(leads, len(inits))
    cases["valid_time"] = cases["initial_time"] + pd.to_timedelta(
        cases["lead_hours"], unit="h"
    )
    return cases


def _destination(lat, lon, brg_rad, dist_km, radius=6371.0):
    """Point reached from (lat, lon) in degrees after dist_km along a great
    circle with initial bearing brg_rad (radians clockwise from north)."""
    p1, l1 = np.radians(lat), np.radians(lon)
    d = np.asarray(dist_km, dtype=float) / radius
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(brg_rad))
    l2 = l1 + np.arctan2(
        np.sin(brg_rad) * np.sin(d) * np.cos(p1), np.cos(d) - np.sin(p1) * np.sin(p2)
    )
    return np.degrees(p2), np.degrees(l2)


# %% Computing Functions
def _ib_ref_builder(
    ibtracs_tracks_path: str, year: int, basin: str | None = None
) -> pd.DataFrame:
    """
    Builds the reference IBTrACS DataFrame for RI computation.
    """
    # --- Load and filter IBTrACS
    ib_ref = toolbox.read_hist_track_file(tracks_path=ibtracs_tracks_path)
    ib_ref["ISO_TIME"] = _to_datetime(ib_ref["ISO_TIME"])
    if basin is not None:
        basin_upper = basin.upper()
        if basin_upper not in ["SI", "SP", "NA", "NI", "WP", "EP"]:
            basin = None

    if basin is not None:
        ib_ref = ib_ref[ib_ref["BASIN"] == basin].copy()

    ib_ref = ib_ref[ib_ref["ISO_TIME"].dt.year == int(year)].copy()
    ib_ref["USA_WIND"] = _to_numeric(ib_ref["USA_WIND"])
    ib_ref["USA_PRES"] = _to_numeric(ib_ref["USA_PRES"])

    return ib_ref


def _build_clim_intensity(
    ibtracs: pd.DataFrame,
    out_path: str,
    year: int,
    train_years: tuple[int, int] = (1980, 2022),
    max_lead_h: int = 120,
    init_hours: tuple[int, ...] = (0, 12),
    basin: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Climatology baseline for intensity, pressure and position.

    For each basin and lead time, computes over all 6-hourly initial times in
    train_years the mean change from the initial time to initial_time + lead of
      - USA_WIND (kt) and USA_PRES (hPa)
      - position, as eastward and northward displacement in km (local
        equirectangular at the initial latitude, longitude wrapped)
    Each case initialised in `year` is then forecast as its observed initial
    value plus that mean change; the displacement is converted back to lat/lon
    at the initial latitude. The basin is taken at the initial time. Only storms
    still present in IBTrACS at the valid time contribute to the means. Each
    variable is NaN where its initial value is missing.

    Saves the forecasts (sid, initial_time, valid_time, wind_max, pressure_min,
    lat, lon; lon in [0, 360)) to out_path and the means table (BASIN,
    lead_hours, mean_dwind, n_wind, mean_dpres, n_pres, mean_dx_km, mean_dy_km,
    n_pos) next to it with a '_means' suffix. Returns the forecasts.
    """
    radius = 6371.0  # km
    if train_years[0] <= year <= train_years[1]:
        print(
            f"[CLIM] Warning: target year {year} lies inside the training period {train_years}."
        )

    ib = _baseline_prep(ibtracs)

    # Observed values at any valid time
    obs = ib.drop(columns="BASIN").rename(
        columns={
            "ISO_TIME": "valid_time",
            "USA_WIND": "wind_v",
            "USA_PRES": "pres_v",
            "LAT": "lat_v",
            "LON": "lon_v",
        }
    )

    # (initial time x lead) cases for the training and target years
    init_year = ib["ISO_TIME"].dt.year
    inits = ib[init_year.between(*train_years) | (init_year == year)].rename(
        columns={
            "ISO_TIME": "initial_time",
            "USA_WIND": "wind_0",
            "USA_PRES": "pres_0",
            "LAT": "lat_0",
            "LON": "lon_0",
        }
    )
    cases = _expand_leads(inits, max_lead_h)

    # Mean changes by basin and lead time over the training period
    train = cases[cases["initial_time"].dt.year.between(*train_years)].merge(
        obs, on=["SID", "valid_time"], how="inner"
    )
    dlon = (train["lon_v"] - train["lon_0"] + 180) % 360 - 180
    train = train.assign(
        dwind=train["wind_v"] - train["wind_0"],
        dpres=train["pres_v"] - train["pres_0"],
        dx_km=radius * np.radians(dlon) * np.cos(np.radians(train["lat_0"])),
        dy_km=radius * np.radians(train["lat_v"] - train["lat_0"]),
    )
    means = train.groupby(["BASIN", "lead_hours"], as_index=False).agg(
        mean_dwind=("dwind", "mean"),
        n_wind=("dwind", "count"),
        mean_dpres=("dpres", "mean"),
        n_pres=("dpres", "count"),
        mean_dx_km=("dx_km", "mean"),
        mean_dy_km=("dy_km", "mean"),
        n_pos=("dy_km", "count"),
    )

    # Forecasts for the target year at the forecast initialisation hours
    fc = cases[
        (cases["initial_time"].dt.year == year)
        & cases["initial_time"].dt.hour.isin(init_hours)
    ]
    if basin is not None and basin.upper() in ["SI", "SP", "NA", "NI", "WP", "EP"]:
        fc = fc[fc["BASIN"] == basin.upper()]
    fc = fc.merge(
        means[
            [
                "BASIN",
                "lead_hours",
                "mean_dwind",
                "mean_dpres",
                "mean_dx_km",
                "mean_dy_km",
            ]
        ],
        on=["BASIN", "lead_hours"],
        how="left",
    )
    fc["wind_max"] = (fc["wind_0"] + fc["mean_dwind"]).clip(lower=0).round(1)
    fc["pressure_min"] = (fc["pres_0"] + fc["mean_dpres"]).round(1)
    fc["lat"] = (
        (fc["lat_0"] + np.degrees(fc["mean_dy_km"] / radius)).clip(-90, 90).round(2)
    )
    fc["lon"] = (
        (
            fc["lon_0"]
            + np.degrees(fc["mean_dx_km"] / (radius * np.cos(np.radians(fc["lat_0"]))))
        )
        % 360
    ).round(2)
    fc = fc.rename(columns={"SID": "sid"})[_BASELINE_COLS]

    fc.to_csv(out_path, index=False)
    means_path = f"{os.path.splitext(out_path)[0]}_means.csv"
    means.to_csv(means_path, index=False)
    if verbose:
        print(f"[CLIM] Saved baseline ({len(fc)} rows): {out_path}")
        print(f"[CLIM] Saved mean change table: {means_path}")

    return fc


def _build_persistence_intensity(
    ibtracs: pd.DataFrame,
    out_path: str,
    year: int,
    max_lead_h: int = 120,
    init_hours: tuple[int, ...] = (0, 12),
    basin: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Persistence baseline for intensity, pressure and position.

    Each case initialised in `year` is forecast as its observed USA_WIND and
    USA_PRES at the initial time, held constant for every valid time out to
    max_lead_h. The position persists the storm's forward motion: the speed and
    heading from the storm's previous record to the initial time, extrapolated
    along a great circle (constant speed, initial heading kept). A storm's first
    record, or one whose previous position is missing, has zero motion. Each
    variable is NaN where its initial value is missing. The basin filter uses
    the basin at the initial time.

    Saves the forecasts (sid, initial_time, valid_time, wind_max, pressure_min,
    lat, lon; lon in [0, 360)) to out_path and returns them.
    """
    ib = _baseline_prep(ibtracs).sort_values(["SID", "ISO_TIME"])

    # Forward motion from the previous record of the same storm. Computed before
    # any year/hour filtering so the previous record is always available.
    prev = ib.groupby("SID")[["ISO_TIME", "LAT", "LON"]].shift(1)
    dt_h = (ib["ISO_TIME"] - prev["ISO_TIME"]).dt.total_seconds() / 3600
    dist_km = metrics.haversine(
        prev["LAT"].to_numpy(),
        prev["LON"].to_numpy(),
        ib["LAT"].to_numpy(),
        ib["LON"].to_numpy(),
    )
    # Heading on arrival at the current fix: back-bearing to the previous fix, reversed
    heading = (
        metrics.bearing(
            ib["LAT"].to_numpy(),
            ib["LON"].to_numpy(),
            prev["LAT"].to_numpy(),
            prev["LON"].to_numpy(),
        )
        + np.pi
    )
    ib["speed_kmh"] = np.nan_to_num(dist_km / dt_h.to_numpy(), nan=0.0)
    ib["heading"] = np.nan_to_num(heading, nan=0.0)

    ib = ib[(ib["ISO_TIME"].dt.year == year) & ib["ISO_TIME"].dt.hour.isin(init_hours)]
    if basin is not None and basin.upper() in ["SI", "SP", "NA", "NI", "WP", "EP"]:
        ib = ib[ib["BASIN"] == basin.upper()]
    inits = ib.rename(
        columns={
            "SID": "sid",
            "ISO_TIME": "initial_time",
            "USA_WIND": "wind_max",
            "USA_PRES": "pressure_min",
            "LAT": "lat",
            "LON": "lon",
        }
    )
    fc = _expand_leads(inits, max_lead_h)
    lat, lon = _destination(
        fc["lat"].to_numpy(),
        fc["lon"].to_numpy(),
        fc["heading"].to_numpy(),
        fc["speed_kmh"].to_numpy() * fc["lead_hours"].to_numpy(),
    )
    fc["lat"] = np.round(lat, 2)
    fc["lon"] = np.round(lon % 360, 2)
    fc = fc[_BASELINE_COLS]

    fc.to_csv(out_path, index=False)
    if verbose:
        print(f"[PERSIST] Saved baseline ({len(fc)} rows): {out_path}")

    return fc


def _fetch_ibtracs(
    ibtracs_tracks_path: str,
    ibtracs_gt_path: str,
    year: int,
    basin: str,
    verbose: bool = False,
) -> pd.DataFrame:
    # read the IBTrACS data
    ibtracs = toolbox.read_hist_track_file(tracks_path=ibtracs_tracks_path)

    # Climatology-persistence and persistence intensity baselines, saved next to
    # the ground truth. The names avoid '_20xx' so _candidate_processing picks
    # them up as models.
    baseline_suffix = f"-y{year}{f'-{basin.upper()}' if basin else ''}.csv"
    baseline_dir = os.path.dirname(ibtracs_gt_path)
    _build_clim_intensity(
        ibtracs,
        os.path.join(baseline_dir, f"CLIM-intensity{baseline_suffix}"),
        year=year,
        basin=basin,
        verbose=verbose,
    )
    _build_persistence_intensity(
        ibtracs,
        os.path.join(baseline_dir, f"PERSIST-intensity{baseline_suffix}"),
        year=year,
        basin=basin,
        verbose=verbose,
    )
    # ensure correct typing
    ibtracs["ISO_TIME"] = _to_datetime(ibtracs["ISO_TIME"])
    ibtracs["USA_WIND"] = _to_numeric(ibtracs["USA_WIND"])
    ibtracs["USA_PRES"] = _to_numeric(ibtracs["USA_PRES"])
    # drop empty rows and duplicates
    ibtracs = ibtracs.dropna(subset=["SID", "ISO_TIME"]).drop_duplicates(
        subset=["SID", "ISO_TIME"]
    )
    ibtracs = ibtracs[ibtracs["ISO_TIME"].dt.year == year]

    ibtracs = _compute_ibtracs_RI(ibtracs, ri_thresh=30, ri_window_h=24)

    if basin is not None:
        basin_upper = basin.upper()
        if basin_upper not in ["SI", "SP", "NA", "NI", "WP", "EP"]:
            basin = None
            print(
                f"[RI] Warning: Invalid basin '{basin_upper}' specified. Ignoring basin filter."
            )

    if basin is not None:
        ibtracs = ibtracs[ibtracs["BASIN"] == basin].copy()

    # Create a temporary DataFrame with the initial time for each storm
    temp_df = (
        ibtracs.copy()
        .reset_index(drop=True)[["SID", "ISO_TIME"]]
        .rename(columns={"ISO_TIME": "initial_time"})
    )

    n_steps = 120 // 6  # assumes up to 5-day forecast
    offsets = pd.to_timedelta(np.arange(1, n_steps + 1) * 6, unit="h")

    # Create the reference rows
    gt_df = temp_df.loc[temp_df.index.repeat(n_steps)].reset_index(drop=True)
    gt_df["valid_time"] = gt_df["initial_time"].to_numpy() + np.tile(
        offsets, len(temp_df)
    )
    gt_df["lead_hours"] = (
        (gt_df["valid_time"] - gt_df["initial_time"]).dt.total_seconds() / 3600
    ).astype(int)

    # Filter gt_df initial times to only those at 00 and 12 UTC
    # (to match conventions for forecasting); off-synoptic IBTrACS records
    # (e.g. landfall at 12:45) are not initial times
    gt_df = gt_df[
        gt_df["initial_time"].dt.hour.isin([0, 12])
        & (gt_df["initial_time"].dt.minute == 0)
    ].copy()

    gt_df = gt_df.merge(
        ibtracs.rename(
            columns={
                "SID": "SID",
                "ISO_TIME": "valid_time",
                "USA_WIND": "wind_max",
                "USA_PRES": "pressure_min",
                "LAT": "lat",
                "LON": "lon",
            }
        ),
        on=["SID", "valid_time"],
        how="left",
    )

    gt_df = gt_df.rename(columns={"SID": "sid"})

    # RI stays NaN where there is no matching valid_time in IBTrACS, so those
    # cases are excluded from RI verification rather than counted as non-RI

    # # If wind and pressure are nan, set lat and lon to nan as well (to avoid evaluating timestamps
    # # resulting from non-us agency sources, which may have lat/lon but no US reported intensity values)
    # missing_mask = gt_df["wind_max"].isna() & gt_df["pressure_min"].isna()
    # gt_df.loc[missing_mask, ["lat", "lon"]] = np.nan

    gt_df.to_csv(ibtracs_gt_path, index=False)
    if verbose:
        print(f"Saved IBTrACS ground truth: {ibtracs_gt_path}")

    return gt_df


def _compute_ibtracs_RI(
    ib: pd.DataFrame, ri_thresh: float, ri_window_h: int, verbose: bool = False
) -> pd.DataFrame:
    """
    Compute RI flags and intensification for IBTrACS itself.
    Returns a copy with float 'RI' (1/0, NaN where intensification is undefined)
    and float 'intensification'.
    """
    ib = ib.copy()
    ib["ISO_TIME"] = _to_datetime(ib["ISO_TIME"])
    ib = ib[ib["ISO_TIME"].dt.hour.isin([0, 6, 12, 18])].copy()

    # Ensure numeric wind
    ib["USA_WIND"] = _to_numeric(ib["USA_WIND"])
    ib["USA_PRES"] = _to_numeric(ib["USA_PRES"])
    ib["LAT"] = _to_numeric(ib["LAT"])
    ib["LON"] = _to_numeric(ib["LON"])

    # --- add zero-intensity reference rows for the window before genesis ---
    genesis = (
        ib.drop_duplicates("SID", keep="first")
        .reset_index(drop=True)[["SID", "ISO_TIME"]]
        .rename(columns={"ISO_TIME": "__gen__"})
    )

    n_steps = ri_window_h // 6  # assumes 6-hourly / window multiple of 6
    offsets = pd.to_timedelta(np.arange(1, n_steps + 1) * 6, unit="h")

    # # Create the reference rows
    # pre = genesis.loc[genesis.index.repeat(n_steps)].reset_index(drop=True)
    # pre["__ri_ref_time__"] = pre["__gen__"].to_numpy() - np.tile(offsets, len(genesis))
    # pre["__ri_ref_intensity__"] = 10.0

    # Reference time (ISO_TIME - window)
    try:
        ib["__ri_ref_time__"] = ib["ISO_TIME"] - pd.Timedelta(hours=ri_window_h)
    except Exception as e:
        print("[RI][DEBUG][ERROR] Failed assigning __ri_ref_time__ in IBTrACS")
        raise

    # Build ref ONLY from the necessary source columns to avoid duplicating '__ri_ref_time__'
    ref = ib.loc[:, ["SID", "ISO_TIME", "USA_WIND"]].rename(
        columns={"ISO_TIME": "__ri_ref_time__", "USA_WIND": "__ri_ref_intensity__"}
    )
    # ref = pd.concat(
    #     [ref, pre[["SID", "__ri_ref_time__", "__ri_ref_intensity__"]]],
    #     ignore_index=True,
    # )

    ib = ib.merge(ref, on=["SID", "__ri_ref_time__"], how="left")
    ib["intensification"] = ib["USA_WIND"] - ib["__ri_ref_intensity__"]
    # NaN where either intensity is missing (e.g. first ri_window_h after genesis)
    ib["RI"] = np.where(
        ib["intensification"].notna(),
        ib["intensification"] >= float(ri_thresh),
        np.nan,
    )

    # Final dtypes (avoid pandas SettingWithCopy warnings downstream)
    ib["intensification"] = pd.to_numeric(ib["intensification"], errors="coerce")

    return ib.drop(columns=["__ri_ref_time__", "__ri_ref_intensity__"])


def _compute_RI(
    df: pd.DataFrame,
    ib: pd.DataFrame,
    RI_thresh: float,
    RI_window: int,
    ens_fraction: Optional[float] = 0.3,
    keep_intensification: bool = False,
    min_member_frac: float = 0.3,
) -> pd.DataFrame:
    """
    Compute RI flags and intensification for a model track DataFrame.
    Returns a copy with float 'RI' (1/0, NaN where it cannot be computed) and
    float 'intensification'.

    For ensembles, RI_fraction is the number of members showing RI over the
    total number of members (members without a defined RI count as non-RI),
    and RI is 1 if RI_fraction is at least ens_fraction (or any member if
    None). Cases where fewer than min_member_frac of the members have a defined
    RI are NaN, so missing forecasts are excluded from verification rather than
    counted as non-RI; RI_fraction is NaN where no member has a defined RI.
    """
    out = df.copy()

    ensemble_flag = any("ensemble" in str(c).lower() for c in out.columns)

    # calculate the reference tme (Valid Time - window) for each row
    try:
        out["t_ref"] = out["valid_time"] - pd.Timedelta(hours=RI_window)
    except Exception as e:
        print("[RI][DEBUG][ERROR] Failed assigning t_ref in model tracks")
        print(
            f"[RI][DEBUG] Columns count: {len(out.columns)} | First 25 columns: {list(out.columns)[:25]}"
        )
        raise

    # Self lookup to get the reference intensity for each row from the forecast values
    if not ensemble_flag:
        self_lookup = df[["sid", "initial_time", "valid_time", "wind_max"]].rename(
            columns={"valid_time": "t_ref", "wind_max": "self_wind"}
        )
        out = out.merge(self_lookup, on=["sid", "initial_time", "t_ref"], how="left")
    elif ensemble_flag:
        self_lookup = df[
            ["sid", "initial_time", "valid_time", "wind_max", "ensemble_idx"]
        ].rename(columns={"valid_time": "t_ref", "wind_max": "self_wind"})
        out = out.merge(
            self_lookup, on=["sid", "initial_time", "t_ref", "ensemble_idx"], how="left"
        )

    # Lookup 2: baseline table, u_ib where t_ib == t_ref
    ib_lookup = ib.copy()[["SID", "ISO_TIME", "USA_WIND"]].rename(
        columns={"SID": "sid", "ISO_TIME": "t_ref", "USA_WIND": "ib_wind"}
    )
    out = out.merge(ib_lookup, on=["sid", "t_ref"], how="left")

    # --- Conditional selection ---
    conditions = [
        out["t_ref"] > out["initial_time"],  # inside the forecast window
        out["t_ref"] <= out["initial_time"],  # at/before initialization
    ]
    choices = [out["self_wind"], out["ib_wind"]]
    out["u_ref"] = np.select(conditions, choices, default=np.nan)
    out = out.drop(columns=["self_wind", "ib_wind"])

    out["intensification"] = out["wind_max"] - out["u_ref"]
    # NaN where the forecast or reference intensity is missing
    out["RI"] = np.where(
        out["intensification"].notna(),
        out["intensification"] >= float(RI_thresh),
        np.nan,
    )

    if ensemble_flag:
        num_members = out["ensemble_idx"].nunique()
        summary = out.groupby(
            ["sid", "initial_time", "valid_time"], as_index=False
        ).agg(
            n_ri=("RI", "sum"),
            n_valid=("RI", "count"),
            intensification=("intensification", "max"),
        )
        # Fraction over all members, not only those with a defined RI
        summary["RI_fraction"] = (summary["n_ri"] / num_members).where(
            summary["n_valid"] > 0
        )
        summary.drop(columns=["n_ri"], inplace=True)

        if ens_fraction is not None:
            ri_flag = summary["RI_fraction"] >= ens_fraction
        else:
            ri_flag = summary["RI_fraction"] > 0.0
        undefined = summary["RI_fraction"].isna() | (
            summary["n_valid"] / num_members < min_member_frac
        )
        summary["RI"] = np.where(undefined, np.nan, ri_flag.astype(float))

        if not keep_intensification:
            summary.drop(columns=["intensification"], inplace=True)
            summary.drop(columns=["RI_fraction", "n_valid"], inplace=True)
        out = summary

    else:
        if not keep_intensification:
            out.drop(columns=["intensification"], inplace=True)
        out.drop(columns=["t_ref", "u_ref"], inplace=True)

    return out


def _mask_low_member_frac(
    df: pd.DataFrame,
    num_members: int,
    ensemble_fraction: float = 0.3,
    key_cols: list[str] = ["sid", "initial_time", "valid_time"],
    var_groups: list[list[str]] = [["wind_max"], ["pressure_min"], ["lat", "lon"]],
) -> pd.DataFrame:
    """
    Set ensemble predictions to NaN for cases where the fraction of members
    with a valid value is below ensemble_fraction. Each variable group is
    masked independently (lat/lon jointly, as a member needs both for a
    position), so e.g. a case can keep its wind_max while losing its position.
    """
    out = df.copy()
    for group in var_groups:
        cols = [c for c in group if c in out.columns]
        if not cols:
            continue
        valid = out[cols].notna().all(axis=1)
        member_frac = valid.groupby([out[k] for k in key_cols]).transform("sum") / (
            num_members
        )
        out.loc[member_frac < ensemble_fraction, cols] = np.nan
    return out


def _evaluate_metrics(
    ground_truth_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    deterministic_metrics: list = None,
    probabilistic_metrics: list = None,
    verbose: bool = True,
    ensemble_fraction: float = 0.3,
) -> dict:
    """
    Evaluate the specified metrics on the results DataFrame.
    Returns the computed deterministic and probabilistic metrics
    in their respective dataframes.
    """
    deterministic_df = None
    probabilistic_df = None

    if deterministic_metrics:
        for eval_metric in deterministic_metrics:
            try:
                if deterministic_df is None:
                    deterministic_df = eval_metric(ground_truth_df, prediction_df)
                else:
                    deterministic_df = eval_metric(ground_truth_df, deterministic_df)

            except Exception as e:
                print(f"[ERROR] Failed to compute {eval_metric.__name__}: {e}")

    if probabilistic_metrics:
        for eval_metric in probabilistic_metrics:
            try:
                if probabilistic_df is None:
                    probabilistic_df = eval_metric(
                        ground_truth_df,
                        prediction_df,
                        ensemble_fraction=ensemble_fraction,
                    )
                else:
                    temp_df = eval_metric(
                        ground_truth_df,
                        prediction_df,
                        ensemble_fraction=ensemble_fraction,
                    )
                    # Append only the columns probabilistic_df does not have yet
                    # (e.g. lead_hours, member_frac are shared across metrics)
                    keys = ["sid", "initial_time", "valid_time"]
                    new_cols = [
                        c for c in temp_df.columns if c not in probabilistic_df.columns
                    ]
                    probabilistic_df = probabilistic_df.merge(
                        temp_df[keys + new_cols], on=keys, how="left"
                    )
            except Exception as e:
                print(f"[ERROR] Failed to compute {eval_metric.__name__}: {e}")

    return deterministic_df, probabilistic_df

    return


def evaluate_tracks(
    eval_directory: str,
    ibtracs_tracks_path: str,
    lead_hours: list[int] = None,
    year: int = 2023,
    select_files: list[str] | None = None,
    recompute: bool = False,
    # only_ri: bool = False,
    # exclude_patterns: tuple[str, ...] = ("RI", "results", "ibtracs", "tracks"),
    # save_results: bool = True,
    # return_dataframes: bool = True,
    verbose: bool = True,
    # ri_verbose: bool | None = False,
    basin: str | None = None,
    ensemble_fraction: float = 0.3,
    ri_thresh: float = 30,
    ri_window: int = 24,
    ri_ens_fraction: float = 0.3,
):
    """
    Evaluate model track files against IBTrACS ground truth.
    Note: ibtracs refers to the initial_time valid_time pairs
    constructed with IBTrACS data, which are used to match against
    the reported tracks.
    ib_ref instead is the reference IBTrACS data used to compute RI
    and intensification for the model tracks.
    """
    save_suffix = f"_{year}_{basin}" if basin is not None else f"_{year}"
    ibtracs_gt_path = os.path.join(eval_directory, f"ibtracs_gt{save_suffix}.csv")

    ib_ref = _ib_ref_builder(ibtracs_tracks_path, year, basin=basin)
    if not recompute:
        # check if the ibtracs ground truth data exists in the eval_directory
        if os.path.exists(ibtracs_gt_path):
            # If the data exists, load it
            ibtracs = pd.read_csv(ibtracs_gt_path, dtype={"sid": str})
            ibtracs = _col_typing(ibtracs)
    else:
        # If the data doesn't exist, compute it
        ibtracs = None

    if ibtracs is None:
        ibtracs = _fetch_ibtracs(
            ibtracs_tracks_path, ibtracs_gt_path, year, basin=basin
        )

    if lead_hours is not None:
        ibtracs = ibtracs[ibtracs["lead_hours"].isin(lead_hours)].copy()

    if select_files is not None:
        track_files = [
            os.path.join(eval_directory, f) for f in select_files if f.endswith(".csv")
        ]
    else:
        track_files = _candidate_processing(
            [os.path.join(eval_directory, f) for f in os.listdir(eval_directory)]
        )

    # sort track files alphabetically for consistent processing order
    track_files.sort()

    for track_file in track_files:

        if verbose:
            print(
                f"[INFO] Evaluating track file: {track_file} against IBTrACS ground truth."
            )
        """
        Evaluate a single track file against the IBTrACS ground truth.
        Note that basin filtering is applied to the IBTrACS data, whose keys
        are used to match against the reported tracks. Thus, the basin filter
        is applied to the model predictions indirectly through the IBTrACS data.
        """
        reported_tracks = pd.read_csv(track_file, dtype={"sid": str})
        reported_tracks = _col_typing(reported_tracks)

        # filter to the specified year - cuts down on computation when merging
        reported_tracks = reported_tracks[
            reported_tracks["initial_time"].dt.year == year
        ]

        ensemble_flag = True if "ensemble_idx" in reported_tracks.columns else False
        dedupe_keys = (
            ["sid", "initial_time", "valid_time"]
            if not ensemble_flag
            else ["sid", "initial_time", "valid_time", "ensemble_idx"]
        )

        reported_tracks = _deduplicate_keys(
            reported_tracks,
            key_cols=dedupe_keys,
            model=os.path.splitext(os.path.basename(track_file))[0],
        )

        if not ensemble_flag:
            results_df = ibtracs[["sid", "initial_time", "valid_time"]].copy()
            results_df = results_df.merge(
                reported_tracks,
                on=["sid", "initial_time", "valid_time"],
                how="left",
            )
        else:
            num_members = reported_tracks["ensemble_idx"].nunique()
            # Create a DataFrame with all combinations of (sid, initial_time, valid_time) and ensemble_idx
            results_df = (
                ibtracs[["sid", "initial_time", "valid_time"]]
                .assign(key=1)
                .merge(
                    pd.DataFrame({"ensemble_idx": range(num_members)}).assign(key=1),
                    on="key",
                )
                .drop(columns="key")
            )
            results_df = results_df.merge(
                reported_tracks,
                on=["sid", "initial_time", "valid_time", "ensemble_idx"],
                how="left",
            )

        # blank metrics list to populate with the metrics to calculate
        deterministic_metrics = []
        probabilistic_metrics = []

        if "wind_max" in results_df.columns or "pressure_min" in results_df.columns:
            deterministic_metrics.append(metrics.AbsErr)
            deterministic_metrics.append(metrics.SquErr)
            if ensemble_flag:
                probabilistic_metrics.append(metrics.FairCRPS)

        if "wind_max" in results_df.columns:
            # Compute RI and intensification only if wind_max is present
            ri_results_df = _compute_RI(
                results_df,
                ib_ref,
                RI_thresh=ri_thresh,
                RI_window=ri_window,
                ens_fraction=ri_ens_fraction if ensemble_flag else None,
                keep_intensification=True,
            )

        if "lat" in results_df.columns and "lon" in results_df.columns:
            # Col typing for sanity
            results_df["lat"] = _to_numeric(results_df["lat"])
            results_df["lon"] = _to_numeric(results_df["lon"])

            deterministic_metrics.append(metrics.DirPosErr)
            deterministic_metrics.append(metrics.CartPosErr)
            if ensemble_flag:
                probabilistic_metrics.append(metrics.TrackCRPS)
                probabilistic_metrics.append(metrics.ProbCircle)

        if not deterministic_metrics and not probabilistic_metrics:
            print(
                f"[INFO] No valid metrics to compute for {track_file} found.",
                "Check if the required columns are present in the track file.",
                "Skipping this file.",
            )
            continue

        if ensemble_flag:
            # Drop cases with too few valid members so they aren't scored
            results_df = _mask_low_member_frac(
                results_df, num_members, ensemble_fraction=ensemble_fraction
            )

        det_df, prob_df = _evaluate_metrics(
            ground_truth_df=ibtracs,
            prediction_df=results_df,
            deterministic_metrics=deterministic_metrics,
            probabilistic_metrics=probabilistic_metrics,
            verbose=verbose,
            ensemble_fraction=ensemble_fraction,
        )

        if det_df is not None:
            det_out_path = os.path.join(
                eval_directory,
                f"{os.path.splitext(os.path.basename(track_file))[0]}_det{save_suffix}.csv",
            )
            det_df.to_csv(det_out_path, index=False)
            if verbose:
                print(f"[INFO] Saved deterministic metrics to {det_out_path}")

        if prob_df is not None:
            prob_out_path = os.path.join(
                eval_directory,
                f"{os.path.splitext(os.path.basename(track_file))[0]}_prob{save_suffix}.csv",
            )
            prob_df.to_csv(prob_out_path, index=False)
            if verbose:
                print(f"[INFO] Saved probabilistic metrics to {prob_out_path}")

        if ri_results_df is not None:
            ri_results_path = os.path.join(
                eval_directory,
                f"{os.path.splitext(os.path.basename(track_file))[0]}_ri{save_suffix}.csv",
            )
            ri_results_df.to_csv(ri_results_path, index=False)
            if verbose:
                print(f"[INFO] Saved RI results to {ri_results_path}")


# %% Testing
if __name__ == "__main__":
    # Run settings from the configuration file (CONFIG_PATH)
    with open(CONFIG_PATH) as f:
        _config = yaml.safe_load(f)

    # lead_hours: null (all), a list, or a start/stop/step range
    _leads = _config.get("lead_hours")
    if isinstance(_leads, dict):
        lead_hours = list(range(_leads["start"], _leads["stop"] + 1, _leads["step"]))
    else:
        lead_hours = _leads

    evaluate_tracks(
        _config["eval_directory"],
        _config["ibtracs_tracks_path"],
        lead_hours=lead_hours,
        year=_config["year"],
        recompute=_config.get("recompute", False),
        basin=_config.get("basin") or None,
        verbose=_config.get("verbose", True),
        select_files=_config.get("select_files"),
        ensemble_fraction=_config.get("ensemble_fraction", 0.3),
        ri_thresh=_config.get("ri_thresh", 30),
        ri_window=_config.get("ri_window", 24),
        ri_ens_fraction=_config.get("ri_ens_fraction", 0.3),
    )

# %%
