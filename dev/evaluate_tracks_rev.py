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


from utils import toolbox
from utils.toolbox import *
from utils import data_lib as dlib
import metrics_rev as metrics
import re


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

    # Ensure boolean columns are in boolean format
    if "RI" in df.columns:
        df["RI"] = df["RI"].astype(bool)

    return df


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


def _fetch_ibtracs(
    ibtracs_tracks_path: str, ibtracs_gt_path: str, year: int, basin: str | None = None
) -> pd.DataFrame:
    # read the IBTrACS data
    ibtracs = toolbox.read_hist_track_file(tracks_path=ibtracs_tracks_path)
    ibtracs = ibtracs[pd.to_datetime(ibtracs["ISO_TIME"]).dt.year == year]
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
    # (to match conventions for forecasting)
    gt_df = gt_df[gt_df["initial_time"].dt.hour.isin([0, 12])].copy()

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

    # Handle missing values for RI (if no matching valid_time in IBTrACS, set RI to False)
    gt_df["RI"] = gt_df["RI"].fillna(False)

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
    Returns a copy with boolean 'RI' and float 'intensification'.
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
    ib["RI"] = ib["intensification"] >= float(ri_thresh)

    # Final dtypes (avoid pandas SettingWithCopy warnings downstream)
    ib["RI"] = ib["RI"].astype(bool)
    ib["intensification"] = pd.to_numeric(ib["intensification"], errors="coerce")

    return ib.drop(columns=["__ri_ref_time__", "__ri_ref_intensity__"])


def _compute_RI(
    df: pd.DataFrame,
    ib: pd.DataFrame,
    RI_thresh: float,
    RI_window: int,
    ens_fraction: Optional[float] = 0.3,
    keep_intensification: bool = False,
) -> pd.DataFrame:
    """
    Compute RI flags and intensification for a model track DataFrame.
    Returns a copy with boolean 'RI' and float 'intensification'.
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
        out["t_ref"] < out["initial_time"],
    ]  # before initialization
    choices = [out["self_wind"], out["ib_wind"]]
    out["u_ref"] = np.select(conditions, choices, default=np.nan)
    out = out.drop(columns=["self_wind", "ib_wind"])

    out["RI"] = False
    out["intensification"] = np.nan

    out["intensification"] = out["wind_max"] - out["u_ref"]
    out["RI"] = out["intensification"] >= float(RI_thresh)

    if ensemble_flag:
        summary = out.groupby(
            ["sid", "initial_time", "valid_time"], as_index=False
        ).agg(RI_fraction=("RI", "mean"), intensification=("intensification", "max"))

        if ens_fraction is not None:
            summary["RI"] = summary["RI_fraction"] >= ens_fraction
        else:
            summary["RI"] = summary["RI_fraction"] > 0.0

        if not keep_intensification:
            summary.drop(columns=["intensification"], inplace=True)
            summary.drop(columns=["RI_fraction"], inplace=True)
        out = summary

    else:
        if not keep_intensification:
            out.drop(columns=["intensification"], inplace=True)
        out.drop(columns=["t_ref", "u_ref"], inplace=True)

    return out


def _evaluate_metrics(
    results_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    error_metrics: list,
    ensemble_flag: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Evaluate the specified metrics on the results DataFrame.
    Returns a dictionary of metric names and their computed values.
    """

    for metric in error_metrics:
        try:
            metric_name = metric.__name__

        except Exception as e:
            print(f"[ERROR] Failed to compute {metric.__name__}: {e}")

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
    save_results: bool = True,
    return_dataframes: bool = True,
    verbose: bool = True,
    ri_verbose: bool | None = False,
    basin: str | None = None,
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
            ibtracs = pd.read_csv(ibtracs_gt_path)
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
        """
        Evaluate a single track file against the IBTrACS ground truth.
        Note that basin filtering is applied to the IBTrACS data, whose keys
        are used to match against the reported tracks. Thus, the basin filter
        is applied to the model predictions indirectly through the IBTrACS data.
        """
        reported_tracks = pd.read_csv(track_file)
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
        error_metrics = []

        if "wind_max" in results_df.columns:
            # Compute RI and intensification only if wind_max is present
            ri_results_df = _compute_RI(
                results_df,
                ib_ref,
                RI_thresh=30,
                RI_window=24,
                ens_fraction=0.3 if ensemble_flag else None,
                keep_intensification=True,
            )

            error_metrics.append(metrics.AbsErr)
            error_metrics.append(metrics.SquErr)
            if ensemble_flag:
                error_metrics.append(metrics.FairCRPS)

        if "lat" in results_df.columns and "lon" in results_df.columns:
            # Col typing for sanity
            results_df["lat"] = _to_numeric(results_df["lat"])
            results_df["lon"] = _to_numeric(results_df["lon"])

            error_metrics.append(metrics.DirPosErr)
            error_metrics.append(metrics.CartPosErr)
            if ensemble_flag:
                error_metrics.append(metrics.TrackCRPS)

        results = _evaluate_metrics(
            results_df,
            ground_truth_df=ibtracs,
            error_metrics=error_metrics,
            ensemble_flag=ensemble_flag,
            verbose=verbose,
        )


# %% Testing
if __name__ == "__main__":
    # Example usage
    eval_directory = "/home/mgomezd1/repos/idea_testing/outputs_rev"
    ibtracs_tracks_path = "/home/mgomezd1/repos/idea_testing/ibtracs"
    year = 2023
    recompute = False
    basin = None
    verbose = True
    select_files = None
    lead_hours = None  # Use default lead hours if None
    # lead_hours = [6, 12, 18, 24, 48, 72, 96, 120]  # Postproc Hours
    evaluate_tracks(
        eval_directory,
        ibtracs_tracks_path,
        save_results=True,
        only_ri=False,
        lead_hours=lead_hours,
        year=year,
        recompute=recompute,
        basin=basin,
        verbose=verbose,
        select_files=select_files,
    )
