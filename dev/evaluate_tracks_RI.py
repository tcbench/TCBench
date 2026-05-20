# %% Imports
import os
from typing import List, Optional, Dict

import pandas as pd
import numpy as np

from utils import toolbox


def _debug_dupe_cols(df: pd.DataFrame, label: str):
    cols = pd.Series(df.columns)
    if cols.duplicated().any():
        dups = list(cols[cols.duplicated(keep=False)])
        print(f"[RI][DEBUG] Duplicate columns in {label}: {dups}")


def _to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _to_numeric(s: pd.Series) -> pd.Series:
    # Robust numeric coercion (handles strings with blanks)
    if s.dtype == object:
        s = s.astype(str).str.strip().replace({"": np.nan})
    return pd.to_numeric(s, errors="coerce")


def _dedupe_columns(
    df: pd.DataFrame, context: str = "", verbose: bool = False
) -> pd.DataFrame:
    """Drop duplicated column labels (keep first) and optionally log them."""
    cols = pd.Series(df.columns)
    if cols.duplicated().any():
        if verbose:
            dups = list(cols[cols.duplicated(keep=False)])
            print(
                f"[RI] Removing duplicated columns in {context}: {dups} (keeping first occurrence)"
            )
        df = df.loc[:, ~cols.duplicated(keep="first")]
    return df


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

    if verbose:
        print("[RI][DEBUG] _compute_ibtracs_RI: start")
        _debug_dupe_cols(ib, "IBTrACS (pre-assign)")
        pre_helpers = [
            c for c in ["__ri_ref_time__", "__ri_ref_intensity__"] if c in ib.columns
        ]
        if pre_helpers:
            print(
                f"[RI][DEBUG] IBTrACS already contains helper columns before assignment: {pre_helpers}"
            )

    # Ensure numeric wind
    ib["USA_WIND"] = _to_numeric(ib["USA_WIND"])

    # Reference time (ISO_TIME - window)
    try:
        ib["__ri_ref_time__"] = ib["ISO_TIME"] - pd.Timedelta(hours=ri_window_h)
    except Exception as e:
        print("[RI][DEBUG][ERROR] Failed assigning __ri_ref_time__ in IBTrACS")
        _debug_dupe_cols(ib, "IBTrACS at failure adding __ri_ref_time__")
        print(
            f"[RI][DEBUG] Columns count: {len(ib.columns)} | First 25 columns: {list(ib.columns)[:25]}"
        )
        raise

    # Build ref ONLY from the necessary source columns to avoid duplicating '__ri_ref_time__'
    ref = ib.loc[:, ["SID", "ISO_TIME", "USA_WIND"]].rename(
        columns={"ISO_TIME": "__ri_ref_time__", "USA_WIND": "__ri_ref_intensity__"}
    )
    if verbose:
        _debug_dupe_cols(ref, "IBTrACS ref (post-rename/subset)")

    ib = ib.merge(ref, on=["SID", "__ri_ref_time__"], how="left")
    ib["intensification"] = ib["USA_WIND"] - ib["__ri_ref_intensity__"]
    ib["RI"] = ib["intensification"] >= float(ri_thresh)

    # Final dtypes (avoid pandas SettingWithCopy warnings downstream)
    ib["RI"] = ib["RI"].astype(bool)
    ib["intensification"] = pd.to_numeric(ib["intensification"], errors="coerce")

    return ib.drop(columns=["__ri_ref_time__", "__ri_ref_intensity__"])


def _build_temp_index(
    df: pd.DataFrame, time_col_init: str, time_col_valid: str
) -> pd.Series:
    """
    temp_index = "<init> tau <lead_hours>"
    """
    lead_h = (
        (_to_datetime(df[time_col_valid]) - _to_datetime(df[time_col_init]))
        .dt.total_seconds()
        .div(3600.0)
        .round()
        .astype("Int64")
    )
    init_str = _to_datetime(df[time_col_init]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return init_str + " tau " + lead_h.astype(str)


def evaluate_tracks_RI(
    ibtracs_folder: str,
    results_folder: str,
    year: int = 2023,
    RI_thresh: float = 34.0,
    RI_window: int = 24,
    keep_intensification: bool = False,
    select_files: Optional[List[str]] = None,
    recompute: bool = False,
    verbose: bool = False,
    ri_verbose: bool | None = None,
) -> Dict[str, str]:
    """
    Compute/ensure <name>_RI.csv files for model track CSVs in `results_folder`.

    Parameters
    ----------
    ibtracs_folder : str
        Folder containing 'ibtracs.ALL.list.v04r01.csv' (read via toolbox).
    results_folder : str
        Folder containing model track CSVs and where _RI.csv will be written.
    year : int
        Filter IBTrACS to this year.
    RI_thresh : float
        RI threshold in knots over the RI_window.
    RI_window : int
        RI window in hours (typically 24).
    keep_intensification : bool
        If True, also write 'intensification' column to the _RI.csv.
    select_files : list[str] | None
        If provided, only process these filenames (must exist in results_folder).
    recompute : bool
        If False, skip files that already have <name>_RI.csv.
    verbose : bool
        Print progress if True.
    ri_verbose : bool | None
        Controls verbosity **inside the RI subroutine** only. If `None` (default),
        it mirrors `verbose`. Set `False` to suppress RI debug prints while keeping
        `evaluate_tracks` progress logs.

    Returns
    -------
    out_map : dict[str, str]
        Mapping model base filename -> path to the saved <name>_RI.csv.
    """
    ri_verbose = verbose if (ri_verbose is None) else ri_verbose

    # --- Load and filter IBTrACS
    ib = toolbox.read_hist_track_file(tracks_path=ibtracs_folder)
    ib["ISO_TIME"] = _to_datetime(ib["ISO_TIME"])
    ib = ib[ib["ISO_TIME"].dt.year == int(year)].copy()

    # Defensive: IBTrACS sometimes carries duplicate helper columns from prior merges
    ib = _dedupe_columns(ib, context="IBTrACS", verbose=ri_verbose)
    if ri_verbose:
        _debug_dupe_cols(ib, "IBTrACS after global dedupe")

    # Pre-compute IBTrACS RI/intensification
    ib_ri = _compute_ibtracs_RI(ib, RI_thresh, RI_window, verbose=ri_verbose)

    # --- Select candidate files
    if select_files is None:
        candidates = [f for f in os.listdir(results_folder) if f.endswith(".csv")]

        def _skip(fname: str) -> bool:
            low = fname.lower()
            return any(
                p in low for p in ("_ri", "_results", "ibtracs")
            )  # skip derived files & ibtracs

        track_files = [f for f in candidates if not _skip(f)]
    else:
        track_files = (
            [select_files] if isinstance(select_files, str) else list(select_files)
        )

    out_map: Dict[str, str] = {}

    for track_file in track_files:
        base = os.path.splitext(track_file)[0]
        out_path = os.path.join(results_folder, f"{base}_RI.csv")

        if (not recompute) and os.path.exists(out_path):
            out_map[base] = out_path
            if verbose:
                print(f"[RI] Exists, skip: {track_file}")
            continue

        # Load model tracks
        path = os.path.join(results_folder, track_file)
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            if verbose:
                print(f"[RI] Failed to read {track_file}: {e}")
            continue

        # Defensive: drop duplicated column labels (e.g., stray 'ref_time' columns)
        df = _dedupe_columns(df, context=track_file, verbose=ri_verbose)

        if ri_verbose:
            _debug_dupe_cols(df, f"input file {track_file} (post-dedupe)")
            if "__ri_ref_time__" in df.columns:
                print(
                    f"[RI][DEBUG] '__ri_ref_time__' already present in {track_file} BEFORE per-SID processing"
                )

        # Basic sanity
        need = {"SID", "Initial Time", "Valid Time", "wind max"}
        missing = need - set(df.columns)
        if missing:
            if verbose:
                print(f"[RI] Missing columns in {track_file}: {missing} (skip)")
            continue

        # Normalize dtypes
        df["SID"] = df["SID"].astype(str)
        df["Initial Time"] = _to_datetime(df["Initial Time"])
        df["Valid Time"] = _to_datetime(df["Valid Time"])
        df["wind max"] = _to_numeric(df["wind max"])

        # Determine if probabilistic via an 'ensemble' column
        ensemble_col = next(
            (c for c in df.columns if "ensemble" in str(c).lower()), None
        )

        # Prepare output columns with correct dtypes
        df_out = df.copy()
        # Ensure boolean dtype to avoid pandas FutureWarning on assignment
        df_out["RI"] = pd.Series(False, index=df_out.index, dtype=bool)
        if keep_intensification:
            df_out["intensification"] = pd.Series(
                np.nan, index=df_out.index, dtype=float
            )

        if ri_verbose:
            _debug_dupe_cols(df_out, f"df_out initial copy for {track_file}")

        # Process per-SID to keep merges smaller
        for sid, df_sid in df_out.groupby("SID", sort=False):
            df_sid = df_sid.copy()

            if ri_verbose:
                _debug_dupe_cols(df_sid, f"SID {sid} pre-ref_time assignment")
                if "__ri_ref_time__" in df_sid.columns:
                    print(
                        f"[RI][DEBUG] '__ri_ref_time__' pre-exists in slice for SID {sid} BEFORE assignment"
                    )

            try:
                # ref_time = Valid Time - RI_window
                df_sid["__ri_ref_time__"] = df_sid["Valid Time"] - pd.Timedelta(
                    hours=RI_window
                )
            except Exception as e:
                # Print detailed state and re-raise to preserve original failure behavior
                print(
                    f"[RI][DEBUG][ERROR] While assigning __ri_ref_time__ for SID {sid} in {track_file}"
                )
                print(
                    f"[RI][DEBUG] Columns count: {len(df_sid.columns)} | Columns: {list(df_sid.columns)}"
                )
                _debug_dupe_cols(df_sid, f"SID {sid} at failure")
                raise

            sid_ref = df_sid.copy()
            if ri_verbose and "temp_index" in sid_ref.columns:
                print(
                    f"[RI][DEBUG] 'temp_index' unexpectedly present in sid_ref before build for SID {sid}"
                )

            sid_ref["temp_index"] = _build_temp_index(
                sid_ref, "Initial Time", "Valid Time"
            )
            if ensemble_col is not None:
                sid_ref["temp_index"] = (
                    sid_ref["temp_index"]
                    + " ens_idx "
                    + sid_ref[ensemble_col].astype(str)
                )

            if ri_verbose:
                dup_temp = sid_ref["temp_index"].duplicated().any()
                if dup_temp:
                    print(
                        f"[RI][DEBUG] Duplicate temp_index values within SID {sid} (this is expected when multiple leads share same key; merge should still be left-join safe)"
                    )

            # temp index corresponding to the reference lead (i.e., (__ri_ref_time__ - Initial Time))
            df_sid["temp_index"] = (
                _to_datetime(df_sid["Initial Time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                + " tau "
                + (
                    (
                        _to_datetime(df_sid["__ri_ref_time__"])
                        - _to_datetime(df_sid["Initial Time"])
                    )
                    .dt.total_seconds()
                    .div(3600.0)
                    .round()
                    .astype("Int64")
                    .astype(str)
                )
            )
            if ensemble_col is not None:
                df_sid["temp_index"] = (
                    df_sid["temp_index"]
                    + " ens_idx "
                    + df_sid[ensemble_col].astype(str)
                )

            # Merge self-reference "wind max ref"
            df_sid = df_sid.merge(
                sid_ref[["temp_index", "wind max"]].rename(
                    columns={"wind max": "wind max ref"}
                ),
                on="temp_index",
                how="left",
                suffixes=("", ""),
            )

            # For rows where __ri_ref_time__ == Initial Time, backfill from IBTrACS USA_WIND at that time
            needs_ib = _to_datetime(df_sid["__ri_ref_time__"]) == _to_datetime(
                df_sid["Initial Time"]
            )
            if needs_ib.any():
                ib_ref_sid = ib_ri[ib_ri["SID"].astype(str) == str(sid)].rename(
                    columns={"ISO_TIME": "Valid Time"}
                )[["Valid Time", "USA_WIND"]]
                ib_join = df_sid.loc[needs_ib, ["Valid Time"]].merge(
                    ib_ref_sid, on="Valid Time", how="left"
                )
                # Numeric wind
                ref_vals = _to_numeric(ib_join["USA_WIND"]).values
                df_sid.loc[needs_ib, "wind max ref"] = ref_vals

            # Compute intensification and RI
            intens = df_sid["wind max"] - df_sid["wind max ref"]
            ri_flag = intens >= float(RI_thresh)

            # Write back into the original output frame indexes
            idx = df_sid.index
            df_out.loc[idx, "RI"] = ri_flag.values
            if keep_intensification:
                df_out.loc[idx, "intensification"] = intens.values

        # Persist to disk
        df_out.to_csv(out_path, index=False)
        out_map[base] = out_path
        if verbose:
            print(f"[RI] Saved: {out_path}")

    return out_map


if __name__ == "__main__":
    # Optional CLI entry point for manual use
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute <name>_RI.csv for model tracks."
    )
    parser.add_argument("--ibtracs_folder", type=str, required=True)
    parser.add_argument("--results_folder", type=str, required=True)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--RI_thresh", type=float, default=34.0)
    parser.add_argument("--RI_window", type=int, default=24)
    parser.add_argument("--keep_intensification", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--select_files",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of CSV filenames to process.",
    )
    args = parser.parse_args()

    _ = evaluate_tracks_RI(
        ibtracs_folder=args.ibtracs_folder,
        results_folder=args.results_folder,
        year=args.year,
        RI_thresh=args.RI_thresh,
        RI_window=args.RI_window,
        keep_intensification=args.keep_intensification,
        select_files=args.select_files,
        recompute=args.recompute,
        verbose=args.verbose,
        ri_verbose=None,
    )
