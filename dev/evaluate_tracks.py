# %% Imports
# OS and IO
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np


from utils import toolbox
from utils.toolbox import *
from utils import data_lib as dlib
import metrics_test as metrics

try:
    from evaluate_tracks_RI import evaluate_tracks_RI as _eval_RI
except Exception:
    _eval_RI = None


def evaluate_tracks(
    eval_folder: str,
    ibtracs_tracks_path: str,
    year: int = 2023,
    select_files: list[str] | None = None,
    recompute: bool = False,
    only_ri: bool = False,
    exclude_patterns: tuple[str, ...] = ("RI", "results", "ibtracs", "tracks"),
    save_results: bool = True,
    return_dataframes: bool = True,
    verbose: bool = True,
    ri_verbose: bool | None = False,
):
    """
    Evaluate track CSV files in `eval_folder`, compute metrics against IBTrACS, and
    save per-model `_results.csv` files. Designed to be called from a notebook.

    Parameters
    ----------
    eval_folder : str
        Directory containing the track CSV files and where results will be saved.
    ibtracs_tracks_path : str
        Directory of IBTrACS track files (passed to toolbox.read_hist_track_file).
    year : int, default 2023
        Filter IBTrACS to this year.
    select_files : list[str] | None
        If provided, only evaluate these filenames (must exist in `eval_folder`).
    recompute : bool
        If False (default), skip files that already have a sibling `<name>_results.csv`.
        If True, recompute all selected files and overwrite results.
    only_ri : bool
        If True, skip metric computation and only compute/save `<name>_RI.csv` files
        (for all selected files). By default False.
    exclude_patterns : tuple[str, ...]
        Filenames containing any of these substrings are skipped.
    save_results : bool
        If True, write `<name>_results.csv` into `eval_folder`.
    return_dataframes : bool
        If True, return a dict of model name -> results DataFrame.
    verbose : bool
        Print progress.
    ri_verbose : bool | None
        Controls verbosity **inside the RI subroutine** only. Default is `False` (quiet).
        If `None`, it mirrors `verbose`. Set `True` to enable RI debug prints while
        keeping `evaluate_tracks` progress logs separate.

    Note: initializations are restricted to synoptic hours {0, 12}.

    Returns
    -------
    result_paths : dict[str, str]
        Mapping from model base filename to saved results CSV path (if saved).
    result_dfs : dict[str, pd.DataFrame] | None
        Mapping from model base filename to results DataFrame (if return_dataframes is True).
    all_cases_df : pd.DataFrame
        Unique (SID, Initial Time, Valid Time) keys observed across all evaluated files.
    """
    ri_verbose = verbose if (ri_verbose is None) else ri_verbose
    # --- Load IBTrACS and filter year
    ibtracs = toolbox.read_hist_track_file(tracks_path=ibtracs_tracks_path)
    ibtracs = ibtracs[pd.to_datetime(ibtracs["ISO_TIME"]).dt.year == year]

    # --- Gather candidate files
    if select_files is None:
        candidates = [f for f in os.listdir(eval_folder) if f.endswith(".csv")]

        def _skip(fname: str) -> bool:
            low = fname.lower()
            return any(pat.lower() in low for pat in exclude_patterns)

        # Drop internal files and anything matching exclude patterns
        track_files_all = [f for f in candidates if not _skip(f)]
    else:
        track_files_all = list(select_files)

    # Check for existing outputs
    def _has_results(fname: str) -> bool:
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(eval_folder, f"{base}_results.csv")
        return os.path.exists(out_path)

    def _has_ri(fname: str) -> bool:
        base = os.path.splitext(fname)[0]
        ri_path = os.path.join(eval_folder, f"{base}_RI.csv")
        return os.path.exists(ri_path)

    # Respect recompute flag and only_ri mode
    if recompute:
        track_files = track_files_all
    else:
        if only_ri:
            track_files = [f for f in track_files_all if not _has_ri(f)]
        else:
            track_files = [f for f in track_files_all if not _has_results(f)]

    if verbose:
        if recompute:
            print(
                f"Candidates in '{eval_folder}': {len(track_files_all)} | to compute (recompute=True): {len(track_files)}"
            )
        else:
            mode = "RI missing" if only_ri else "no _results.csv yet"
            print(
                f"Candidates in '{eval_folder}': {len(track_files_all)} | to compute ({mode}): {len(track_files)}"
            )

    all_cases = []
    result_paths: dict[str, str] = {}
    result_dfs: dict[str, pd.DataFrame] = {}

    # --- Iterate files
    for track_file in track_files:
        if verbose:
            print(f"\nWorking on {track_file}...")
        filename = track_file.split(".")[0]

        # RI-only mode: compute/ensure RI and continue
        if only_ri:
            ri_path = os.path.join(eval_folder, f"{filename}_RI.csv")
            if (_eval_RI is not None) and (recompute or (not os.path.exists(ri_path))):
                if verbose:
                    action = (
                        "Recomputing"
                        if (recompute and os.path.exists(ri_path))
                        else "Creating"
                    )
                    print(f"[RI] {action} RI file for {track_file}...")
                try:
                    _ = _eval_RI(
                        ibtracs_folder=ibtracs_tracks_path,
                        results_folder=eval_folder,
                        year=year,
                        RI_thresh=34.0,
                        RI_window=24,
                        keep_intensification=False,
                        select_files=[track_file],
                        recompute=recompute,
                        verbose=ri_verbose,
                    )
                except Exception as e:
                    print(f"[RI] Warning: could not compute RI for {track_file}: {e}")
            # No results_df in RI-only mode
            result_paths[filename] = os.path.join(
                eval_folder, f"{filename}_results.csv"
            )
            continue

        postproc = "postprocessing" in track_file.lower()

        # Read the track file
        track_df = pd.read_csv(os.path.join(eval_folder, track_file))

        # Defensive: drop duplicated column labels if any (e.g., accidental 'ref_time' duplicates)
        _cols = pd.Series(track_df.columns)
        if _cols.duplicated().any():
            if verbose:
                dups = list(_cols[_cols.duplicated(keep=False)])
                print(f"Warning: duplicated columns in {track_file}: {dups} — keeping first.")
            track_df = track_df.loc[:, ~_cols.duplicated(keep='first')]

        # If post-processing file, normalize column names
        if postproc and "pres min" in track_df.columns:
            track_df = track_df.rename(columns={"pres min": "pressure min"})

        # Restrict model initializations to synoptic hours {0, 12}
        if "Initial Time" in track_df.columns:
            track_df["Initial Time"] = pd.to_datetime(
                track_df["Initial Time"], errors="coerce"
            )
            track_df = track_df[track_df["Initial Time"].dt.hour.isin({0, 12})].copy()

        # Apply IBTrACS SID filter
        track_df = track_df[track_df["SID"].isin(ibtracs["SID"].unique())]

        # Detect ensemble column (probabilistic)
        probabilistic = False
        ensemble_col = None
        for col in track_df.columns:
            if "ensemble" in str(col).lower():
                ensemble_col = col
                probabilistic = True
                break

        # Clean up duplicates for deterministic tracks
        if not probabilistic:
            track_df = track_df.drop_duplicates(
                subset=["Initial Time", "Valid Time", "SID"], keep="first"
            )

        # Representative member copy to enrich later
        if probabilistic and ensemble_col in track_df.columns:
            ensemble_idx = track_df.iloc[0][ensemble_col]
            mask_member = track_df[ensemble_col] == ensemble_idx
            result_df = track_df.loc[mask_member].copy()
            result_df = result_df.drop(columns=[ensemble_col])
        else:
            result_df = track_df.copy()

        # Pick metric list
        if postproc:
            error_metrics = [metrics.AE, metrics.SE]
            if probabilistic:
                error_metrics += [metrics.FCRPS]
        else:
            error_metrics = [metrics.DPE, metrics.AE, metrics.SE, metrics.CARTE]
            if probabilistic:
                error_metrics += [metrics.FCRPS, metrics.HCRPS]

        # Compute metrics and aggregate probabilistic ones
        for metric in error_metrics:
            if verbose:
                print(f"Calculating {metric}...")
            result = metric(reference=ibtracs, predictions=track_df)

            keys = ["Initial Time", "Valid Time", "SID"]
            if probabilistic and len(result) == len(track_df):
                arr = (
                    result
                    if getattr(result, "ndim", 1) == 2
                    else np.asarray(result)[:, None]
                )
                tmp = track_df[[ensemble_col] + keys].copy()
                for j, label in enumerate(metric.return_labels):
                    tmp[label] = arr[:, j]
                grp = tmp.groupby(keys, dropna=False)
                mean_df = grp[metric.return_labels].mean().reset_index()
                std_df = grp[metric.return_labels].std().reset_index()
                for label in metric.return_labels:
                    std_df.loc[std_df[label] == 0, label] = np.nan
                # merge back means/stds as *_mean / *_std
                result_df = result_df.merge(
                    mean_df, on=keys, how="left", suffixes=("", "")
                )
                result_df = result_df.merge(
                    std_df, on=keys, how="left", suffixes=("", "_stdsrc")
                )
                for label in metric.return_labels:
                    if label in result_df.columns:
                        result_df.rename(columns={label: label + "_mean"}, inplace=True)
                    stdcol = label + "_stdsrc"
                    if stdcol in result_df.columns:
                        result_df.rename(columns={stdcol: label + "_std"}, inplace=True)
            else:
                # Deterministic outputs: assign directly
                if getattr(result, "ndim", 1) == 1:
                    result_df[metric.return_labels[0]] = np.asarray(result)
                else:
                    for j, label in enumerate(metric.return_labels):
                        result_df[label] = np.asarray(result)[:, j]

        # Accumulate evaluated keys for persistence baseline
        if not only_ri:
            key_cols = ["SID", "Initial Time", "Valid Time"]
            have_keys = [c for c in key_cols if c in result_df.columns]
            if len(have_keys) == 3:
                tmp_keys = (
                    result_df[have_keys]
                    .dropna(
                        subset=["Initial Time", "Valid Time"]
                    )  # require both times present
                    .drop_duplicates()
                )
                all_cases.append(tmp_keys)

        # Save results (skip when only_ri=True)
        out_path = os.path.join(eval_folder, f"{filename}_results.csv")
        if (not only_ri) and save_results:
            if verbose and recompute and os.path.exists(out_path):
                print(f"Overwriting existing results: {out_path}")
            result_df.to_csv(out_path, index=False)

        # Ensure RI file exists for this track (all model types)
        ri_path = os.path.join(eval_folder, f"{filename}_RI.csv")
        if (_eval_RI is not None) and (recompute or (not os.path.exists(ri_path))):
            if verbose:
                action = (
                    "Recomputing"
                    if (recompute and os.path.exists(ri_path))
                    else "Creating"
                )
                print(f"[RI] {action} RI file for {track_file}...")
            try:
                _ = _eval_RI(
                    ibtracs_folder=ibtracs_tracks_path,
                    results_folder=eval_folder,
                    year=year,
                    RI_thresh=34.0,
                    RI_window=24,
                    keep_intensification=False,
                    select_files=[track_file],
                    recompute=recompute,
                    verbose=ri_verbose,
                )
            except Exception as e:
                print(f"[RI] Warning: could not compute RI for {track_file}: {e}")

        result_paths[filename] = out_path
        if (not only_ri) and return_dataframes:
            result_dfs[filename] = result_df

    # Combine all unique cases
    all_cases_df = (
        pd.concat(all_cases, ignore_index=True).drop_duplicates()
        if all_cases and (not only_ri)
        else pd.DataFrame(columns=["SID", "Initial Time", "Valid Time"])
    )

    return result_paths, (result_dfs if return_dataframes else None), all_cases_df


if __name__ == "__main__":
    # Backward-compatible script entrypoint with current defaults
    eval_folder = os.path.join(os.curdir, "outputs")
    ibtracs_dir = os.path.join(os.curdir, "data", "ibtracs")
    paths, dfs, keys = evaluate_tracks(
        eval_folder=eval_folder,
        ibtracs_tracks_path=ibtracs_dir,
        year=2023,
        select_files=None,
        recompute=False,
        only_ri=False,
        exclude_patterns=("RI", "results", "ibtracs"),
        save_results=True,
        return_dataframes=False,
        verbose=True,
    )
    print("\nSaved:")
    for k, p in paths.items():
        print(f"  {k}: {p}")
    print(f"\nUnique evaluated cases: {len(keys)} rows")
