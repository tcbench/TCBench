# %%
# Persistence baseline builder (deterministic)
# ------------------------------------------------------------
# This script computes a *pure* persistence baseline directly from IBTrACS,
# on the IBTrACS 6‑hour verification grid. For each storm/time pair (SID, t0)
# and for each lead L in {6,12,...,120} hours, it forms the pair (t0, t0+L)
# **only when both timestamps exist in IBTrACS** and evaluates:
#   - DPE_GCD (km): great‑circle distance between position at t0 and t0+L
#   - AE_wind  (kt): |USA_WIND(t0+L) − USA_WIND(t0)|
#   - AE_pressure (hPa): |USA_PRES(t0+L) − USA_PRES(t0)|
#   - SE_wind  (kt^2): (USA_WIND(t0+L) − USA_WIND(t0))^2
#   - SE_pressure (hPa^2): (USA_PRES(t0+L) − USA_PRES(t0))^2
# The output is saved to `<EVAL_DIR>/persistence_results.csv` with one row per
# valid (SID, Initial Time, Valid Time) pair and a column `Hour` with the lead.
#
# This uses IBTrACS valid times every 6 h but restricts **initial times** to
# synoptic 00/12 UTC only. That is, t0 ∈ {00Z,12Z}, while t = t0+L can be any
# 6-hourly timestamp present in IBTrACS (00/06/12/18Z). It does **not** look at
# model outputs; coverage is purely determined by IBTrACS availability.
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
from datetime import timedelta

from utils import toolbox

# Default location where evaluate scripts save figures/results
EVAL_DIR = os.path.join(os.curdir, "outputs")
IB_PATH = os.path.join(os.curdir, "data", "ibtracs")

# Leads to compute (hours)
LEADS = list(range(6, 121, 6))

# Initial times to use (UTC hours). Keep verification at 6-hourly VT.
INIT_HOURS = [0, 12]

# %% helpers


def _require_dt(s):
    out = pd.to_datetime(s, errors="coerce")
    if out.isna().any():
        raise RuntimeError("IBTrACS has non‑parseable timestamps.")
    return out


# %% core


def build_persistence(
    eval_dir: str = EVAL_DIR,
    ibtracs_df: pd.DataFrame | None = None,
    leads: list[int] = LEADS,
) -> str:
    """Compute persistence baseline from IBTrACS (year 2023) and save CSV.

    Parameters
    ----------
    eval_dir : str
        Output folder where `persistence_results.csv` will be written.
    ibtracs_df : DataFrame | None
        If provided, use this preloaded IBTrACS dataframe; otherwise it is
        loaded from disk via `toolbox.read_hist_track_file`.
    leads : list[int]
        Lead times (hours) to compute.

    Returns
    -------
    str
        Full path to the written CSV.
    """
    # Load IBTrACS if needed and filter to 2023
    if ibtracs_df is None:
        ib = toolbox.read_hist_track_file(tracks_path=IB_PATH)
    else:
        ib = ibtracs_df.copy()

    if "ISO_TIME" not in ib.columns:
        raise RuntimeError("IBTrACS dataframe missing ISO_TIME column.")

    ib = ib.copy()
    ib["ISO_TIME"] = _require_dt(ib["ISO_TIME"])  # ensure datetime
    ib = ib[ib["ISO_TIME"].dt.year == 2023].reset_index(drop=True)

    # Restrict **initial** times to 00/12 UTC only; keep IBTrACS VT at 6-hourly
    ib_inits = ib[ib["ISO_TIME"].dt.hour.isin(INIT_HOURS)].reset_index(drop=True)
    if ib_inits.empty:
        raise RuntimeError(
            "No IBTrACS entries at 00/12 UTC for 2023 to serve as initial times."
        )

    # Minimal columns
    cols = ["SID", "ISO_TIME", "LAT", "LON", "USA_WIND", "USA_PRES"]
    missing = [c for c in cols if c not in ib.columns]
    if missing:
        raise RuntimeError(f"IBTrACS missing columns: {missing}")

    # Prepare left (t0) and right (t) tables once
    left0 = ib_inits.rename(
        columns={
            "ISO_TIME": "Initial Time",
            "LAT": "LAT_0",
            "LON": "LON_0",
            "USA_WIND": "WIND_0",
            "USA_PRES": "PRES_0",
        }
    )[["SID", "Initial Time", "LAT_0", "LON_0", "WIND_0", "PRES_0"]]

    rightT = ib.rename(
        columns={
            "ISO_TIME": "Valid Time",
            "LAT": "LAT_t",
            "LON": "LON_t",
            "USA_WIND": "WIND_t",
            "USA_PRES": "PRES_t",
        }
    )[["SID", "Valid Time", "LAT_t", "LON_t", "WIND_t", "PRES_t"]]

    rows = []
    forecast_rows = []
    for L in leads:
        # Compute target valid time for each t0
        left = left0.copy()
        left["Valid Time"] = left["Initial Time"] + pd.to_timedelta(L, unit="h")
        # Inner merge keeps only pairs that exist at both t0 and t0+L
        m = left.merge(rightT, on=["SID", "Valid Time"], how="inner")
        if m.empty:
            continue
        # Ensure numeric types for haversine
        m[["LAT_0", "LON_0", "LAT_t", "LON_t"]] = m[
            ["LAT_0", "LON_0", "LAT_t", "LON_t"]
        ].apply(pd.to_numeric, errors="coerce")
        # Great‑circle distance (km) between (LAT/LON)_0 and (LAT/LON)_t
        dpe = toolbox.haversine(
            m["LAT_t"].to_numpy(),
            m["LON_t"].to_numpy(),
            m["LAT_0"].to_numpy(),
            m["LON_0"].to_numpy(),
        )
        # Intensity deltas (ensure numeric), then AE and SE
        d_wind = pd.to_numeric(m["WIND_t"], errors="coerce") - pd.to_numeric(
            m["WIND_0"], errors="coerce"
        )
        d_pres = pd.to_numeric(m["PRES_t"], errors="coerce") - pd.to_numeric(
            m["PRES_0"], errors="coerce"
        )
        ae_wind = d_wind.abs()
        ae_pres = d_pres.abs()
        se_wind = d_wind**2
        se_pres = d_pres**2

        out = m[["SID", "Initial Time", "Valid Time"]].copy()
        out["Hour"] = int(L)
        out["DPE_GCD"] = dpe
        out["AE_wind"] = ae_wind.to_numpy()
        out["SE_wind"] = se_wind.to_numpy()
        out["AE_pressure"] = ae_pres.to_numpy()
        out["SE_pressure"] = se_pres.to_numpy()
        rows.append(out)

        # --- also prepare a "tracks-like" persistence forecast row for RI/CRPS tooling
        outF = m[["SID", "Initial Time", "Valid Time"]].copy()
        outF["Hour"] = int(L)
        # persistence = keep the initial state for all leads
        outF["lat"] = m["LAT_0"].to_numpy()
        outF["long"] = m["LON_0"].to_numpy()
        outF["wind max"] = pd.to_numeric(m["WIND_0"], errors="coerce").to_numpy()
        outF["pres min"] = pd.to_numeric(m["PRES_0"], errors="coerce").to_numpy()
        # also provide a common alias some loaders expect
        outF["Lead Time (h)"] = outF["Hour"]
        forecast_rows.append(outF)

    if not rows:
        raise RuntimeError("No persistence rows were created. Check IBTrACS data.")

    out_df = pd.concat(rows, ignore_index=True)
    out_df.sort_values(["SID", "Initial Time", "Valid Time"], inplace=True)

    os.makedirs(eval_dir, exist_ok=True)
    out_path = os.path.join(eval_dir, "persistence_results.csv")
    out_df.to_csv(out_path, index=False)
    print(f"✅ Saved persistence baseline: {out_path}  (rows={len(out_df)})")

    # Save a tracks-like CSV that downstream evaluators (e.g., evaluate_tracks_RI)
    # can ingest. Initial times are restricted to 00/12Z; Valid Time remains 6‑hourly.
    if forecast_rows:
        tracks_df = pd.concat(forecast_rows, ignore_index=True)
        tracks_df.sort_values(["SID", "Initial Time", "Valid Time"], inplace=True)
        tracks_path = os.path.join(eval_dir, "persistence_tracks.csv")
        tracks_df.to_csv(tracks_path, index=False)
        print(f"✅ Saved persistence tracks: {tracks_path}  (rows={len(tracks_df)})")
    else:
        raise RuntimeError("Internal error: no persistence forecast rows were created.")

    return out_path


# %% CLI entry point
if __name__ == "__main__":
    build_persistence()
