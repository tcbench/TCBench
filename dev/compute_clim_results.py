# %% Pseudo-climatological baseline from IBTrACS empirical deltas (FAST/LOW-MEM)
#
# This script builds a probabilistic baseline by sampling empirical deltas
# (intensification_kt, intensification_hpa, lat_change, lon_change) from a historical pool
# conditional on (Basin, lead_time), and applying them to each 2023 IBTrACS
# forecast case (t0 -> t0+lead). It computes CRPS metrics *directly from the
# samples* per case (O(m log m)), without materializing a giant predictions
# table. Optionally it also writes simple deterministic baselines by taking
# the ensemble mean of sampled states (for DPE/AE/SE).
# Includes pressure deltas (intensification_hpa) and CRPS for pressure.
#
# Output: {EVAL_DIR}/2023_climatology_results.csv
# Columns are aligned with plotting code (e.g. CRPS_haversine_mean, CRPS_vmax_mean, CRPS_pmin_mean,
# and their _std placeholders). Deterministic columns (DPE_GCD_mean, AE_*, SE_*)
# are included when WRITE_DET_MEAN=True.
# Initial times are restricted to 00Z and 12Z; valid times are kept at 6-hourly resolution (00/06/12/18Z).

import os
import numpy as np
import pandas as pd
from utils import toolbox

# ------------------ CONFIG ------------------
EVAL_DIR = os.environ.get(
    "TCBENCH_EVAL_DIR",
    os.path.join(os.curdir, "outputs"),
)
CLIM_CSV = os.environ.get(
    "IBTRACS_CLIM_CSV",
    os.path.join(EVAL_DIR, "ibtracs_clim.csv"),
)
# Cap samples per (basin, lead) per case to keep memory/CPU in check
MAX_SAMPLES_PER_LEAD = int(os.environ.get("CLIM_MAX_SAMPLES", 1000))  # ↓ default 400
RANDOM_SEED = int(os.environ.get("CLIM_RANDOM_SEED", 42))
# Optional deterministic outputs from ensemble mean
WRITE_DET_MEAN = True

# Lead hours to consider. By default, infer from climatology file.
EXPLICIT_LEADS = None  # e.g., [6, 12, 18, ..., 120]

# Restrict initialization times to synoptic 00Z/12Z (valid times remain 6-hourly)
INIT_HOURS = [0, 12]

# -------------------------------------------
# Helpers


def _coerce_numeric_cols(df: pd.DataFrame, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _to_float_safe(val):
    try:
        return float(val)
    except Exception:
        try:
            v = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
            return float(v) if pd.notna(v) else np.nan
        except Exception:
            return np.nan


def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("Initial Time", "Valid Time", "ISO_TIME"):
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")
    return out


def _read_ibtracs_2023() -> pd.DataFrame:
    ib = toolbox.read_hist_track_file(
        tracks_path=os.path.join(os.curdir, "data", "ibtracs")
    )
    ib = ib[ib["ISO_TIME"].dt.year == 2023].copy()
    keep = ["SID", "ISO_TIME", "LAT", "LON", "USA_WIND", "USA_PRES", "BASIN"]
    keep = [c for c in keep if c in ib.columns]
    ib = ib[keep].copy()
    ib = _coerce_numeric_cols(ib, ["LAT", "LON", "USA_WIND", "USA_PRES"])
    return ib


def _build_cases(ib2023: pd.DataFrame, leads: list[int]) -> pd.DataFrame:
    """Build all (SID, t0, t0+lead) pairs that exist in IBTrACS 2023.
    Returns columns: SID, Initial Time, Valid Time, Basin,
    LAT_0, LON_0, WIND_0, PRES_0, LAT_t, LON_t, WIND_t, PRES_t, Hour
    """
    ib = ib2023.copy().sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)

    # Filter INITIAL times to 00Z/12Z for synoptic starts; keep RIGHT side unfiltered
    ib_left = ib[ib["ISO_TIME"].dt.hour.isin(INIT_HOURS)].copy()
    left = ib_left.rename(
        columns={
            "ISO_TIME": "Initial Time",
            "LAT": "LAT_0",
            "LON": "LON_0",
            "USA_WIND": "WIND_0",
            "USA_PRES": "PRES_0",
        }
    )
    right = ib.rename(
        columns={
            "ISO_TIME": "Valid Time",
            "LAT": "LAT_t",
            "LON": "LON_t",
            "USA_WIND": "WIND_t",
            "USA_PRES": "PRES_t",
        }
    )
    left = _coerce_numeric_cols(left, ["LAT_0", "LON_0", "WIND_0", "PRES_0"])
    right = _coerce_numeric_cols(right, ["LAT_t", "LON_t", "WIND_t", "PRES_t"])

    rows = []
    right = right.set_index(["SID", "Valid Time"])  # fast lookups

    for sid, grp in left.groupby("SID"):
        basin = grp["BASIN"].iloc[0] if "BASIN" in grp.columns else np.nan
        times = grp["Initial Time"].values
        lat0 = grp.get("LAT_0").values
        lon0 = grp.get("LON_0").values
        w0 = grp.get("WIND_0").values
        p0 = (
            grp.get("PRES_0").values
            if "PRES_0" in grp.columns
            else np.full(len(times), np.nan)
        )
        for i, t0 in enumerate(times):
            for lh in leads:
                vt = pd.Timestamp(t0) + pd.to_timedelta(int(lh), unit="h")
                key = (sid, vt)
                if key in right.index:
                    r = right.loc[key]
                    if isinstance(r, pd.DataFrame):
                        r = r.iloc[0]
                    rows.append(
                        {
                            "SID": sid,
                            "Initial Time": pd.Timestamp(t0),
                            "Valid Time": pd.Timestamp(vt),
                            "Basin": basin,
                            "LAT_0": _to_float_safe(lat0[i]),
                            "LON_0": _to_float_safe(lon0[i]),
                            "WIND_0": _to_float_safe(w0[i]),
                            "PRES_0": _to_float_safe(p0[i]),
                            "LAT_t": _to_float_safe(r.get("LAT_t", np.nan)),
                            "LON_t": _to_float_safe(r.get("LON_t", np.nan)),
                            "WIND_t": _to_float_safe(r.get("WIND_t", np.nan)),
                            "PRES_t": _to_float_safe(r.get("PRES_t", np.nan)),
                            "Hour": int(lh),
                        }
                    )
    # Debug: how many synoptic inits did we keep?
    n_inits = len(left.drop_duplicates(subset=["SID", "Initial Time"]))
    print(f"[CLIM] Inits @00/12Z kept: {n_inits}; total cases built: {len(rows)}")
    return pd.DataFrame(rows)


def _load_climatology_pool(path: str, exclude_year=2023) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Climatology CSV not found: {path}")
    clim = pd.read_csv(path)

    # If time columns exist, drop rows from the eval year
    for c in ["Valid Time", "Initial Time", "Storm Start"]:
        if c in clim.columns:
            clim[c] = pd.to_datetime(clim[c], errors="coerce")
    if "Valid Time" in clim.columns:
        clim = clim[clim["Valid Time"].dt.year != exclude_year]
    elif "Initial Time" in clim.columns:
        clim = clim[clim["Initial Time"].dt.year != exclude_year]

    required = [
        "Basin",
        "lead_time",
        "intensification_kt",
        "intensification_hpa",
        "lat_change",
        "lon_change",
    ]
    missing = [c for c in required if c not in clim.columns]
    if missing:
        raise ValueError(f"Climatology CSV missing columns: {missing}")
    return clim


def _build_pool_dict(clim: pd.DataFrame):
    """Pre-group climatology pool: (Basin, lead) -> dict of numpy arrays."""
    clim = clim.copy()
    clim["lead_time"] = clim["lead_time"].astype(int)
    pool = {}
    for (basin, lead), g in clim.groupby(["Basin", "lead_time"], sort=False):
        pool[(basin, int(lead))] = {
            "dV": g["intensification_kt"].to_numpy(dtype=float, copy=True),
            "dP": g["intensification_hpa"].to_numpy(dtype=float, copy=True),
            "dlat": g["lat_change"].to_numpy(dtype=float, copy=True),
            "dlon": g["lon_change"].to_numpy(dtype=float, copy=True),
        }
    return pool


# --- fast CRPS utilities -----------------------------------------------------


def _haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(a))


def _crps_ensemble_univariate(samples: np.ndarray, y: float) -> float:
    """Fair CRPS for a finite ensemble using O(m log m) formula.
    CRPS = (1/m) * sum |x_i - y| - (1/(2 m (m-1))) * sum_{i!=j} |x_i - x_j|
    The second term is computed from the sorted samples without pairwise O(m^2).
    Returns np.nan if m < 1 or all-NaN.
    """
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    m = x.size
    if m == 0:
        return np.nan
    term1 = np.mean(np.abs(x - y))
    if m == 1:
        return float(term1)
    xs = np.sort(x)
    i = np.arange(1, m + 1, dtype=float)
    S = np.sum((2 * i - m - 1) * xs)  # equals sum_{i<j} (x_j - x_i)
    mean_pair_abs = (2.0 / (m * (m - 1))) * S
    return float(term1 - 0.5 * mean_pair_abs)


# --- deterministic CARTE (Along/Cross) helper --------------------------------


def _carte_from_point(lat0, lon0, lat_t, lon_t, lat_p, lon_p):
    """Compute Along-Track Error, Cross-Track Error, and segment-relative DPE (km)
    using a local equirectangular projection around the start point.
    If the reference segment is degenerate, returns (0, 0, haversine).
    """
    # Convert degrees → radians
    R = 6371.0
    lat0r = np.radians(lat0)
    lon0r = np.radians(lon0)
    lattr = np.radians(lat_t)
    lontr = np.radians(lon_t)
    latpr = np.radians(lat_p)
    lonpr = np.radians(lon_p)

    # Local equirectangular about start point
    def _xy(lat_r, lon_r):
        dx = R * (lon_r - lon0r) * np.cos(lat0r)
        dy = R * (lat_r - lat0r)
        return dx, dy

    rsx, rsy = 0.0, 0.0  # start at origin
    rex, rey = _xy(lattr, lontr)  # reference end
    px, py = _xy(latpr, lonpr)  # predicted point

    dx = rex - rsx
    dy = rey - rsy
    seg_norm2 = dx * dx + dy * dy
    if not np.isfinite(seg_norm2) or seg_norm2 < 1e-12:
        # Degenerate segment: fall back to direct great-circle distance
        dpe = _haversine_km(lat_p, lon_p, lat_t, lon_t)
        return 0.0, 0.0, float(dpe)

    # Projection scalar along the ref segment
    t = ((px - rsx) * dx + (py - rsy) * dy) / seg_norm2
    # Vector from projection on the segment to the predicted point
    projx = rsx + t * dx
    projy = rsy + t * dy
    cx = px - projx
    cy = py - projy

    seg_len = np.sqrt(seg_norm2)
    along_err = (t - 1.0) * seg_len  # signed along difference vs true end
    cross_err = np.sign(cx * (-dy) + cy * dx) * np.hypot(cx, cy)  # signed cross
    dpe_cart = float(np.hypot(along_err, cross_err))
    return float(along_err), float(cross_err), dpe_cart


# ---------------------------------------------------------------------------


def build_climatology_results():
    rng = np.random.default_rng(RANDOM_SEED)

    # 1) Load references and climatology
    ib2023 = _read_ibtracs_2023()
    clim = _load_climatology_pool(CLIM_CSV)

    # infer leads from climatology if not given
    if EXPLICIT_LEADS is None:
        leads = sorted(int(x) for x in pd.unique(clim["lead_time"].astype(int)))
    else:
        leads = list(EXPLICIT_LEADS)

    # 2) Build all valid (SID, t0, vt) cases in 2023 and keep only cases with t0 position
    cases = _build_cases(ib2023, leads)
    if cases.empty:
        raise RuntimeError("No (SID, t0, vt) pairs found in IBTrACS 2023.")
    cases = cases.dropna(subset=["LAT_0", "LON_0"]).reset_index(drop=True)

    # 3) Pre-group pool for fast lookup
    pool = _build_pool_dict(clim)

    # After building `pool`
    for lead in [24, 72, 120]:
        counts = {
            b: len(pool.get((b, lead), {}).get("dV", []))
            for b in pd.unique(cases["Basin"])
        }
        print(f"Pool sizes at {lead}h:", counts)

    # 4) Loop cases and compute CRPS directly from samples (no giant preds table)
    out_rows = []
    for _, row in cases.iterrows():
        key = (row["Basin"], int(row["Hour"]))
        if key not in pool:
            continue  # no empirical pool for this basin/lead
        p = pool[key]
        n = p["dV"].size
        if n == 0:
            continue
        # Draw without replacement up to MAX_SAMPLES_PER_LEAD
        if n > MAX_SAMPLES_PER_LEAD:
            idx = rng.choice(n, size=MAX_SAMPLES_PER_LEAD, replace=False)
            dV = p["dV"][idx]
            dlat = p["dlat"][idx]
            dlon = p["dlon"][idx]
            dP = p["dP"][idx]
        else:
            dV, dlat, dlon, dP = p["dV"], p["dlat"], p["dlon"], p["dP"]

        # Apply deltas to t0 state (persistence + delta)
        lat_s = float(row["LAT_0"]) + dlat
        lon_s = float(row["LON_0"]) + dlon
        wind0 = float(row["WIND_0"]) if np.isfinite(row["WIND_0"]) else np.nan
        wind_s = (
            wind0 + dV if np.isfinite(wind0) else np.full_like(dV, np.nan, dtype=float)
        )
        # Pressure: apply dP to p0 if p0 is finite, else nan array
        p0 = float(row["PRES_0"]) if np.isfinite(row["PRES_0"]) else np.nan
        if np.isfinite(p0):
            pres_s = p0 + dP
        else:
            pres_s = np.full_like(dP, np.nan, dtype=float)

        # Truth at valid time
        lat_t, lon_t = float(row["LAT_t"]), float(row["LON_t"])
        w_t = float(row["WIND_t"]) if pd.notna(row["WIND_t"]) else np.nan
        p_t = float(row["PRES_t"]) if pd.notna(row["PRES_t"]) else np.nan

        # Track CRPS: distances to truth, y=0
        dists = _haversine_km(lat_s, lon_s, lat_t, lon_t)
        crps_h = _crps_ensemble_univariate(dists, 0.0)

        # Intensity CRPS: wind
        crps_v = _crps_ensemble_univariate(wind_s, w_t) if np.isfinite(w_t) else np.nan

        # Pressure CRPS
        crps_p = _crps_ensemble_univariate(pres_s, p_t) if np.isfinite(p_t) else np.nan

        # Optional deterministic metrics from ensemble mean (optional)
        dpe_mean = ae_wind = se_wind = np.nan
        along_te = cross_te = dpe_cart = np.nan
        ae_pres = se_pres = np.nan
        if WRITE_DET_MEAN:
            lat_m = float(np.nanmean(lat_s))
            lon_m = float(np.nanmean(lon_s))
            dpe_mean = float(_haversine_km(lat_m, lon_m, lat_t, lon_t))
            # CARTE (deterministic) from ensemble-mean position
            try:
                a, c, dc = _carte_from_point(
                    lat0=row["LAT_0"],
                    lon0=row["LON_0"],
                    lat_t=lat_t,
                    lon_t=lon_t,
                    lat_p=lat_m,
                    lon_p=lon_m,
                )
                along_te, cross_te, dpe_cart = a, c, dc
            except Exception:
                pass
            # Intensity AE/SE from ensemble-mean wind
            if np.isfinite(w_t) and np.isfinite(wind_s).any():
                w_m = float(np.nanmean(wind_s))
                ae_wind = abs(w_m - w_t)
                se_wind = (w_m - w_t) ** 2
            # Pressure AE/SE from ensemble-mean pressure
            if np.isfinite(p_t) and np.isfinite(pres_s).any():
                p_m = float(np.nanmean(pres_s))
                ae_pres = abs(p_m - p_t)
                se_pres = (p_m - p_t) ** 2

        out_rows.append(
            {
                "SID": row["SID"],
                "Initial Time": row["Initial Time"],
                "Valid Time": row["Valid Time"],
                "Hour": int(row["Hour"]),
                # Probabilistic metrics (mean column names kept for compatibility)
                "CRPS_haversine_mean": crps_h,
                "CRPS_haversine_std": np.nan,
                "CRPS_vmax_mean": crps_v,
                "CRPS_vmax_std": np.nan,
                "CRPS_pmin_mean": crps_p,
                "CRPS_pmin_std": np.nan,
                # Deterministic from ensemble mean (optional)
                "DPE_GCD_mean": dpe_mean,
                "AE_wind_mean": ae_wind,
                "SE_wind_mean": se_wind,
                "AE_pressure_mean": ae_pres,
                "SE_pressure_mean": se_pres,
                # Deterministic CARTE from ensemble mean
                "Along_TE_mean": along_te,
                "Cross_TE_mean": cross_te,
                "DPE_cart_mean": dpe_cart,
                # Placeholders for std (not computed in this fast path)
                "Along_TE_std": np.nan,
                "Cross_TE_std": np.nan,
                "DPE_cart_std": np.nan,
            }
        )

    if not out_rows:
        raise RuntimeError("No climatology baseline rows were produced.")

    out = pd.DataFrame(out_rows)
    out = out.sort_values(["SID", "Initial Time", "Valid Time", "Hour"]).reset_index(
        drop=True
    )

    # 5) Save results CSV compatible with plotting
    out_path = os.path.join(EVAL_DIR, "2023_climatology_results.csv")
    out.to_csv(out_path, index=False)
    print(
        f"✅ Saved climatology baseline results: {out_path}  (rows={len(out)}) [t0=00/12Z, vt=6-hourly]"
    )


if __name__ == "__main__":
    build_climatology_results()

# %%
