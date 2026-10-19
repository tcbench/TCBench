# %% Imports
import pandas as pd
import numpy as np
from itertools import cycle

__all__ = [
    # --- Helper vars ---
    "GOOGLE_KEYS",
    "SHOW_COUNTS",
    "PANEL_GRID",
    "ALL_BASES",
    "NON_R2_BASES",
    "NICE_NAME",
    "UNITS",
    "SHORT_TITLE",
    "YLIMS",
    "CRPS_BASES",
    "CRPS_TO_MAE",
    "CARTE_BASES",
    "strip_tokens",
    "variant_rank",
    "MAX_DISTINCT",
    "PERSIST_COLOR",
    "CLIM_COLOR",
    "CLIM_KEYS",
    "RENAME_SHORT",
    "EXPLICIT_LEADS",
    "PP_FILL_LEADS",
    "FULL_FILL_LEADS",
    "INIT_HOURS",
    "COVERAGE_DENOM",
    "VERBOSE_COVERAGE",
    "color_list",
    "colors",
    "_COLOR_CYCLE",
    "_COLOR_MAP_GLOBAL",
    "TEST_YEAR",
    #  --- Helper Functions ---
    "_log",
    "q25",
    "q75",
    "base_key",
    "rank_of",
    "_is_google",
    "_is_genc",
    "_is_pangu_post",
    "_is_climatology",
    "clean_label",
    "_pick_se_col_for_r2",
    "pretty_curve_label",
    "_legend_sort_key",
    "color_for",
    "_build_ib_denom_from_df",
    "_build_ib_denominator_counts_unified",
    "_aligned_ib_denominator_for_model",
    "_ibtracs_active_tracks_by_lead",
    "_coverage_series_vs_ib_from_raw",
]

# %% Reference Variables
GOOGLE_KEYS = ("weatherlab_FNV3", "FNV3", "WeatherLab")

SHOW_COUNTS = False  # plot sample size per lead on a right-hand y-axis

# === Panel grid (3 rows x 4 cols) ===
# Row 1: Track metrics (4 panels)
# Row 2: Pressure: AE, R², CRPS, (empty)
# Row 3: Wind:     AE, R², CRPS, (empty)
PANEL_GRID = [
    ["DPE_GCD", "CRPS_haversine", "Along_TE"],
    ["AE_pressure", "R2_pressure", "CRPS_pmin"],
    ["AE_wind", "R2_wind", "CRPS_vmax"],
]

# Helpers
ALL_BASES = [b for row in PANEL_GRID for b in row if b is not None]
NON_R2_BASES = [b for b in ALL_BASES if not str(b).startswith("R2_")]

NICE_NAME = {
    # TRACK position metrics
    "DPE_GCD": "Track — direct position error (DPE)",
    "CRPS_haversine": "Track — CRPS (track displacement)",
    # TRACK geometry (CARTE)
    "Along_TE": "Track — along-track error (CARTE)",
    "Cross_TE": "Track — cross-track error (CARTE)",
    "DPE_cart": "Track — direct position error (CARTE)",
    # INTENSITY metrics
    "AE_wind": "Intensity — abs. error (max wind)",
    "SE_wind": "Intensity — squared error (max wind)",
    "CRPS_vmax": "Intensity — CRPS (max wind)",
    "AE_pressure": "Intensity — abs. error (min pressure)",
    "SE_pressure": "Intensity — squared error (min pressure)",
    "CRPS_pmin": "Intensity — CRPS (min pressure)",
}

UNITS = {
    "DPE_GCD": "km",
    "CRPS_haversine": "km",
    "AE_wind": "kt",
    "SE_wind": "kt²",
    "CRPS_vmax": "kt",
    "AE_pressure": "hPa",
    "SE_pressure": "hPa²",
    "CRPS_pmin": "hPa",
    "Along_TE": "km",
    "Cross_TE": "km",
    "DPE_cart": "km",
}

# --- Add R² (unitless) entries
NICE_NAME.update(
    {
        "R2_pressure": "Intensity — Coefficient of Determination (Pressure)",
        "R2_wind": "Intensity — Coefficient of Determination (Wind)",
    }
)
UNITS.update(
    {
        "R2_pressure": "",  # unitless
        "R2_wind": "",  # unitless
    }
)

# --- Short title map for panels
SHORT_TITLE = {
    "DPE_GCD": "DPE — track",
    "CRPS_haversine": "CRPS — track",
    "Along_TE": "Along‑track error",
    "Cross_TE": "Cross‑track error",
    "AE_pressure": "AE — pressure",
    "CRPS_pmin": "CRPS — pressure",
    "AE_wind": "AE — wind",
    "CRPS_vmax": "CRPS — wind",
    "R2_pressure": "R² — pressure",
    "R2_wind": "R² — wind",
}

YLIMS = {
    "DPE_GCD": (0, 2000),
    "Cross_TE": (0, 900),
    "DPE_cart": (0, 900),
    # "Cross_TE": (0, 3000000),  # km, cross-track error (CARTE)
    # "DPE_cart": (0, 3000000),  # km, CARTE DPE
}


# map CRPS bases to the deterministic metric to use for persistence MAE overlay
CRPS_BASES = {"CRPS_haversine", "CRPS_vmax", "CRPS_pmin"}
CRPS_TO_MAE = {
    "CRPS_haversine": "DPE_GCD",  # distance → use DPE (km)
    "CRPS_vmax": "AE_wind",  # wind → use AE_wind (kt)
    "CRPS_pmin": "AE_pressure",  # pressure → use AE_pressure (hPa)
}
# Cartesian (CARTE) bases where we do NOT show baselines on the raw plots
CARTE_BASES = {"Along_TE", "Cross_TE", "DPE_cart"}

# --- consolidate duplicates that map to the same logical forecast (e.g., GEFS vs GEFS_clean)
# define a base key (label) by stripping common suffixes
strip_tokens = ["_clean", "_corrected", "(downloaded)", "_results"]

# variant ranking: prefer clean > corrected > base
variant_rank = {"_clean": 3, "_corrected": 2}


# Define color map keyed by forecast label using default color cycle
# Based on Martin Krzywinski's 15-color palette for "Designing for Color Blindness"
# https://mk.bcgsc.ca/colorblind/palettes.mhtml#12-color-palette-for-colorbliness
# COLORBLIND_COLORS = [
#     "#68023F",  # Tyrian Purple
#     "#008169", # Deep Sea
#     "#EF0096",  # persian rose
#     "#00DCB5", # aquamarine
#     "#FFCFE2", # azalea
#     "#003C86", # congress blue
#     "#9400E6", # veronica
#     "#009FFA", # bleu de france
#     "#FF71FD", # shocking pink
#     "#7DFFFA", # electric blue
#     "#6A0213", # rosewood
#     "#008607", # india green
#     "#F60239", # tractor red
#     "#00E307", # radioactive green
#     "#FFDC3D", # gargoyle gas
# ]
# color_list = COLORBLIND_COLORS
# colors = cycle(COLORBLIND_COLORS)

MAX_DISTINCT = [
    "#%02x%02x%02x" % (202, 82, 52),
    "#%02x%02x%02x" % (30, 185, 164),
    "#%02x%02x%02x" % (138, 50, 49),
    "#%02x%02x%02x" % (77, 60, 143),
    "#%02x%02x%02x" % (60, 133, 78),
    "#%02x%02x%02x" % (0, 115, 137),
    "#%02x%02x%02x" % (171, 81, 186),
    "#%02x%02x%02x" % (212, 142, 72),
    "#%02x%02x%02x" % (131, 125, 46),
    "#%02x%02x%02x" % (222, 133, 125),
    "#%02x%02x%02x" % (205, 73, 118),
    # "#%02x%02x%02x" % (222, 129, 194), Used for MT-LB (climatology inspired baseline)
    "#%02x%02x%02x" % (166, 147, 218),
    "#%02x%02x%02x" % (127, 50, 106),
    "#%02x%02x%02x" % (85, 109, 188),
    "#%02x%02x%02x" % (85, 168, 213),
    "#%02x%02x%02x" % (114, 185, 85),
    "#%02x%02x%02x" % (58, 86, 18),
    "#%02x%02x%02x" % (191, 174, 71),
    "#%02x%02x%02x" % (126, 76, 9),
]

#
# --- plotting
# Baseline styles
PERSIST_COLOR = "black"
CLIM_COLOR = "#%02x%02x%02x" % (
    222,
    129,
    194,
)  # "cyan"  # distinct from persistence; dashed thick too


CLIM_KEYS = ("climatology",)

# Pretty label for legends/plots; also rename Google's model to short form and italicize
RENAME_SHORT = {
    "weatherlab_FNV3": "FNV3",
    "2023_weatherlab_FNV3": "FNV3",
}


EXPLICIT_LEADS = (
    6,
    12,
    18,
    24,
    30,
    36,
    42,
    48,
    54,
    60,
    66,
    72,
    78,
    84,
    90,
    96,
    102,
    108,
    114,
    120,
)

#
# Leads where post-processing models have values; other models use the full 6h grid
PP_FILL_LEADS = (6, 12, 18, 24, 48, 72, 96, 120)
FULL_FILL_LEADS = EXPLICIT_LEADS

# Only count initializations at synoptic hours 00Z and 12Z
INIT_HOURS = (0, 12)

# Coverage denominator mode: "ibtracs" = fixed IBTrACS verification-pair counts (same for all models)
# or "aligned" = IBTrACS pairs only for (SID, t0) that appear in a given model file

COVERAGE_DENOM = "ibtracs"  # choose: "ibtracs" or "aligned"
# Print per-model coverage diagnostics while plotting coverage
VERBOSE_COVERAGE = False

color_list = MAX_DISTINCT
colors = cycle(MAX_DISTINCT)

# TAB_COLORS = [mpl.colors.to_hex(c) for c in mpl.colormaps["tab10"].colors]
# color_list = TAB_COLORS
# colors = cycle(TAB_COLORS)


# --- Global consistent color mapping (same color per model everywhere)
_COLOR_CYCLE = iter(color_list)
_COLOR_MAP_GLOBAL: dict[str, str] = {}

TEST_YEAR = 2023


# %% Helpfer Functions
def _log(*args, **kwargs):
    return


# helpers for naming quantile aggregations
def q25(s):
    return s.quantile(0.25)


def q75(s):
    return s.quantile(0.75)


def _is_google(lbl: str) -> bool:
    low = lbl.lower()
    return any(k.lower() in low for k in GOOGLE_KEYS)


def base_key(fname: str) -> str:
    key = fname.replace("_results.csv", "")
    for t in strip_tokens:
        key = key.replace(t, "")
    return key


def rank_of(fname: str) -> int:
    for tok, r in variant_rank.items():
        if tok in fname:
            return r
    return 1  # base


# Treat GENC specially in styling/labels (italic + dotted), but keep its own color
def _is_genc(lbl: str) -> bool:
    return "genc" in str(lbl).lower()


# Detect the ANN post-processing variant to style it (dotted) consistently
def _is_pangu_post(lbl: str) -> bool:
    low = str(lbl).lower()
    base = (
        clean_label(str(lbl)).lower()
        if "clean_label" in globals()
        else str(lbl).lower()
    )
    return (
        ("postprocessing_panguweather_ann" in low)
        or ("postprocessing_panguweather_mlr" in low)
        or ("postprocessing_panguweather_unet" in low)
        or (
            base
            in {"pangu_post", "pangu_post_ann", "pangu_post_mlr", "pangu_post_unet"}
        )
    )


def _is_climatology(lbl: str) -> bool:
    return any(k in lbl.lower() for k in CLIM_KEYS)


def clean_label(lbl):
    for s in ["_clean", "_corrected", "(downloaded)", "_results", "_fixed"]:
        lbl = lbl.replace(s, "")
    # strip leading year prefix like 2023_
    if lbl.startswith("2023_"):
        lbl = lbl[5:]
    return lbl


# --- Helper: Pick best available SE column for R² computation
def _pick_se_col_for_r2(df: pd.DataFrame, variable: str) -> str | None:
    """
    Choose the best available SE column for R² computation for either 'wind' or 'pressure'.
    Preference order:
      1) SE_*_mean (probabilistic results)
      2) SE_*       (deterministic results)
      3) AE_*_mean  (fallback → square to get SE)
      4) AE_*       (fallback → square to get SE)
    Returns the column name found or None if nothing usable exists.
    """
    assert variable in {"wind", "pressure"}
    base = "pressure" if variable == "pressure" else "wind"
    se_mean = f"SE_{base}_mean"
    se_plain = f"SE_{base}"
    ae_mean = f"AE_{base}_mean"
    ae_plain = f"AE_{base}"
    cols = df.columns
    if se_mean in cols:
        return se_mean
    if se_plain in cols:
        return se_plain
    if ae_mean in cols:
        return ae_mean
    if ae_plain in cols:
        return ae_plain
    return None


def pretty_curve_label(lbl: str) -> str:
    lowlbl = lbl.lower()
    base = clean_label(lbl)
    # Persistence remap: always label as "Persistence [BASE]"
    if "persistence" in lowlbl:
        return "Persistence [BASE]"
    base = RENAME_SHORT.get(base, base)
    if "postprocessing_panguweather_ann" in lowlbl or base.upper() == "PANGU_POST":
        return "PANGU_POST_ANN"
    if "postprocessing_panguweather_mlr" in lowlbl:
        return "PANGU_POST_MLR"
    if "postprocessing_panguweather_unet" in lowlbl:
        return "PANGU_POST_UNET"

    if _is_google(lbl) or base.lower() in ("fnv3",):
        return "$\\it{FNV3}$"
    if _is_genc(lbl) or base.lower() == "genc":
        return "$\\it{GENC}$"
    if _is_climatology(lbl) or base.lower() == "climatology":
        return "MT-LB"
    return base


# Helper to sort legend labels so persistence is always first, then PANGU, then PANGU_POST, then others, then Google/FNV3 last
def _legend_sort_key(lbl: str):
    low = lbl.lower()
    # Persistence first
    if "persistence [base]" in low or "persistence" in low:
        return (-1, lbl)
    # PANGU core next
    if low == "pangu":
        return (0, lbl)
    # PANGU_POST family next
    if low in ("pangu_post", "pangu_post_ann", "pangu_post_mlr", "pangu_post_unet"):
        return (1, lbl)
    # Push Google's WeatherLab FNV3 last
    if _is_google(lbl) or "fnv3" in low:
        return (999, lbl)
    return (2, lbl)


def color_for(pretty_label: str) -> str:
    """Return a stable color for a given pretty label across figures."""
    key = pretty_label.strip()
    # Persistence: always black
    if "persistence" in key.lower():
        return "black"
    # Force proprietary Google WeatherLab FNV3 to gray across ALL figures
    if _is_google(key) or ("fnv3" in key.lower()) or ("weatherlab" in key.lower()):
        return "#7a7a7a"
    global _COLOR_CYCLE
    if key not in _COLOR_MAP_GLOBAL:
        try:
            _COLOR_MAP_GLOBAL[key] = next(_COLOR_CYCLE)
        except StopIteration:
            _COLOR_CYCLE = iter(color_list)
            _COLOR_MAP_GLOBAL[key] = next(_COLOR_CYCLE)
    return _COLOR_MAP_GLOBAL[key]


def _build_ib_denom_from_df(ib_df: pd.DataFrame, leads=EXPLICIT_LEADS) -> pd.Series:
    """Counts IBTrACS (SID, t0, vt) keys by lead from an in-memory IBTrACS dataframe.
    Robust to tz-aware timestamps and sub-hour noise by normalizing to *hour* grid.
    """
    if ib_df is None or ib_df.empty:
        return pd.Series(dtype=float)
    df = ib_df.copy()
    if "ISO_TIME" not in df.columns or "SID" not in df.columns:
        return pd.Series(dtype=float)

    # Normalize timestamps: coerce, drop tz, floor to hour
    dt = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    if hasattr(dt.dt, "tz_localize"):
        try:
            dt = dt.dt.tz_localize(None)
        except Exception:
            pass
    dt = dt.dt.floor("h")
    df = df.assign(ISO_TIME=dt).dropna(subset=["ISO_TIME"]).copy()

    counts: dict[float, int] = {}
    lead_timedeltas = [pd.Timedelta(hours=int(l)) for l in leads]
    lead_hours = [float(int(l)) for l in leads]

    for sid, grp in df.groupby("SID", sort=False):
        h = grp["ISO_TIME"].drop_duplicates().sort_values()

        # Keep only 6-hourly synoptic times (00, 06, 12, 18 UTC)
        h_6h = h[h.dt.hour.isin([0, 6, 12, 18])]
        hset = set(h_6h)
        if not hset:
            continue

        # Restrict initializations to 00Z/12Z if INIT_HOURS is defined
        if "INIT_HOURS" in globals() and INIT_HOURS:
            t0_candidates = [t for t in h_6h if t.hour in INIT_HOURS]
        else:
            t0_candidates = list(h_6h)
        if not t0_candidates:
            continue

        for t0 in t0_candidates:
            for td, lh in zip(lead_timedeltas, lead_hours):
                if (t0 + td) in hset:
                    counts[lh] = counts.get(lh, 0) + 1

    s = pd.Series(counts, dtype=float).sort_index()
    return s


def _build_ib_denominator_counts_unified(
    leads=EXPLICIT_LEADS, ib_df: pd.DataFrame | None = None
) -> pd.Series:
    """STRICT: Denominator for coverage = # of IBTrACS (SID, t0, vt) per lead.
    Uses ONLY the provided `ib_df` (must be prefiltered to 2023). Raises if missing/empty.
    """
    if ib_df is None or ib_df.empty:
        raise RuntimeError(
            "IBTrACS dataframe is required and cannot be empty for coverage computation."
        )
    s = _build_ib_denom_from_df(ib_df, leads)
    if s.empty:
        # Debug info to aid diagnosis if it ever happens again
        sample = ib_df.head(3)
        raise RuntimeError(
            "IBTrACS denominator could not be constructed from the provided dataframe. "
            f"Rows={len(ib_df)}, unique SIDs={ib_df['SID'].nunique()} (showing head):\n{sample.to_string(index=False)[:500]}"
        )
    return s


# --- IBTrACS denominator aligned to a model's initializations
def _aligned_ib_denominator_for_model(
    model_df: pd.DataFrame, ib_df: pd.DataFrame, leads=EXPLICIT_LEADS
) -> pd.Series:
    """
    Denominator aligned to a given model:
    counts IBTrACS (SID, t0, t0+L) pairs only for (SID, t0) initial times that
    appear in the model file. This avoids penalizing models for missing inits.
    """
    if model_df is None or model_df.empty or ib_df is None or ib_df.empty:
        return pd.Series(dtype=float)

    # --- Normalize model times ---
    m = model_df.copy()
    for c in ("Initial Time", "Valid Time"):
        if c in m.columns:
            m[c] = pd.to_datetime(m[c], errors="coerce")
    m = m.dropna(subset=["SID", "Initial Time"]).copy()

    m["Initial Time"] = (
        m["Initial Time"]
        .dt.tz_localize(None, nonexistent="shift_forward", ambiguous="NaT")
        .dt.floor("h")
    )

    # Keep only 6-hourly synoptic times
    m = m[m["Initial Time"].dt.hour.isin([0, 6, 12, 18])]

    # Restrict to configured init hours (e.g. 00Z/12Z)
    if "INIT_HOURS" in globals() and INIT_HOURS:
        m = m[m["Initial Time"].dt.hour.isin(INIT_HOURS)]

    model_t0 = set(zip(m["SID"], m["Initial Time"]))

    # --- Normalize IBTrACS times ---
    df = ib_df.copy()
    dt = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    if hasattr(dt.dt, "tz_localize"):
        try:
            dt = dt.dt.tz_localize(None)
        except Exception:
            pass
    dt = dt.dt.floor("h")
    df = df.assign(ISO_TIME=dt).dropna(subset=["ISO_TIME"]).copy()

    counts: dict[float, int] = {}
    lead_timedeltas = [pd.Timedelta(hours=int(l)) for l in leads]
    lead_hours = [float(int(l)) for l in leads]

    for sid, grp in df.groupby("SID", sort=False):
        h = grp["ISO_TIME"].drop_duplicates().sort_values()

        # Keep only 6-hourly synoptic times
        h_6h = h[h.dt.hour.isin([0, 6, 12, 18])]
        if h_6h.empty:
            continue
        hset = set(h_6h)

        # Only consider t0 that exist in the model for this SID
        t0_for_sid = [t0 for (s, t0) in model_t0 if s == sid]

        # INIT_HOURS filter already applied above when building model_t0,
        # but apply again if needed for safety
        if "INIT_HOURS" in globals() and INIT_HOURS:
            t0_for_sid = [t0 for t0 in t0_for_sid if t0.hour in INIT_HOURS]

        if not t0_for_sid:
            continue

        for t0 in t0_for_sid:
            for td, lh in zip(lead_timedeltas, lead_hours):
                if (t0 + td) in hset:
                    counts[lh] = counts.get(lh, 0) + 1

    return pd.Series(counts, dtype=float).sort_index()


# --- Active tracks per lead (unique SIDs with a valid pair for each lead)
def _ibtracs_active_tracks_by_lead(
    ib_df: pd.DataFrame, leads=EXPLICIT_LEADS
) -> pd.Series:
    """Return # of *unique SIDs* that are still active at each lead.
    A SID is 'active' at lead L if it has at least one pair of hourly records
    (t0, t0+L) within the same storm in IBTrACS 2023.
    """
    if ib_df is None or ib_df.empty:
        return pd.Series(dtype=float)
    if "ISO_TIME" not in ib_df.columns or "SID" not in ib_df.columns:
        return pd.Series(dtype=float)

    df = ib_df.copy()
    dt = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    if hasattr(dt.dt, "tz_localize"):
        try:
            dt = dt.dt.tz_localize(None)
        except Exception:
            pass
    dt = dt.dt.floor("h")
    df = df.assign(ISO_TIME=dt).dropna(subset=["ISO_TIME"]).copy()

    active_counts: dict[float, int] = {}
    lead_list = [int(l) for l in leads]

    for sid, grp in df.groupby("SID", sort=False):
        h = grp["ISO_TIME"].drop_duplicates().sort_values()
        # all 6-hourly timestamps available for this SID
        h_int_all = (h.astype("int64", copy=False) // 3_600_000_000_000).to_numpy()
        h_int_all = h_int_all[h_int_all % 6 == 0]
        if len(h_int_all) == 0:
            continue
        hset = set(h_int_all.tolist())
        # t0 candidates restricted to 00Z/12Z, vt may be any 6-hour step
        if "INIT_HOURS" in globals() and INIT_HOURS:
            t0_candidates = [t for t in h_int_all if (t % 24) in INIT_HOURS]
        else:
            t0_candidates = list(h_int_all)
        if not t0_candidates:
            continue
        for lh in lead_list:
            # 'alive' if there exists t0 such that (t0 + lh) is observed within the same SID
            if any((t0 + lh) in hset for t0 in t0_candidates):
                active_counts[float(lh)] = active_counts.get(float(lh), 0) + 1

    return pd.Series(active_counts, dtype=float).sort_index()


def _coverage_series_vs_ib_from_raw(
    dfm: pd.DataFrame, ib_denom: pd.Series
) -> pd.Series:
    """
    Compute coverage % per lead for a *raw* model results DataFrame
    with columns [SID, Initial Time, Valid Time, ...]. Any presence of a row
    for a given (SID, t0, vt) counts as coverage (regardless of which fields/metrics
    are present). Divides by the IBTrACS denominator at the same lead.
    """
    if dfm is None or dfm.empty:
        return pd.Series(dtype=float)
    dfm = dfm.copy()
    for c in ("Initial Time", "Valid Time"):
        if c in dfm.columns:
            dfm[c] = pd.to_datetime(dfm[c], errors="coerce")
    dfm = dfm.dropna(subset=["SID", "Initial Time", "Valid Time"]).copy()
    if "INIT_HOURS" in globals() and INIT_HOURS:
        dfm = dfm[dfm["Initial Time"].dt.hour.isin(INIT_HOURS)].copy()
    # Enforce both Initial/Valid times lie on the 6h grid
    dfm["init_hr"] = (
        dfm["Initial Time"].astype("int64", copy=False) // 3_600_000_000_000
    )
    dfm["valid_hr"] = dfm["Valid Time"].astype("int64", copy=False) // 3_600_000_000_000
    dfm = dfm[(dfm["init_hr"] % 6 == 0) & (dfm["valid_hr"] % 6 == 0)].copy()
    # integer lead in hours (rounded) to robustly hit the 6h grid
    dfm["lead_hours"] = (
        ((dfm["Valid Time"] - dfm["Initial Time"]).dt.total_seconds() / 3600.0)
        .round()
        .astype(int)
    )

    cov_vals = {}
    for lh, grp in dfm.groupby("lead_hours"):
        # Use integer hours for keys to avoid subtle tz/floor issues
        keys = set(zip(grp["SID"], grp["init_hr"], grp["valid_hr"]))
        den = int(ib_denom.get(float(lh), 0)) if not ib_denom.empty else 0
        cov_vals[float(lh)] = (100.0 * len(keys) / den) if den > 0 else np.nan
    return pd.Series(cov_vals, dtype=float).sort_index()
