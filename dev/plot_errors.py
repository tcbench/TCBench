# %% Plot: per-lead-time error curves with error bars (paired subplots)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
import matplotlib as mpl
import matplotlib.ticker as mticker
from utils import toolbox
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

# Toolbox helper aliases for explicit usage
from utils.toolbox import load_eval_csv as _load_eval_csv
from utils.toolbox import compute_r2_by_lead_from_results as _r2_from_results
from utils.toolbox import pick_metric_col

import sys
import argparse

parser = argparse.ArgumentParser(
    description="Plot TCBench error comparisons and coverage."
)
parser.add_argument(
    "--ibtracs_path",
    type=str,
    help="Path to the IBTrACS track folder containing the file (CSV) for 2023, used for coverage computation.",
    default=os.path.join(os.curdir, "data", "ibtracs"),
)
parser.add_argument(
    "--eval_dir",
    type=str,
    help="Path to the directory containing the evaluation results CSV files.",
    default=os.path.join(os.curdir, "outputs"),
)
args = parser.parse_args()


# --- Silence all stdout/stderr prints and warnings in this module
import warnings as _warnings


def _log(*args, **kwargs):
    return


print = _log  # make all prints no-ops inside this module
_warnings.filterwarnings("ignore")


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "mathtext.fontset": "cm",  # Computer Modern math to match LaTeX math
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    }
)


# helpers for naming quantile aggregations
def q25(s):
    return s.quantile(0.25)


def q75(s):
    return s.quantile(0.75)


EVAL_DIR = args.eval_dir

PERSIST_PATH = os.path.join(EVAL_DIR, "persistence_results.csv")
_persist_raw = None
if os.path.exists(PERSIST_PATH):
    try:
        _persist_raw = pd.read_csv(PERSIST_PATH, low_memory=False)
        for c in ("Initial Time", "Valid Time"):
            if c in _persist_raw.columns:
                _persist_raw[c] = pd.to_datetime(_persist_raw[c], errors="coerce")
        if {"Initial Time", "Valid Time"}.issubset(_persist_raw.columns):
            _persist_raw["lead_hours"] = (
                _persist_raw["Valid Time"] - _persist_raw["Initial Time"]
            ).dt.total_seconds() / 3600.0
    except Exception as _e:
        print(f"⚠️ Could not load persistence_results.csv for MAE overlay: {_e}")

# ---- IBTrACS 2023 (ground-truth key set for coverage) — STRICT (no fallbacks) ----
if toolbox is None:
    raise RuntimeError(
        "IBTrACS toolbox is required but not available: cannot compute coverage without IBTrACS."
    )
try:
    ibtracs = toolbox.read_hist_track_file(tracks_path=args.ibtracs_path)
    ibtracs = ibtracs[ibtracs["ISO_TIME"].dt.year == 2023].copy()
    if ibtracs.empty:
        raise RuntimeError("Loaded IBTrACS (2023) is empty.")
except Exception as _e:
    raise RuntimeError(f"Failed to load IBTrACS for coverage denominator: {_e}")

# --- collect result files but blacklist unwanted ones
result_files = sorted(
    [
        f
        for f in os.listdir(EVAL_DIR)
        if ("RI" not in f) and (f.endswith("_results.csv") or ("baseline" in f))
    ]
)

# keep only the ANN post-processing variant among postprocessing_* files
_PP_KEEP = "postprocessing_panguweather_ann"
result_files = [
    f
    for f in result_files
    if ("postprocessing" not in f.lower()) or (_PP_KEEP in f.lower())
]

# Ensure climatology baseline is included if present
CLIM_FILE = os.path.join(EVAL_DIR, "2023_climatology_results.csv")
if os.path.exists(CLIM_FILE):
    clim_name = os.path.basename(CLIM_FILE)
    if clim_name not in result_files:
        result_files.append(clim_name)

blacklist_substr = [
    "Gencast",  # keep filtering Gencast
    "TIGGE_IFS",  # drop any IFS variant (results, corrected, clean)
]
result_files = [f for f in result_files if not any(b in f for b in blacklist_substr)]
if not result_files:
    raise FileNotFoundError("No *_results.csv files found in EVAL_DIR after filtering.")

# --- consolidate duplicates that map to the same logical forecast (e.g., GEFS vs GEFS_clean)
# define a base key (label) by stripping common suffixes
strip_tokens = ["_clean", "_corrected", "(downloaded)", "_results"]


def base_key(fname: str) -> str:
    key = fname.replace("_results.csv", "")
    for t in strip_tokens:
        key = key.replace(t, "")
    return key


# variant ranking: prefer clean > corrected > base
variant_rank = {"_clean": 3, "_corrected": 2}


def rank_of(fname: str) -> int:
    for tok, r in variant_rank.items():
        if tok in fname:
            return r
    return 1  # base


best = {}
for f in result_files:
    k = base_key(f)
    r = rank_of(f)
    if (k not in best) or (r > best[k][0]):
        best[k] = (r, f)

# keep only the chosen representative files
result_files = [v[1] for v in best.values()]
result_files.sort()

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


# collect stats
stats = {}
for fn in result_files:
    label = fn.replace("_results.csv", "")
    df = _load_eval_csv(os.path.join(EVAL_DIR, fn))
    for base in NON_R2_BASES:
        col = pick_metric_col(df, base)
        if not col:
            continue
        g = (
            df[["lead_hours", col]]
            .dropna(subset=["lead_hours"])
            .groupby("lead_hours", dropna=False)[col]
        )
        agg = g.agg(["mean", "count", q25, q75]).sort_index()
        # unify column names
        agg = agg.rename(columns={"q25": "p25", "q75": "p75"})
        if len(agg):
            stats.setdefault(base, {})[label] = agg

# --- diagnostics: tell the user if a metric base has no data in any file
for base in NON_R2_BASES:
    if not stats.get(base):
        # check which files are missing the expected column
        missing_in = []
        present_in = []
        for fn in result_files:
            df_cols = pd.read_csv(
                os.path.join(EVAL_DIR, fn), nrows=1, low_memory=False
            ).columns
            if (base in df_cols) or (f"{base}_mean" in df_cols):
                present_in.append(fn)
            else:
                missing_in.append(fn)
        print(f"⚠️  No data found to plot for '{base}'.")
        if present_in:
            print(f"   Found in: {present_in}")
        if missing_in:
            print(
                f"   Missing in: {missing_in[:6]}{' ...' if len(missing_in)>6 else ''}"
            )

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
color_list = MAX_DISTINCT
colors = cycle(MAX_DISTINCT)

# TAB_COLORS = [mpl.colors.to_hex(c) for c in mpl.colormaps["tab10"].colors]
# color_list = TAB_COLORS
# colors = cycle(TAB_COLORS)

forecast_labels = sorted({lbl for base in stats for lbl in stats[base]})
color_map = {lbl: col for lbl, col in zip(forecast_labels, colors)}

GOOGLE_KEYS = ("weatherlab_FNV3", "FNV3", "WeatherLab")


def _is_google(lbl: str) -> bool:
    low = lbl.lower()
    return any(k.lower() in low for k in GOOGLE_KEYS)


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


CLIM_KEYS = ("climatology",)


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


# Pretty label for legends/plots; also rename Google's model to short form and italicize
RENAME_SHORT = {
    "weatherlab_FNV3": "FNV3",
    "2023_weatherlab_FNV3": "FNV3",
}


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


# --- Global consistent color mapping (same color per model everywhere)
_COLOR_CYCLE = iter(color_list)
_COLOR_MAP_GLOBAL: dict[str, str] = {}


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


#
# --- plotting
# Baseline styles
PERSIST_COLOR = "black"
CLIM_COLOR = "#%02x%02x%02x" % (
    222,
    129,
    194,
)  # "cyan"  # distinct from persistence; dashed thick too

nrows = len(PANEL_GRID)
ncols = len(PANEL_GRID[0])
fig, axes = plt.subplots(
    nrows, ncols, figsize=(16, 4.1 * nrows), sharex=False, squeeze=False
)

for r, row in enumerate(PANEL_GRID):
    for c, base in enumerate(row):
        ax = axes[r, c]
        if base is None:
            ax.set_visible(False)
            continue
        y_max_plotted = 0.0

        # Special handling for R² panels: build per-forecast series on the fly
        if base in ("R2_pressure", "R2_wind"):
            variable = "pressure" if base == "R2_pressure" else "wind"
            model_handles, model_labels = [], []
            base_handles, base_labels = [], []
            for fn in result_files:
                label = fn.replace("_results.csv", "")
                df_src = _load_eval_csv(os.path.join(EVAL_DIR, fn))

                # Skip RI & TIGGE_IFS only; include persistence and climatology as baselines
                low = fn.lower()
                if any(tok in low for tok in ("ri", "tigge_ifs")):
                    continue

                # Ensure we have the standard SE_* column expected by the R² helper,
                # accepting probabilistic means (SE_*_mean) or falling back to AE_*(_mean)**2.
                df_use = df_src.copy()
                se_cand = _pick_se_col_for_r2(df_use, variable)
                if se_cand is None:
                    # nothing usable for this variable → skip this model
                    continue
                if variable == "pressure":
                    target = "SE_pressure"
                else:
                    target = "SE_wind"
                if se_cand != target:
                    if se_cand.endswith("_mean") and se_cand.startswith("SE_"):
                        # probabilistic: copy mean(SE) into the expected name
                        df_use[target] = pd.to_numeric(df_use[se_cand], errors="coerce")
                    elif se_cand.startswith("AE_"):
                        # fallback: square AE (or AE_mean) to approximate SE
                        df_use[target] = (
                            pd.to_numeric(df_use[se_cand], errors="coerce") ** 2
                        )
                    else:
                        # unexpected, but try copying numerically
                        df_use[target] = pd.to_numeric(df_use[se_cand], errors="coerce")
                else:
                    # Ensure numeric dtype
                    df_use[target] = pd.to_numeric(df_use[target], errors="coerce")

                # Compute R² series using SE_* (derived from AE_* if needed) and GT from IBTrACS
                try:
                    r2 = _r2_from_results(
                        df_use, ibtracs, variable=variable, year=None, debug=False
                    )
                except ValueError as e:
                    print(f"[R2] Skipping {fn} ({variable}): {e}")
                    continue
                except Exception as e:
                    print(f"[R2] Skipping {fn} ({variable}) due to error: {e}")
                    continue
                if r2.empty:
                    print(f"[R2] No usable R² for {fn} ({variable}) — empty series.")
                    continue

                is_persist = "persistence" in label.lower()
                is_clim = _is_climatology(label)
                lab = pretty_curve_label(label)

                if is_persist:
                    style = dict(
                        linestyle="--",
                        color=PERSIST_COLOR,
                        linewidth=3.0,
                        markersize=3,
                        label="Persistence",
                    )
                elif is_clim:
                    style = dict(
                        linestyle="--",
                        color=CLIM_COLOR,
                        linewidth=3.0,
                        markersize=3,
                        label="MT-LB",
                    )
                else:
                    style = dict(
                        linestyle="-",
                        marker="o",
                        linewidth=1.5,
                        markersize=3,
                        color=color_for(lab),
                        label=lab,
                    )
                    if _is_google(label):
                        # FNV3: force gray color; now use dotted to match special treatment
                        style.update(
                            dict(
                                color="#7a7a7a",
                                linestyle=":",
                                marker="s",
                                linewidth=2.0,
                            )
                        )
                    elif _is_genc(label) or _is_pangu_post(label):
                        # GENC and PANGU_POST: dotted line, keep their colors
                        style.update(dict(linestyle=":", marker="s", linewidth=2.0))

                (h,) = ax.plot(r2.index.values.astype(float), r2.values, **style)

                if is_persist:
                    base_handles.append(h)
                    base_labels.append("Persistence")
                elif is_clim:
                    base_handles.append(h)
                    base_labels.append("MT-LB")
                else:
                    model_handles.append(h)
                    model_labels.append(lab)
                try:
                    y_max_plotted = max(y_max_plotted, float(np.nanmax(r2.values)))
                except Exception:
                    pass

            # Axes formatting
            short_title = SHORT_TITLE.get(base, base)
            ax.set_title(short_title)
            ax.set_xlabel("Lead time (hours)")
            ax.set_ylabel("")
            ax.set_xlim(6, 120)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(24))
            ax.xaxis.set_minor_locator(mticker.MultipleLocator(6))
            ax.grid(True, which="major", alpha=0.35)
            ax.grid(True, which="minor", alpha=0.12)
            # Zero-skill reference line and fixed, publication-friendly range
            ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
            ax.set_ylim(-0.7, 1.0)

            # Legends: models and baselines
            leg_models = None
            if model_handles:
                pairs = list(zip(model_handles, model_labels))
                pairs.sort(key=lambda hl: _legend_sort_key(hl[1]))
                h_sorted, l_sorted = zip(*pairs)
                leg_models = ax.legend(
                    h_sorted,
                    l_sorted,
                    fontsize=11,
                    loc="upper left",
                    frameon=False,
                    ncols=2,
                )
            if base_handles:
                leg_base = ax.legend(
                    base_handles,
                    base_labels,
                    fontsize=11,
                    loc="lower right",
                    frameon=False,
                    title="Baselines",
                )
                if leg_models is not None:
                    ax.add_artist(leg_models)
            elif leg_models is None:
                ax.legend(fontsize=11, loc="upper left", frameon=False)

            # Done with R² panel
            continue

        # --- Default panels (existing behaviour)
        per_forecast = stats.get(base, {})
        if not per_forecast:
            ax.set_visible(False)
            continue

        ax2 = ax.twinx() if SHOW_COUNTS else None

        # Prepare containers to build two legends: models vs baselines
        model_handles, model_labels = [], []
        base_handles, base_labels = [], []

        for forecast_label, agg in per_forecast.items():
            x = agg.index.values.astype(float)
            y = agg["mean"].values

            is_persist = "persistence" in forecast_label.lower()
            is_clim = _is_climatology(forecast_label)

            # Skip baselines for CARTE panels
            if base in CARTE_BASES and (is_persist or is_clim):
                continue

            # Label text
            if is_persist:
                label_txt = "Persistence"
            elif is_clim:
                label_txt = "MT-LB"
            else:
                label_txt = pretty_curve_label(forecast_label)

            # Skip persistence on CRPS panels (we overlay MAE instead), but DO show climatology
            if base in CRPS_BASES and is_persist:
                continue

            # Build style
            if is_persist:
                plot_kwargs = dict(
                    linestyle="--",
                    color=PERSIST_COLOR,
                    linewidth=3.0,
                    markersize=3,
                    zorder=10,
                    label=label_txt,
                )
            elif is_clim:
                plot_kwargs = dict(
                    linestyle="--",
                    color=CLIM_COLOR,
                    linewidth=3.0,
                    markersize=3,
                    zorder=9,
                    label=label_txt,
                )
            else:
                plot_kwargs = dict(
                    linestyle="-",
                    marker="o",
                    linewidth=1.5,
                    markersize=3,
                    label=label_txt,
                    color=color_for(label_txt),
                )
                # special styling for Google WeatherLab FNV3 (proprietary)
                if _is_google(forecast_label):
                    # FNV3: force gray + dotted
                    plot_kwargs.update(
                        dict(color="#7a7a7a", linestyle=":", marker="s", linewidth=2.0)
                    )
                elif _is_genc(forecast_label) or _is_pangu_post(forecast_label):
                    # GENC and PANGU_POST: dotted, keep color
                    plot_kwargs.update(dict(linestyle=":", marker="s", linewidth=2.0))

            (h,) = ax.plot(x, y, **plot_kwargs)
            try:
                y_max_plotted = max(y_max_plotted, float(np.nanmax(y)))
            except Exception:
                pass
            if is_persist or is_clim:
                base_handles.append(h)
                base_labels.append(label_txt)
            else:
                model_handles.append(h)
                model_labels.append(label_txt)

            # plot sample size on right axis (same color, dashed)
            if SHOW_COUNTS and ax2 is not None:
                if is_persist:
                    linecolor = PERSIST_COLOR
                elif is_clim:
                    linecolor = CLIM_COLOR
                else:
                    linecolor = color_map.get(forecast_label, None)
                ax2.plot(
                    x,
                    agg["count"].values,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.6,
                    color=linecolor,
                    label=f"{clean_label(forecast_label)} (n)",
                )

        # If this is a CRPS panel, overlay persistence MAE (from persistence_results.csv)
        if base in CRPS_BASES and _persist_raw is not None:
            mae_base = CRPS_TO_MAE.get(base)
            if mae_base is not None:
                # pick the right column from persistence file (prefer *_mean if present)
                mae_col = (
                    f"{mae_base}_mean"
                    if f"{mae_base}_mean" in _persist_raw.columns
                    else (mae_base if mae_base in _persist_raw.columns else None)
                )
                if mae_col is not None and "lead_hours" in _persist_raw.columns:
                    g_mae = (
                        _persist_raw[["lead_hours", mae_col]]
                        .dropna(subset=["lead_hours"])
                        .groupby("lead_hours")[mae_col]
                    )
                    agg_mae = g_mae.mean().sort_index()
                    if len(agg_mae):
                        (h_mae,) = ax.plot(
                            agg_mae.index.values.astype(float),
                            agg_mae.values,
                            linestyle="--",
                            color="black",
                            linewidth=3,
                            label="Persistence (MAE)",
                            zorder=9,
                        )
                        try:
                            y_max_plotted = max(
                                y_max_plotted, float(np.nanmax(agg_mae.values))
                            )
                        except Exception:
                            pass
                        # ensure it appears under the Baselines legend group
                        base_handles.append(h_mae)
                        base_labels.append("Persistence (MAE)")

        # Climatology baseline y_max update
        # Find the h_clim plot and update y_max_plotted if present
        if os.path.exists(CLIM_FILE):
            try:
                _clim_df = pd.read_csv(CLIM_FILE, low_memory=False)
                for cdt in ("Initial Time", "Valid Time"):
                    if cdt in _clim_df.columns:
                        _clim_df[cdt] = pd.to_datetime(_clim_df[cdt], errors="coerce")
                _clim_df["lead_hours"] = (
                    _clim_df["Valid Time"] - _clim_df["Initial Time"]
                ).dt.total_seconds() / 3600.0
                clim_col = pick_metric_col(_clim_df, base)
                if clim_col is not None:
                    c_agg = (
                        _clim_df[["lead_hours", clim_col]]
                        .dropna()
                        .groupby("lead_hours")[clim_col]
                        .mean()
                        .sort_index()
                    )
                    if len(c_agg):
                        try:
                            y_max_plotted = max(
                                y_max_plotted, float(np.nanmax(c_agg.values))
                            )
                        except Exception:
                            pass
            except Exception:
                pass

        short_title = SHORT_TITLE.get(base, base)
        ax.set_title(short_title)
        ax.set_xlabel("Lead time (hours)")
        ax.set_ylabel("")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(6, 120)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(24))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(6))
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.12)
        if base in YLIMS:
            ax.set_ylim(*YLIMS[base])
        else:
            # Dynamic headroom if no explicit cap provided (e.g., CRPS_haversine)
            if y_max_plotted > 0:
                ax.set_ylim(0, y_max_plotted * 1.10)

        # Right axis formatting for counts
        if SHOW_COUNTS and ax2 is not None:
            ax2.set_ylabel("Sample size (cases)")
            ax2.grid(False)

        # Legends: separate models and baselines (with a category title)
        if SHOW_COUNTS and ax2 is not None:
            # If counts are shown, merge their handles into the model legend
            h2, l2 = ax2.get_legend_handles_labels()
            model_handles_all = model_handles + h2
            model_labels_all = model_labels + l2
        else:
            model_handles_all = model_handles
            model_labels_all = model_labels

        # Primary legend: models
        if model_handles_all:
            # sort so PANGU then PANGU_POST are adjacent and at the top
            pairs = list(zip(model_handles_all, model_labels_all))
            pairs.sort(key=lambda hl: _legend_sort_key(hl[1]))
            model_handles_sorted, model_labels_sorted = zip(*pairs)
            ncols_ = 2 if base == "AE_wind" else 1
            leg_models = ax.legend(
                model_handles_sorted,
                model_labels_sorted,
                fontsize=11,
                loc="upper left",
                frameon=False,
                ncols=ncols_,
            )
        else:
            leg_models = None
        # Baselines legend with a title
        if base_handles:
            if base in ("DPE_GCD", "CRPS_haversine"):
                leg_base = ax.legend(
                    base_handles,
                    base_labels,
                    fontsize=11,
                    loc="lower right",
                    bbox_to_anchor=(1.0, 0.10),  # raise slightly while staying in-axes
                    frameon=False,
                    title="Baselines",
                )
            else:
                leg_base = ax.legend(
                    base_handles,
                    base_labels,
                    fontsize=11,
                    loc="lower right",
                    frameon=False,
                    title="Baselines",
                )
            if leg_models is not None:
                ax.add_artist(leg_models)
        elif leg_models is None:
            # Fallback single legend if nothing else
            ax.legend(fontsize=11, loc="upper left", frameon=False)

TEST_YEAR = 2023
fig.suptitle(
    f"RAW (non-filled) comparison — error vs lead (test year {TEST_YEAR}). Models on native coverage; R² uses IBTrACS.",
    fontsize=14,
    y=0.995,
)


fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("TCBench_raw_comparison.pdf", format="pdf", bbox_inches="tight")
plt.show()

#
# ---- Shared helpers for FAIR/CLIM-FAIR coverage (IBTrACS-denominator & raw-model coverage)

IBTRACS_DIR = args.ibtracs_path
IBTRACS_2023_CSV = os.path.join(IBTRACS_DIR, "IBTrACS_2023.csv")


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


# Cache once for reuse in figures
ACTIVE_TRACKS_2023 = _ibtracs_active_tracks_by_lead(ibtracs, EXPLICIT_LEADS)
# Precompute IBTrACS verification-pair counts (keys) by lead
IB_KEYS_2023 = _build_ib_denominator_counts_unified(leads=EXPLICIT_LEADS, ib_df=ibtracs)
print("[DEBUG][IB_DENOM] leads present:", list(IB_KEYS_2023.index.astype(int)))


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


# %% FAIR comparison figure (models filled with persistence on IBTrACS grid)
# This reproduces persistence_plotter functionality inside plot_errors.py

#
# Fresh load of persistence (do not reuse any earlier in-memory copy)
if os.path.exists(PERSIST_PATH):
    _persist_fair = pd.read_csv(PERSIST_PATH, low_memory=False)
    for c in ("Initial Time", "Valid Time"):
        if c in _persist_fair.columns:
            _persist_fair[c] = pd.to_datetime(_persist_fair[c], errors="coerce")
    if {"Initial Time", "Valid Time"}.issubset(_persist_fair.columns):
        _persist_fair["lead_hours"] = (
            _persist_fair["Valid Time"] - _persist_fair["Initial Time"]
        ).dt.total_seconds() / 3600.0
else:
    _persist_fair = None

if _persist_fair is not None:
    persist_df = _persist_fair.copy()
    # small debug to ensure same population as persistence_plotter
    try:
        _cnt = (
            persist_df[["lead_hours", "DPE_GCD"]]
            .dropna()
            .groupby("lead_hours")["DPE_GCD"]
            .size()
            .sum()
        )
        print(f"[FAIR] Loaded persistence rows (DPE_GCD non-NaN total): {_cnt}")
    except Exception:
        pass

    # Define helper used by the fair comparison bits
    if "pick_persist_col" not in globals():

        def pick_persist_col(df: pd.DataFrame, base: str) -> str | None:
            cols = set(df.columns)
            if base in cols:
                return base
            mean_col = f"{base}_mean"
            if mean_col in cols:
                return mean_col
            # legacy fallback for DPE
            if base == "DPE_GCD" and "DPE" in cols:
                return "DPE"
            return None

    # Build persistence tables per metric (keys + value renamed to 'persist')
    KEYS = ["SID", "Initial Time", "Valid Time"]
    persist_tables = {}
    for base in ("DPE_GCD", "AE_pressure", "AE_wind", "SE_pressure", "SE_wind"):
        col = pick_persist_col(persist_df, base)
        if col is not None and set(KEYS).issubset(persist_df.columns):
            tab = persist_df[KEYS + [col]].copy().rename(columns={col: "persist"})
            persist_tables[base] = tab

    # Build IBTrACS denominator counts (same helper as used in CLIM-FAIR, inlined here)
    def _ib_denom_counts_for_fair(
        leads=(
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
    ):
        return _build_ib_denominator_counts_unified(
            leads=leads, ib_df=globals().get("ibtracs", None)
        )

    def filled_metric_series(
        model_df: pd.DataFrame, base_col: str, model_col: str, allowed_leads
    ) -> pd.DataFrame:
        """Return KEYS + ['lead_hours','filled','from_model'] for one metric on the IB grid.
        Uses model metric when available; otherwise persistence metric. from_model marks source.
        Only keeps rows whose lead_hours ∈ allowed_leads (per-model).
        """
        mmin = (
            model_df[KEYS + [model_col]].copy()
            if model_col in model_df.columns
            else model_df[KEYS].copy()
        )
        base_tab = persist_tables[base_col]
        merged = base_tab.merge(mmin, on=KEYS, how="left", suffixes=("", "_model"))
        if model_col in merged.columns:
            filled_vals = merged[model_col].where(
                ~merged[model_col].isna(), merged["persist"]
            )
            from_model = ~merged[model_col].isna()
        else:
            filled_vals = merged["persist"]
            from_model = pd.Series(False, index=merged.index)

        out = merged[KEYS].copy()
        out["lead_hours"] = (
            out["Valid Time"] - out["Initial Time"]
        ).dt.total_seconds() / 3600.0

        # --- Keep only the allowed fill leads (per-model)
        out["lead_hours"] = out["lead_hours"].round().astype(float)
        mask_allowed = out["lead_hours"].isin(allowed_leads)
        out = out.loc[mask_allowed].copy()
        out["filled"] = pd.Series(filled_vals, index=merged.index).loc[out.index].values
        out["from_model"] = (
            pd.Series(from_model, index=merged.index).loc[out.index].values
        )
        return out

    # Discover model files to overlay (strictly exclude baselines & climatology)
    overlay_files = []
    for f in os.listdir(EVAL_DIR):
        if not f.endswith("_results.csv"):
            continue
        low = f.lower()
        if any(tok in low for tok in ("persistence", "ri", "climatology", "tigge_ifs")):
            continue
        # also exclude the explicit climatology results file if present
        if os.path.exists(CLIM_FILE) and f == os.path.basename(CLIM_FILE):
            continue
        # only include ANN post-processing among postprocessing files
        if "postprocessing" in low:
            if "postprocessing_panguweather_ann" in low:
                overlay_files.append(f)
            else:
                continue
        else:
            overlay_files.append(f)

    # Strict helper to add lead_hours (no synthetic fallback)
    def _add_lead_hours_strict(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in ("Initial Time", "Valid Time"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        df["lead_hours"] = (
            df["Valid Time"] - df["Initial Time"]
        ).dt.total_seconds() / 3600.0
        return df

    overlays = {}
    for fn in sorted(overlay_files):
        dfm = pd.read_csv(os.path.join(EVAL_DIR, fn), low_memory=False)
        dfm = _add_lead_hours_strict(dfm)
        label = fn.replace("_results.csv", "")
        # Safety: never treat climatology as a model to be "filled"
        if _is_climatology(label) or "climatology" in label.lower():
            # Skip entirely from overlays so it is never merged with persistence
            continue
        overlays[label] = dfm

    # Debug: list overlays to ensure climatology is not present
    try:
        bad = [
            k
            for k in overlays.keys()
            if _is_climatology(k) or "climatology" in k.lower()
        ]
        if bad:
            print(f"⚠️  Unexpected climatology in overlays (will be ignored): {bad}")
    except Exception:
        pass

    # --- Figure: six error panels (2x3)
    metrics_to_plot_fair = [
        ("DPE_GCD", "a. DPE", "km"),
        ("CRPS_haversine", "b. CRPS", "km"),
        ("AE_pressure", "c. AE ", "hPa"),
        ("CRPS_pmin", "d. CRPS", "hPa"),
        ("AE_wind", "e. AE", "kt"),
        ("CRPS_vmax", "f. CRPS", "kt"),
    ]

    # Use GridSpec to allocate 3 rows of metrics (2x3), no embedded coverage panel
    fig2 = plt.figure(figsize=(14, 12.5))
    fig2.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.06)
    gs = fig2.add_gridspec(nrows=3, ncols=2, height_ratios=[1, 1, 1])
    # Six metric axes in 3 rows x 2 cols
    axes2 = [
        fig2.add_subplot(gs[i // 2, i % 2]) for i in range(len(metrics_to_plot_fair))
    ]

    coverage_by_base = {}
    coverage_any = {}

    for i, (base, title, unit) in enumerate(metrics_to_plot_fair):
        ax = axes2[i]
        ax2 = ax.twinx()
        ax2.set_ylabel("Cases (n)")
        # Reset legend containers for each panel
        model_handles, model_labels = [], []
        base_handles, base_labels = [], []
        y_max_plotted = 0.0

        # Persistence mean (thick black dashed), if available for this metric or its MAE proxy for CRPS
        h_persist = None
        # For CRPS panels, persistence CRPS equals the corresponding MAE (deterministic delta forecast).
        persist_source_col = None
        if base in persist_df.columns:
            persist_source_col = base
            persist_label = "Persistence"
        else:
            # Try MAE proxy for CRPS panels
            persist_source_col = CRPS_TO_MAE.get(base)
            persist_label = (
                "Persistence (MAE proxy)" if persist_source_col is not None else None
            )
            if (
                persist_source_col is not None
                and persist_source_col not in persist_df.columns
            ):
                persist_source_col = None  # column truly unavailable
        if persist_source_col is not None:
            gp = (
                persist_df[["lead_hours", persist_source_col]]
                .dropna()
                .groupby("lead_hours")[persist_source_col]
            )
            p_agg = gp.agg(["mean"]).sort_index()
            # restrict to allowed fill leads for baselines: always full EXPLICIT_LEADS
            p_agg = p_agg.reindex(EXPLICIT_LEADS).dropna()
            h_persist = ax.plot(
                p_agg.index.values,
                p_agg["mean"].values,
                linestyle="--",
                color=PERSIST_COLOR,
                linewidth=3,
                label=persist_label,
            )[0]
            try:
                y_max_plotted = max(
                    y_max_plotted, float(np.nanmax(p_agg["mean"].values))
                )
            except Exception:
                pass
            base_handles.append(h_persist)
            base_labels.append(persist_label)

        # Climatology mean (thick dashed cyan), if available
        h_clim = None
        if os.path.exists(CLIM_FILE):
            try:
                _clim_df = pd.read_csv(CLIM_FILE, low_memory=False)
                for cdt in ("Initial Time", "Valid Time"):
                    if cdt in _clim_df.columns:
                        _clim_df[cdt] = pd.to_datetime(_clim_df[cdt], errors="coerce")
                _clim_df["lead_hours"] = (
                    _clim_df["Valid Time"] - _clim_df["Initial Time"]
                ).dt.total_seconds() / 3600.0
                clim_col = pick_metric_col(_clim_df, base)  # prefer *_mean
                if clim_col is not None:
                    c_agg = (
                        _clim_df[["lead_hours", clim_col]]
                        .dropna()
                        .groupby("lead_hours")[clim_col]
                        .mean()
                        .sort_index()
                    )
                    # restrict to allowed fill leads for baselines: always full EXPLICIT_LEADS
                    c_agg = c_agg.reindex(EXPLICIT_LEADS).dropna()
                    if len(c_agg):
                        h_clim = ax.plot(
                            c_agg.index.values,
                            c_agg.values,
                            linestyle="--",
                            color=CLIM_COLOR,
                            linewidth=3,
                            label="MT-LB",
                            zorder=8,
                        )[0]
                        try:
                            y_max_plotted = max(
                                y_max_plotted, float(np.nanmax(c_agg.values))
                            )
                        except Exception:
                            pass
            except Exception as _e:
                print(f"⚠️ Could not plot climatology baseline for {base}: {_e}")
        # If climatology was plotted, include it in the Baselines legend group
        if h_clim is not None:
            base_handles.append(h_clim)
            base_labels.append("MT-LB")

        # Right axis: number of IBTrACS verification pairs (SID, t0, vt) per lead
        ax2.plot(
            IB_KEYS_2023.index.values.astype(float),
            IB_KEYS_2023.values,
            linestyle=":",
            linewidth=1.0,
            color="0.5",
            alpha=0.8,
        )
        ax2.set_ylabel("Cases (n)")

        coverage_msgs = []
        for label, dfm in overlays.items():
            # Extra safety (should already be filtered in overlays):
            if _is_climatology(label) or "climatology" in label.lower():
                continue
            pretty_label = pretty_curve_label(label)
            model_col = pick_metric_col(dfm, base)
            if model_col is None:
                continue

            # Determine allowed leads for this model: post-processing models get sparse, others get full
            is_postproc = "postprocessing" in label.lower()
            allowed_leads = PP_FILL_LEADS if is_postproc else FULL_FILL_LEADS

            # FAIR filling: for CRPS panels use the corresponding MAE as the persistence "filled" value.
            if base.startswith("CRPS"):
                persist_base = CRPS_TO_MAE.get(base)
                if persist_base not in persist_tables:
                    # If for some reason the proxy table is missing, fall back to model-only mean
                    agg = (
                        dfm[["lead_hours", model_col]]
                        .dropna()
                        .groupby("lead_hours")[model_col]
                        .mean()
                        .sort_index()
                    )
                    # restrict to allowed fill leads per-model
                    try:
                        idx = agg.index.astype(float)
                    except Exception:
                        idx = agg.index
                    agg = (
                        pd.Series(agg.values, index=idx).reindex(allowed_leads).dropna()
                    )
                    sfilled = None
                else:
                    sfilled = filled_metric_series(
                        dfm,
                        base_col=persist_base,
                        model_col=model_col if model_col else "__missing__",
                        allowed_leads=allowed_leads,
                    )
                    agg = sfilled.groupby("lead_hours")["filled"].mean().sort_index()
                    try:
                        idx = agg.index.astype(float)
                    except Exception:
                        idx = agg.index
                    agg = (
                        pd.Series(agg.values, index=idx).reindex(allowed_leads).dropna()
                    )
            else:
                sfilled = filled_metric_series(
                    dfm,
                    base_col=base,
                    model_col=model_col if model_col else "__missing__",
                    allowed_leads=allowed_leads,
                )
                agg = sfilled.groupby("lead_hours")["filled"].mean().sort_index()
                try:
                    idx = agg.index.astype(float)
                except Exception:
                    idx = agg.index
                agg = pd.Series(agg.values, index=idx).reindex(allowed_leads).dropna()
            plot_kwargs = dict(
                linestyle="-",
                marker="o",
                linewidth=1.3,
                markersize=3,
                label=pretty_label,
                color=color_for(pretty_label),
            )
            if _is_google(label):
                plot_kwargs.update(
                    dict(color="#7a7a7a", linestyle=":", marker="s", linewidth=2.0)
                )
            elif _is_genc(label) or _is_pangu_post(label):
                plot_kwargs.update(dict(linestyle=":", marker="s", linewidth=2.0))
            (h_model,) = ax.plot(agg.index.values, agg.values, **plot_kwargs)
            try:
                y_max_plotted = max(y_max_plotted, float(np.nanmax(agg.values)))
            except Exception:
                pass
            model_handles.append(h_model)
            model_labels.append(pretty_label)
            if sfilled is not None and "from_model" in sfilled.columns:
                cov = sfilled.groupby("lead_hours")["from_model"].mean().sort_index()
                coverage_msgs.append((label, float(cov.mean()) if len(cov) else 0.0))

        ax.set_title(f"{title} ({unit})")
        ax.set_xlabel("Lead time (hours)")
        # ax.set_ylabel(f"{title} ({unit})")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(6, 120)
        # Set y-axis cap for DPE panel
        if base == "DPE_GCD":
            ax.set_ylim(0, 2000)
        # Remove hard cap for CRPS_haversine, use dynamic headroom
        # if base == "CRPS_haversine":
        #     ax.set_ylim(0, 400)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(24))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(6))
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.12)
        # Extra room on the right for twin y-axis labels
        ax.margins(x=0.02)
        # Dynamic y-limits if not capped
        if y_max_plotted > 0:
            # Give 10% headroom so curves don't hit the top
            ax.set_ylim(0, y_max_plotted * 1.10)
        # Build separate legends: models and baselines
        if model_handles:
            pairs = list(zip(model_handles, model_labels))
            pairs.sort(key=lambda hl: _legend_sort_key(hl[1]))
            model_handles_sorted, model_labels_sorted = zip(*pairs)
            leg_models = ax.legend(
                model_handles_sorted,
                model_labels_sorted,
                fontsize=11,
                ncols=2,
                loc="upper left",
                frameon=False,
            )
        else:
            leg_models = None
        if base_handles:
            leg_base = ax.legend(
                base_handles,
                base_labels,
                fontsize=11,
                loc="lower right",
                frameon=False,
                title="Baselines",
            )
            if leg_models is not None:
                ax.add_artist(leg_models)

    # Final layout, then add column/row group titles in figure coordinates
    fig2.suptitle(
        "FAIR comparison — metrics on IBTrACS grid (missing values filled by persistence)",
        fontsize=14,
        y=0.98,
    )
    fig2.tight_layout(rect=[0.04, 0.06, 0.98, 0.90])

    # --- Compute positions for column headers and vertical group labels
    # Metric axes in row-major order: 0:(DPE_GCD),1:(CRPS_haversine); 2:(AE_p),3:(CRPS_p); 4:(AE_w),5:(CRPS_w)
    left_col_axes = [axes2[0], axes2[2], axes2[4]]
    right_col_axes = [axes2[1], axes2[3], axes2[5]]

    # Column centers (x) and the top y of the top row
    left_center_x = np.mean(
        [ax.get_position().x0 + ax.get_position().width / 2.0 for ax in left_col_axes]
    )
    right_center_x = np.mean(
        [ax.get_position().x0 + ax.get_position().width / 2.0 for ax in right_col_axes]
    )
    top_y = max(axes2[0].get_position().y1, axes2[1].get_position().y1)

    # Place column titles slightly above the top row, with rounded boxes
    col_y = min(0.98, top_y + 0.035)  # clamp within the figure
    col_bbox = dict(
        boxstyle="round,pad=0.3,rounding_size=0.15",
        facecolor="white",
        edgecolor="0.5",
        linewidth=0.8,
        alpha=0.95,
    )
    fig2.text(
        left_center_x,
        col_y,
        "Deterministic",
        ha="center",
        va="bottom",
        fontsize=15,
        fontweight="bold",
        bbox=col_bbox,
    )
    fig2.text(
        right_center_x,
        col_y,
        "Probabilistic",
        ha="center",
        va="bottom",
        fontsize=15,
        fontweight="bold",
        bbox=col_bbox,
    )

    # --- Vertical row-group labels on the left: "Track" (row 1) and "Intensity" (rows 2–3)
    # Row 1 (track) center y:
    row1_y0, row1_y1 = axes2[0].get_position().y0, axes2[0].get_position().y1
    row1_center_y = 0.5 * (row1_y0 + row1_y1)
    # Rows 2 and 3 (intensity) center y: average the centers of rows 2 and 3
    row2_center_y = 0.5 * (axes2[2].get_position().y0 + axes2[2].get_position().y1)
    row3_center_y = 0.5 * (axes2[4].get_position().y0 + axes2[4].get_position().y1)
    intensity_center_y = 0.5 * (row2_center_y + row3_center_y)

    # Left margin for vertical labels: a bit left of the leftmost axes
    left_edge = min(ax.get_position().x0 for ax in left_col_axes)
    label_x = max(0.005, left_edge - 0.055)

    # Bold with LaTeX mathtext and add rounded box style
    col_bbox_vert = dict(
        boxstyle="round,pad=0.3,rounding_size=0.15",
        facecolor="white",
        edgecolor="0.5",
        linewidth=0.8,
        alpha=0.95,
    )
    fig2.text(
        label_x,
        row1_center_y,
        r"$\mathbf{Track}$ Error",
        rotation=90,
        ha="center",
        va="center",
        fontsize=14,
        bbox=col_bbox_vert,
    )
    fig2.text(
        label_x,
        intensity_center_y,
        r"$\mathbf{Intensity Error}$ - Wind (kt) & Pressure (hPa)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=14,
        bbox=col_bbox_vert,
    )

    fig2.savefig("TCBench_fair_comparison.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # --- Separate Coverage figure (vs IBTrACS): % of IB keys where the raw model has any row
    fig_cov = plt.figure(figsize=(10, 5.0))
    cov_ax = fig_cov.add_subplot(111)
    cov_ax.set_xlabel("Lead time (hours)")
    cov_ax.set_title("Coverage (% of IB pairs)")
    cov_ax.set_xlim(6, 120)
    cov_ax.xaxis.set_major_locator(mticker.MultipleLocator(24))
    cov_ax.xaxis.set_minor_locator(mticker.MultipleLocator(6))
    cov_ax.grid(True, which="major", alpha=0.35)
    cov_ax.grid(True, which="minor", alpha=0.12)

    cov_max = 0.0
    for label, dfm in overlays.items():
        ib_denom_cov = (
            _aligned_ib_denominator_for_model(dfm, ibtracs)
            if COVERAGE_DENOM == "aligned"
            else IB_KEYS_2023
        )
        if VERBOSE_COVERAGE:
            _tmp = dfm.copy()
            for c in ("Initial Time", "Valid Time"):
                if c in _tmp.columns:
                    _tmp[c] = pd.to_datetime(_tmp[c], errors="coerce")
            _tmp = _tmp.dropna(subset=["SID", "Initial Time", "Valid Time"]).copy()
            if "INIT_HOURS" in globals() and INIT_HOURS:
                _tmp = _tmp[_tmp["Initial Time"].dt.hour.isin(INIT_HOURS)].copy()
            _tmp["init_hr"] = (
                _tmp["Initial Time"].astype("int64", copy=False) // 3_600_000_000_000
            )
            _tmp["valid_hr"] = (
                _tmp["Valid Time"].astype("int64", copy=False) // 3_600_000_000_000
            )
            _tmp = _tmp[(_tmp["init_hr"] % 6 == 0) & (_tmp["valid_hr"] % 6 == 0)]
            model_keys = set(zip(_tmp["SID"], _tmp["init_hr"], _tmp["valid_hr"]))
            denom_sum = int(np.nan_to_num(ib_denom_cov.reindex(EXPLICIT_LEADS).sum()))
            print(
                f"[COVERAGE] {pretty_curve_label(label)} — aligned IB pairs (sum)={denom_sum}, model pairs={len(model_keys)}"
            )

        cov_series = _coverage_series_vs_ib_from_raw(dfm, ib_denom_cov)
        cov_series = cov_series.reindex(EXPLICIT_LEADS).astype(float)
        x = np.asarray(EXPLICIT_LEADS, dtype=float)
        y = cov_series.values
        m = ~np.isnan(y)
        x, y = x[m], y[m]
        if x.size == 0:
            continue
        order = np.argsort(x)
        x, y = x[order], y[order]

        try:
            cov_max = max(cov_max, float(np.nanmax(y)))
        except Exception:
            pass

        pretty_label = pretty_curve_label(label)
        plot_kwargs = dict(
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=pretty_label,
            zorder=3,
            color=color_for(pretty_label),
        )
        if _is_google(label):
            plot_kwargs.update(
                dict(color="#7a7a7a", linestyle=":", marker="s", linewidth=2.0)
            )
        elif _is_genc(label) or _is_pangu_post(label):
            plot_kwargs.update(dict(linestyle=":", marker="s", linewidth=2.0))
        (line,) = cov_ax.plot(x, y, **plot_kwargs)
        cov_ax.fill_between(x, 0, y, color=line.get_color(), alpha=0.15, zorder=2)

    if cov_max > 0:
        cov_ax.set_ylim(0, cov_max * 1.10)

    handles_cov, labels_cov = cov_ax.get_legend_handles_labels()
    pairs_cov = list(zip(handles_cov, labels_cov))
    pairs_cov.sort(key=lambda hl: _legend_sort_key(hl[1]))
    handles_cov_sorted, labels_cov_sorted = zip(*pairs_cov) if pairs_cov else ([], [])
    cov_ax.legend(
        handles_cov_sorted,
        labels_cov_sorted,
        fontsize=11,
        ncols=3,
        frameon=False,
        loc="upper left",
    )

    fig_cov.tight_layout(rect=[0.04, 0.10, 0.98, 0.95])
    fig_cov.savefig("TCBench_coverage.pdf", format="pdf", bbox_inches="tight")
    plt.show()
else:
    print("⚠️  Skipping FAIR comparison figure: persistence_results.csv not found.")


# %% plot scorecard

# --- TCBench scorecard (WeatherBench-style) ---

# ==== CONFIG ====
EVAL_DIR = args.eval_dir
# Which metrics and labels to plot (one panel per metric)
METRICS = [
    ("AE_wind", "Abs. Error vmax (kt)"),
    ("AE_pressure", "Abs. Error pmin (hPa)"),
    ("RI_CSI", "Rapid Intensification — CSI"),
]
# Which lead hours to show as columns
LEADS = [12, 24, 36, 48, 72, 96, 120]
# Baseline to compare against (name must match the row label below)
BASELINE_NAME = "Persistence"  # or e.g., "2023_PANGU", "2023_TIGGE_GEFS"
ROUND_TO = {"AE_wind": 1, "AE_pressure": 1, "RI_CSI": 3}  # annotation rounding


# ==== HELPERS ====
def add_lead_hours(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("Initial Time", "Valid Time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if {"Initial Time", "Valid Time"}.issubset(df.columns):
        df["lead_hours"] = (
            df["Valid Time"] - df["Initial Time"]
        ).dt.total_seconds() / 3600.0
    else:
        # fallback (rare)
        df["lead_hours"] = np.nan
    return df


def pretty_name(filename_no_ext: str) -> str:
    # remove year prefix and tidy underscores
    name = filename_no_ext
    name = name.replace("_results", "")
    name = name.replace("_clean", "")
    # keep year if it's part of the labeling scheme, but compact
    name = name.replace("2023_", "")
    if name.startswith("2023_"):
        name = name[5:]
    low = name.lower()
    if _is_google(name):
        return "$\\it{FNV3}$"
    if "genc" in low:
        return "$\\it{GENC}$"
    if "postprocessing_panguweather_ann" in low:
        return "PANGU_POST_ANN"
    if "postprocessing_panguweather_mlr" in low:
        return "PANGU_POST_MLR"
    if "postprocessing_panguweather_unet" in low:
        return "PANGU_POST_UNET"
    return name


def load_results_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return add_lead_hours(df)


def aggregate_by_lead(df: pd.DataFrame, metric_col: str) -> pd.Series:
    g = (
        df[["lead_hours", metric_col]]
        .dropna(subset=["lead_hours"])
        .groupby("lead_hours")[metric_col]
    )
    return g.mean().sort_index()


_PPOST_KEYS = ("postprocessing_panguweather_ann",)
# ==== LOAD DATA ====
# persistence (as its own row)
persist_path = os.path.join(EVAL_DIR, "persistence_results.csv")
if not os.path.exists(persist_path):
    raise FileNotFoundError("persistence_results.csv not found in EVAL_DIR")

persistence = load_results_file(persist_path)
# model result files (exclude persistence + any *_RI files, and exclude climatology)
model_files = [
    f
    for f in os.listdir(EVAL_DIR)
    if f.endswith("_results.csv")
    and "persistence" not in f.lower()
    and "ri" not in f.lower()
    and "climatology" not in f.lower()
    and "tigge_ifs" not in f.lower()
]
model_files = [
    f
    for f in model_files
    if ("postprocessing" not in f.lower()) or any(k in f.lower() for k in _PPOST_KEYS)
]
_MODEL_FILEMAP = {}
models = {}
for fn in sorted(model_files):
    df = load_results_file(os.path.join(EVAL_DIR, fn))
    pretty = pretty_name(fn[:-4])
    models[pretty] = df  # strip .csv
    _MODEL_FILEMAP[pretty] = fn

# Inject persistence into models dict using a consistent label
models = {"Persistence": persistence, **models}


# Reorder so PANGU then PANGU_POST
def _reorder_models_for_display(d: dict) -> dict:
    order = {}
    if "Persistence" in d:
        order["Persistence"] = d["Persistence"]
    # PANGU (base) before PANGU_POST
    pangu_keys = [
        k
        for k in d.keys()
        if k.upper().startswith("PANGU") and k != "PANGU_POST" and k != "Persistence"
    ]
    for k in pangu_keys:
        order[k] = d[k]
    if "PANGU_POST" in d:
        order["PANGU_POST"] = d["PANGU_POST"]
    for k, v in d.items():
        if k not in order:
            order[k] = v
    return order


# Helper to enforce consistent row order in scorecards: Persistence, PANGU, PANGU_POST, others, FNV3/Google last
def _order_rows_fnv3_last(index_list):
    """Return index order with Persistence first, then PANGU, PANGU_POST,
    then all other non-Google rows, and FNV3 (Google) last."""
    names = list(index_list)
    is_google = (
        lambda n: _is_google(n) or str(n).lower() == "fnv3" or "fnv3" in str(n).lower()
    )
    order = []
    if "Persistence" in names:
        order.append("Persistence")
    if "PANGU" in names:
        order.append("PANGU")
    if "PANGU_POST" in names:
        order.append("PANGU_POST")
    # add all other non-google rows (preserve existing relative order)
    order += [n for n in names if (n not in order) and (not is_google(n))]
    # finally any Google/FNV3 rows
    order += [n for n in names if (n not in order) and is_google(n)]
    # keep only those actually present (defensive)
    order = [n for n in order if n in names]
    return order


models = _reorder_models_for_display(models)


# Inject RI_CSI per‑lead values (from *_RI.csv) into each model dataframe so it shows up in the scorecard
def _compute_ri_csi_series(df_ri: pd.DataFrame) -> pd.Series:
    """
    Return per‑lead CSI series if present in an RI file. Accepts 'RI_CSI' or 'CSI'.
    """
    df = df_ri.copy()
    for c in ("Initial Time", "Valid Time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if "lead_hours" not in df.columns and {"Initial Time", "Valid Time"}.issubset(
        df.columns
    ):
        df["lead_hours"] = (
            df["Valid Time"] - df["Initial Time"]
        ).dt.total_seconds() / 3600.0
    col = (
        "RI_CSI" if "RI_CSI" in df.columns else ("CSI" if "CSI" in df.columns else None)
    )
    if col is None or "lead_hours" not in df.columns:
        return pd.Series(dtype=float)
    s = (
        df[["lead_hours", col]]
        .dropna(subset=["lead_hours", col])
        .groupby(df["lead_hours"].round().astype(int))[col]
        .mean()
        .sort_index()
    )
    s.index = s.index.astype(int)
    return s


# Inject RI_CSI per‑lead values (from *_RI.csv) into each model dataframe so it shows up in the scorecard
for pretty_label, fn in list(_MODEL_FILEMAP.items()):
    ri_fn = fn.replace("_results.csv", "_RI.csv")
    ri_path = os.path.join(EVAL_DIR, ri_fn)
    if not os.path.exists(ri_path):
        continue
    try:
        df_ri = pd.read_csv(ri_path, low_memory=False)
    except Exception:
        continue
    s_csi = _compute_ri_csi_series(df_ri)
    if s_csi.empty:
        continue
    # Map per‑lead CSI into the model rows (constant per lead), so aggregate_by_lead() picks it up
    mdf = models.get(pretty_label)
    if mdf is None or "lead_hours" not in mdf.columns:
        continue
    m = mdf.copy()
    m["lead_hours_int"] = m["lead_hours"].round().astype(int)
    m["RI_CSI"] = m["lead_hours_int"].map(s_csi).astype(float)
    m.drop(columns=["lead_hours_int"], inplace=True)
    models[pretty_label] = m
#
# Add a title to the deterministic scorecard before starting the probabilistic block
try:
    _fig_det = plt.gcf()
    _fig_det.suptitle(
        "TCBench Scorecard — Deterministic (AE & RI CSI), Test Year 2023 — Baseline: Persistence [BASE]",
        y=0.98,
        fontsize=14,
    )
    # Ensure the title is not clipped by tight_layout / constrained_layout
    try:
        _fig_det.tight_layout(rect=[0.02, 0.03, 0.98, 0.94])
    except Exception:
        pass
except Exception:
    pass
# %% probabilistic scorecard (CRPS)
METRICS_PROB = [
    ("CRPS_haversine", "CRPS — track displacement (km)"),
    ("CRPS_vmax", "CRPS — max wind (kt)"),
    ("CRPS_pmin", "CRPS — min pressure (hPa)"),
]
LEADS_PROB = [12, 24, 36, 48, 72, 96, 120]
BASELINE_NAME = "Persistence"
ROUND_TO = {"CRPS_haversine": 0, "CRPS_vmax": 1, "CRPS_pmin": 1}
# load files (reuse EVAL_DIR); keep only ANN postproc among postprocessing files
persist_path = os.path.join(EVAL_DIR, "persistence_results.csv")
persistence = load_results_file(persist_path)
model_files = [
    f
    for f in os.listdir(EVAL_DIR)
    if f.endswith("_results.csv")
    and "persistence" not in f.lower()
    and "ri" not in f.lower()
    and "climatology" not in f.lower()
    and "tigge_ifs" not in f.lower()
]
model_files = [
    f
    for f in model_files
    if ("postprocessing" not in f.lower()) or any(k in f.lower() for k in _PPOST_KEYS)
]
models = {"Persistence": persistence}
for fn in sorted(model_files):
    df = load_results_file(os.path.join(EVAL_DIR, fn))
    models[pretty_name(fn[:-4])] = df
models = _reorder_models_for_display(models)
# build tables
tables_abs = {}
tables_pct = {}
for base, _label in [(m[0], m[1]) for m in METRICS_PROB]:
    rows_abs = {}
    # --- Build/derive baseline series first
    base_proxy = CRPS_TO_MAE.get(base)
    # persistence: use CRPS column if present, else MAE proxy
    pers_col = pick_metric_col(persistence, base)
    if pers_col is None and base_proxy is not None:
        pers_col = pick_metric_col(persistence, base_proxy)
    if pers_col is not None:
        s_pers = aggregate_by_lead(persistence, pers_col)
        rows_abs["Persistence"] = [s_pers.get(lh, np.nan) for lh in LEADS_PROB]
    # --- Add models (skip those missing the metric)
    for model_name, df in models.items():
        if model_name == "Persistence":
            continue
        col = pick_metric_col(df, base)
        if col is None:
            # allow deterministic-only models by skipping silently
            continue
        s = aggregate_by_lead(df, col)
        rows_abs[model_name] = [s.get(lh, np.nan) for lh in LEADS_PROB]
    if not rows_abs or "Persistence" not in rows_abs:
        continue
    A = pd.DataFrame.from_dict(rows_abs, orient="index", columns=LEADS_PROB)
    baseline = A.loc["Persistence"]
    P = (A.subtract(baseline, axis=1)).divide(baseline, axis=1) * 100.0
    # Ensure FNV3/Google is last for readability
    order_rows = _order_rows_fnv3_last(A.index.tolist())
    A = A.loc[order_rows]
    P = P.loc[order_rows]
    tables_abs[base] = A
    tables_pct[base] = P
# plot heatmaps
ncols = len(METRICS_PROB)
fig_prob, axes = plt.subplots(
    1, ncols, figsize=(5.2 * ncols, 9), constrained_layout=False
)
if ncols == 1:
    axes = [axes]

_HATCH_LW_THIN = 0.15
_HATCH_PATTERN = "/"
for ax, (base, title) in zip(
    axes[: len(METRICS_PROB)], [(m[0], m[1]) for m in METRICS_PROB]
):
    A = tables_abs.get(base)
    P = tables_pct.get(base)
    if A is None or P is None:
        ax.set_visible(False)
        continue
    vmax = np.nanpercentile(np.abs(P.values), 95)
    vmax = max(5, float(vmax))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    # Use a truncated, desaturated version of RdBu_r for better readability
    cmap = plt.get_cmap("RdBu_r")
    rows, cols = A.shape
    # Identify Google/FNV3 rows and other special rows
    names_list = A.index.tolist()
    google_rows = [i for i, name in enumerate(names_list) if _is_google(name)]
    genc_rows = [i for i, name in enumerate(names_list) if _is_genc(name)]
    pangu_post_rows = [i for i, name in enumerate(names_list) if _is_pangu_post(name)]
    special_rows = set(google_rows + genc_rows + pangu_post_rows)
    # Subtle hatch for proprietary Google row
    _rc_hatch_prev_lw = mpl.rcParams.get("hatch.linewidth", 1.0)
    _rc_hatch_prev_col = mpl.rcParams.get("hatch.color", "black")
    mpl.rcParams["hatch.linewidth"] = 0.15
    mpl.rcParams["hatch.color"] = "black"
    # Draw subtle background for special rows (e.g., Google WeatherLab FNV3 row, GENC, PANGU_POST)
    for gi in special_rows:
        ax.add_patch(
            plt.Rectangle(
                (0, gi),
                cols,
                1,
                facecolor=(0.92, 0.92, 0.92, 1.0),
                edgecolor="none",
                zorder=0,
            )
        )
    # Draw heatmap cells with solid white border, and hatch for special rows
    for i in range(rows):
        for j in range(cols):
            val_pct = P.iloc[i, j]
            if np.isnan(val_pct):
                color = (0.9, 0.9, 0.9, 1.0)
            else:
                col = cmap(norm(val_pct))
                # Add slight transparency for the track CRPS panel to ease hatch/text readability
                if base == "CRPS_haversine":
                    color = (col[0], col[1], col[2], 0.92)
                else:
                    color = col
            is_special_row = i in special_rows
            ax.add_patch(
                plt.Rectangle(
                    (j, i),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=1.0,
                    hatch=(_HATCH_PATTERN if is_special_row else None),
                    zorder=1,
                )
            )
    mpl.rcParams["hatch.linewidth"] = _rc_hatch_prev_lw
    mpl.rcParams["hatch.color"] = _rc_hatch_prev_col
    # Write text annotations after rectangles so text is on top
    r = ROUND_TO.get(base, 1)
    for i in range(rows):
        for j in range(cols):
            val_abs = A.iloc[i, j]
            if np.isnan(val_abs):
                txt = "–"
                color = "0.4"
            else:
                if r == 0:
                    txt = f"{int(round(val_abs))}"
                else:
                    txt = f"{val_abs:.{r}f}"
                # For Google row, set white text for Track CRPS panel
                if i in google_rows and base == "CRPS_haversine":
                    color = "white"
                else:
                    color = "black"
            ax.text(
                j + 0.5,
                i + 0.5,
                txt,
                ha="center",
                va="center",
                fontsize=12,
                color=color,
            )
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_xticks(np.arange(cols) + 0.5)
    ax.set_xticklabels([str(lh) for lh in LEADS_PROB])
    ax.set_yticks(np.arange(rows) + 0.5)
    # Set yticklabels, replacing FNV3 and GENC rows with LaTeX italic label
    ylabels = []
    for name in A.index.tolist():
        if _is_google(name):
            ylabels.append("$\\it{FNV3}$")
        elif _is_genc(name):
            ylabels.append("$\\it{GENC}$")
        else:
            ylabels.append(name)
    ax.set_yticklabels(ylabels)
    ax.set_title(title, fontsize=15)
    # ax.set_xlabel("Lead Time (hours)")
    if ax is axes[0]:
        ax.set_ylabel("Models", fontsize=16)
    cax = ax.inset_axes([0.05, -0.08, 0.90, 0.03])
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal"
    )
    cb.set_label("% Difference vs Baseline (Lower Is Better)")
fig_prob.suptitle(
    f"TCBench Scorecard — Probabilistic (CRPS), Test Year {TEST_YEAR} — Baseline: Persistence [BASE]",
    fontsize=16,
    y=0.98,
)
"""
fig_prob.text(
    0.5,
    0.01,
    "Cells show mean CRPS; colors show % difference vs baseline at the same lead.",
    ha="center",
    fontsize=10,
)
"""
fig_prob.tight_layout(rect=[0, 0.04, 1, 0.95])
fig_prob.savefig(
    "TCBench_scorecard_probabilistic.pdf", format="pdf", bbox_inches="tight"
)
plt.show()


# Helper to compute RI CSI per lead directly from *_RI.csv files if needed
def _ri_scorecard_rows(lead_list, allowed_labels: set[str] | None = None):
    """Return dict: pretty_model_name -> list of CSI per lead in lead_list.
    Uses ibtracs_RI_gt.csv + *_RI.csv in EVAL_DIR. Missing predictions treated as no-RI.
    If `allowed_labels` is provided, only rows whose pretty label is in that set are kept.
    """
    gt_path = os.path.join(EVAL_DIR, "ibtracs_RI_gt.csv")
    if not os.path.exists(gt_path):
        return {}

    # Load GT
    gt = pd.read_csv(gt_path, low_memory=False)
    for c in ("Initial Time", "Valid Time"):
        if c in gt.columns:
            gt[c] = pd.to_datetime(gt[c], errors="coerce")

    # Choose RI true column
    ri_cols = [c for c in gt.columns if c.lower() in {"ri", "ri_true"}]
    if not ri_cols:
        return {}
    gt = gt[["SID", "Initial Time", "Valid Time", ri_cols[0]]].rename(
        columns={ri_cols[0]: "RI_true"}
    )
    gt["RI_true"] = (
        (
            gt["RI_true"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "1": True,
                    "true": True,
                    "t": True,
                    "yes": True,
                    "0": False,
                    "false": False,
                    "f": False,
                    "no": False,
                }
            )
        )
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )

    # Compute integer lead hours and restrict to requested leads
    gt = gt.dropna(subset=["Initial Time", "Valid Time"]).copy()
    gt["lead_hours"] = (
        (((gt["Valid Time"] - gt["Initial Time"]).dt.total_seconds()) / 3600.0)
        .round()
        .astype(int)
    )
    gt = gt[gt["lead_hours"].isin(lead_list)].copy()

    # Collect prediction files, applying filtering: exclude TIGGE_IFS, only keep ANN postprocessing among postprocessing_* variants
    pred_files = []
    for f in os.listdir(EVAL_DIR):
        if not f.endswith("_RI.csv"):
            continue
        low = f.lower()
        if "ibtracs_ri_gt" in low:
            continue
        if "tigge_ifs" in low:
            continue  # drop TIGGE_IFS everywhere
        # keep only the ANN postprocessing among postprocessing_* variants
        if "postprocessing" in low and "postprocessing_panguweather_ann" not in low:
            continue
        pred_files.append(f)
    pred_files.sort()

    rows: dict[str, list[float]] = {}
    for fn in pred_files:
        df = pd.read_csv(os.path.join(EVAL_DIR, fn), low_memory=False)
        for c in ("Initial Time", "Valid Time"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")

        # Find prediction column
        pcols = [
            c
            for c in df.columns
            if c.lower() in {"ri", "ri_pred", "ri_hat", "ri_prediction"}
        ]
        if not pcols:
            continue

        df = df[["SID", "Initial Time", "Valid Time", pcols[0]]].rename(
            columns={pcols[0]: "RI_pred"}
        )
        df = df.dropna(subset=["Initial Time", "Valid Time"]).copy()
        df["lead_hours"] = (
            (((df["Valid Time"] - df["Initial Time"]).dt.total_seconds()) / 3600.0)
            .round()
            .astype(int)
        )

        # Align to GT keys; treat missing as negative
        merged = gt.merge(
            df, on=["SID", "Initial Time", "Valid Time", "lead_hours"], how="left"
        )
        pred = (
            merged["RI_pred"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "1": True,
                    "true": True,
                    "t": True,
                    "yes": True,
                    "0": False,
                    "false": False,
                    "f": False,
                    "no": False,
                }
            )
        )
        # Robust boolean casting without silent downcasting warnings:
        pred = pred.astype("boolean").fillna(False).astype(bool)
        merged["RI_pred"] = pred

        y = merged["RI_true"].astype(bool)
        yhat = merged["RI_pred"].astype(bool)
        grp = merged["lead_hours"]
        TP = (y & yhat).groupby(grp).sum().astype(int)
        FP = (~y & yhat).groupby(grp).sum().astype(int)
        FN = (y & ~yhat).groupby(grp).sum().astype(int)
        CSI = TP / (TP + FP + FN).replace(0, np.nan)

        # Map RI filename to pretty label logic used elsewhere (strip _RI), skip unwanted postprocessing variants and TIGGE_IFS
        raw_name = os.path.splitext(fn)[
            0
        ]  # e.g., "2023_PANGU_RI" or "persistence_tracks_RI"
        base_name = raw_name.replace(
            "_RI", ""
        )  # -> "2023_PANGU" or "persistence_tracks"
        low_raw = base_name.lower()
        # Safety filters (should already be applied when building pred_files)
        if "tigge_ifs" in low_raw:
            continue
        if (
            "postprocessing" in low_raw
            and "postprocessing_panguweather_ann" not in low_raw
        ):
            continue
        # Normalize persistence label so it matches the scorecard baseline row
        if "persistence" in low_raw:
            label = "Persistence"
        else:
            # Map to the same pretty label used elsewhere
            label = pretty_curve_label(base_name)

        # If a whitelist of model labels is provided, enforce it (ensures same set as earlier scorecards)
        if allowed_labels is not None and label not in allowed_labels:
            continue

        rows[label] = [float(CSI.get(lh, np.nan)) for lh in lead_list]

    # Ensure we return rows for every model we want to display
    # (even if that model has no *_RI.csv file). Missing models are
    # represented by NaNs so they still appear in the scorecard.
    if allowed_labels is not None:
        for lbl in sorted(allowed_labels):
            if lbl not in rows:
                rows[lbl] = [np.nan] * len(lead_list)

    return rows


tables_abs = {}  # absolute values (for RI: CSI values)
tables_pct = {}  # % difference vs baseline (for RI: % diff vs persistence)

for base, _label in [(m[0], m[1]) for m in METRICS]:
    rows_abs = {}
    if base == "RI_CSI":
        # Use RI-specific lead grid for RI CSI panel
        RI_LEADS = [24, 48, 72, 96, 120]
        # Keep the RI panel consistent with other scorecards *including* the baseline
        _allowed_labels = set(models.keys())  # include "Persistence"
        rows_abs = _ri_scorecard_rows(RI_LEADS, allowed_labels=_allowed_labels)
        if not rows_abs:
            # If no RI available, skip this panel entirely
            continue
        A = pd.DataFrame.from_dict(rows_abs, orient="index", columns=RI_LEADS)
        # Ensure baseline row exists under the name 'Persistence'
        # Try to find any row that looks like persistence
        base_row = None
        for idx in A.index:
            if "persistence" in str(idx).lower():
                base_row = idx
                break
        if base_row is None:
            # No baseline -> only absolute CSI, leave color table neutral
            P = pd.DataFrame(np.nan, index=A.index, columns=A.columns)
        else:
            A = A.rename(index={base_row: "Persistence"})
            baseline = A.loc["Persistence"]
            P = (A.subtract(baseline, axis=1)).divide(baseline, axis=1) * 100.0
    else:
        # Regular deterministic metrics (AE_wind, AE_pressure)
        for model_name, df in models.items():
            col = pick_metric_col(df, base)
            if col is None:
                continue
            s = aggregate_by_lead(df, col)
            rows_abs[model_name] = [s.get(lh, np.nan) for lh in LEADS]
        if not rows_abs:
            continue
        A = pd.DataFrame.from_dict(rows_abs, orient="index", columns=LEADS)
        if BASELINE_NAME not in A.index:
            raise ValueError(
                f"Baseline '{BASELINE_NAME}' not found among rows: {list(A.index)}"
            )
        baseline = A.loc[BASELINE_NAME]
        P = (A.subtract(baseline, axis=1)).divide(baseline, axis=1) * 100.0

    # Reorder rows for display: Persistence first, PANGU then PANGU_POST, others, Google/FNV3 last
    order_rows = _order_rows_fnv3_last(A.index.tolist())
    A = A.loc[order_rows]
    P = P.loc[order_rows]
    tables_abs[base] = A
    tables_pct[base] = P

#
#
# ==== PLOT ====
ncols = len(METRICS)  # no extra RI panel
fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 9), constrained_layout=False)

if ncols == 1:
    axes = [axes]

_HATCH_LW_THIN = 0.15  # very thin hatch lines for proprietary rows
_HATCH_PATTERN = "//"  # diagonal thin hatch
for ax, (base, title) in zip(axes[: len(METRICS)], [(m[0], m[1]) for m in METRICS]):
    A = tables_abs.get(base)
    P = tables_pct.get(base)
    if A is None or P is None:
        ax.set_visible(False)
        continue

    # --- Color mapping:
    # For RI_CSI, color by absolute CSI (baseline values are ~0, so % diff is not informative).
    # For other metrics, color by % difference vs baseline (P).
    if base == "RI_CSI":
        # absolute CSI in [0, max]; use a sequential map
        vals_for_colors = A.values
        cmin = 0.0
        cmax = float(np.nanpercentile(vals_for_colors, 95))
        if not np.isfinite(cmax) or cmax <= 0:
            cmax = 0.1
        norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
        orig_cmap = plt.get_cmap("YlGn")
        truncated_cmap = LinearSegmentedColormap.from_list(
            "YlGn", orig_cmap(np.linspace(0.0, 0.8, 256))
        )
        cmap = truncated_cmap
        # cmap = plt.get_cmap("YlGn")
    else:
        # symmetric around 0 for % differences
        vals_for_colors = P.values
        vmax = float(np.nanpercentile(np.abs(vals_for_colors), 95))
        vmax = max(5, vmax)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        cmap = plt.get_cmap("RdBu_r")

    # Draw heatmap
    rows, cols = A.shape
    x_ticks = np.arange(cols)
    y_ticks = np.arange(rows)

    # Identify Google/FNV3 rows and other special rows
    names_list = A.index.tolist()
    google_rows = [i for i, name in enumerate(names_list) if _is_google(name)]
    genc_rows = [i for i, name in enumerate(names_list) if _is_genc(name)]
    pangu_post_rows = [i for i, name in enumerate(names_list) if _is_pangu_post(name)]
    special_rows = set(google_rows + genc_rows + pangu_post_rows)
    # Subtle hatch for special rows
    _rc_hatch_prev_lw = mpl.rcParams.get("hatch.linewidth", 1.0)
    _rc_hatch_prev_col = mpl.rcParams.get("hatch.color", "black")
    mpl.rcParams["hatch.linewidth"] = 0.15
    mpl.rcParams["hatch.color"] = "black"
    # Draw subtle background for special rows (proprietary)
    for gi in special_rows:
        ax.add_patch(
            plt.Rectangle(
                (0, gi),
                cols,
                1,
                facecolor=(0.92, 0.92, 0.92, 1.0),
                edgecolor="none",
                zorder=0,
            )
        )
    # Index helpers for coloring and hatching
    idx_persist = None
    if "Persistence" in A.index:
        idx_persist = list(A.index).index("Persistence")

    for i in range(rows):
        for j in range(cols):
            # Choose value used for coloring
            val_color = vals_for_colors[i, j]
            if np.isnan(val_color):
                color = (0.9, 0.9, 0.9, 1.0)
            else:
                col = cmap(norm(val_color))
                # Slight transparency for DPE panel to ease hatch/text readability
                if base == "DPE_GCD":
                    color = (col[0], col[1], col[2], 0.92)
                else:
                    color = col

            # Make Persistence row visually neutral (light gray) on RI_CSI panel only
            if base == "RI_CSI" and idx_persist is not None and i == idx_persist:
                color = (0.92, 0.92, 0.92, 1.0)

            is_special_row = i in special_rows
            ax.add_patch(
                plt.Rectangle(
                    (j, i),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=1.0,
                    hatch=(_HATCH_PATTERN if is_special_row else None),
                    zorder=1,
                )
            )
    mpl.rcParams["hatch.linewidth"] = _rc_hatch_prev_lw
    mpl.rcParams["hatch.color"] = _rc_hatch_prev_col
    # Write text annotations after rectangles so text is on top
    # Annotation precision: force 3 decimals for RI_CSI (CSI is small), otherwise use ROUND_TO
    r = 3 if base == "RI_CSI" else ROUND_TO.get(base, 1)
    for i in range(rows):
        for j in range(cols):
            val_abs = A.iloc[i, j]
            if np.isnan(val_abs):
                txt = "–"
                color = "0.4"
            else:
                if r == 0:
                    txt = f"{int(round(val_abs))}"
                else:
                    txt = f"{val_abs:.{r}f}"
                # Improve contrast on special rows/panels
                if i in google_rows and base in ("DPE_GCD", "CRPS_haversine"):
                    color = "white"
                else:
                    color = "black"
            ax.text(
                j + 0.5,
                i + 0.5,
                txt,
                ha="center",
                va="center",
                fontsize=12,
                color=color,
            )

    # Axes cosmetics
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_xticks(x_ticks + 0.5)
    # Set x-axis tick labels to match each panel's own columns (not global LEADS)
    ax.set_xticklabels([str(int(lh)) for lh in A.columns.tolist()], rotation=0)
    ax.set_yticks(y_ticks + 0.5)
    # Set yticklabels, replacing FNV3 and GENC rows with LaTeX italic label
    ylabels = []
    for name in A.index.tolist():
        if _is_google(name):
            ylabels.append("$\\it{FNV3}$")
        elif _is_genc(name):
            ylabels.append("$\\it{GENC}$")
        else:
            ylabels.append(name)
    ax.set_yticklabels(ylabels)
    ax.set_title(title, fontsize=16)
    # ax.set_xlabel("Lead time (hours)")
    if ax is axes[0]:
        ax.set_ylabel("Models", fontsize=16)

    # Colorbar
    # Build a mini colorbar under each panel
    cax = ax.inset_axes([0.05, -0.08, 0.90, 0.03])
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal"
    )
    if base == "RI_CSI":
        cb.set_label("CSI (higher is better)")
        cb.ax.tick_params(labelsize=11)
    else:
        cb.set_label("% Difference vs Baseline (Lower is Better)")
        cb.ax.tick_params(labelsize=11)

# Figure title and notes
# fig.suptitle(
#   f"TCBench Scorecard — Deterministic, Test Year {TEST_YEAR} — Baseline: Persistence",
#  fontsize=16,
# y=0.98,
# )
"""
fig.text(
    0.5,
    0.01,
    "Cells show mean error; colors show % difference vs baseline at the same lead.",
    ha="center",
    fontsize=10,
)
"""
# Deterministic scorecard title (set on this figure handle)
fig.suptitle(
    "TCBench Scorecard — Deterministic (AE & RI CSI), Test Year 2023 — Baseline: Persistence [BASE]",
    y=0.98,
    fontsize=14,
)
fig.tight_layout(rect=[0, 0.04, 1, 0.95])
fig.savefig("TCBench_scorecard_deterministic.pdf", format="pdf", bbox_inches="tight")
plt.show()

# %%
# ==== CONFIG ====

GT_FILE = "ibtracs_RI_gt.csv"  # ground truth file saved by you
TREAT_MISSING_AS_NEGATIVE = (
    True  # if a model lacks a (SID, t0, vt) prediction, count as 'no-RI'
)

# ==== HELPERS ====
KEYS = ["SID", "Initial Time", "Valid Time"]


def _to_dt(df, cols=("Initial Time", "Valid Time")):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _to_bool(series: pd.Series):
    """Robustly coerce a series of RI flags to booleans, preserving NaN."""
    if series.dtype == bool:
        return series
    s = series.astype(str).str.strip().str.lower()
    mapdict = {
        "1": True,
        "true": True,
        "t": True,
        "yes": True,
        "y": True,
        "0": False,
        "false": False,
        "f": False,
        "no": False,
        "n": False,
    }
    out = s.map(mapdict)
    # Keep original NaN as NaN
    out[series.isna()] = np.nan
    return out


def _pretty_name(fn_no_ext: str):
    """Tidy filename into a short model label and map post-processing variants.
    - postprocessing_panguweather_ann*  -> PANGU_POST_ANN
    - postprocessing_panguweather_mlr*  -> PANGU_POST_MLR
    - postprocessing_panguweather_unet* -> PANGU_POST_UNET
    - legacy 'PANGU_POST'               -> PANGU_POST_ANN
    """
    name = fn_no_ext
    for tok in ["_RI", "_results", "_clean", "_corrected", "(downloaded)"]:
        name = name.replace(tok, "")
    name = name.replace("2023_", "")
    low = name.lower()
    if "postprocessing_panguweather_ann" in low:
        return "PANGU_POST_ANN"
    if "postprocessing_panguweather_mlr" in low:
        return "PANGU_POST_MLR"
    if "postprocessing_panguweather_unet" in low:
        return "PANGU_POST_UNET"
    if name.strip().upper() == "PANGU_POST":
        return "PANGU_POST_ANN"
    return name


def _load_gt(path):
    gt = pd.read_csv(path, low_memory=False)
    gt = _to_dt(gt)
    # pick RI true column name
    ri_cols = [c for c in gt.columns if c.lower() in {"ri", "ri_true"}]
    if not ri_cols:
        raise ValueError(
            f"No RI column found in {path} (looked for 'RI' or 'RI_true')."
        )
    gt = gt[KEYS + [ri_cols[0]]].rename(columns={ri_cols[0]: "RI_true"})
    # Coerce to pandas nullable boolean first, then fill. This avoids the
    # FutureWarning about downcasting object dtype on fillna.
    _ri_true = _to_bool(gt["RI_true"]).astype("boolean")
    gt["RI_true"] = _ri_true.fillna(False)
    return gt


def _load_model_ri(path):
    df = pd.read_csv(path, low_memory=False)
    df = _to_dt(df)
    # predicted RI column
    ri_cols = [
        c
        for c in df.columns
        if c.lower() in {"ri", "ri_pred", "ri_hat", "ri_prediction"}
    ]
    if not ri_cols:
        raise ValueError(f"No RI prediction column in {path}")
    return df[KEYS + [ri_cols[0]]].rename(columns={ri_cols[0]: "RI_pred"})


def _confusion_and_scores(gt_df: pd.DataFrame, pred_df: pd.DataFrame):
    # align to ground truth keys, keep all GT rows
    merged = gt_df.merge(pred_df, on=KEYS, how="left")
    # Coerce preds to pandas nullable boolean to avoid FutureWarning on fillna
    pred = _to_bool(merged["RI_pred"]).astype("boolean")
    coverage = pred.notna().mean()  # fraction of GT keys with a prediction
    if TREAT_MISSING_AS_NEGATIVE:
        pred = pred.fillna(False)
    else:
        # keep only rows where a prediction exists
        keep = pred.notna()
        merged = merged[keep].copy()
    merged["RI_pred"] = pred

    y = merged["RI_true"].astype(bool).values
    yhat = merged["RI_pred"].astype(bool).values

    TP = int(np.sum((y == True) & (yhat == True)))
    FP = int(np.sum((y == False) & (yhat == True)))
    FN = int(np.sum((y == True) & (yhat == False)))
    TN = int(np.sum((y == False) & (yhat == False)))
    N = int(len(merged))
    RI_pos = int(np.sum(y))

    # CSI = TP / (TP+FP+FN)
    denom_csi = TP + FP + FN
    CSI = TP / denom_csi if denom_csi > 0 else np.nan

    # Peirce Skill Score (Hanssen–Kuipers): TPR - FPR
    TPR = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    FPR = FP / (FP + TN) if (FP + TN) > 0 else np.nan
    PSS = (TPR - FPR) if (not np.isnan(TPR) and not np.isnan(FPR)) else np.nan

    return {
        "CSI": CSI,
        "PSS": PSS,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "N": N,
        "RI_pos": RI_pos,
        "coverage": coverage,
    }


# ==== LOAD DATA ====
gt_path = os.path.join(EVAL_DIR, GT_FILE)
gt = _load_gt(gt_path)

model_files = [
    f
    for f in os.listdir(EVAL_DIR)
    if f.endswith("_RI.csv")
    and f != GT_FILE
    and "ibtracs_ri_gt" not in f.lower()
    and "tigge_ifs" not in f.lower()
]

if not model_files:
    raise FileNotFoundError("No *_RI.csv model files found.")

rows = []
for fn in sorted(model_files):
    try:
        pred = _load_model_ri(os.path.join(EVAL_DIR, fn))
        scores = _confusion_and_scores(gt, pred)
        raw_label = _pretty_name(os.path.splitext(fn)[0])
        label = (
            pretty_curve_label(raw_label)
            if "pretty_curve_label" in globals()
            else raw_label
        )
        rows.append({"model": label, **scores})
    except Exception as e:
        print(f"⚠️ Skipping {fn}: {e}")

scores_df = pd.DataFrame(rows)
if scores_df.empty:
    raise RuntimeError("No valid model RI files after loading.")

# ==== RI SKILL BY LEAD TIME (CSI & PSS curves, no coverage panel) ====


def _lead_hours(df):
    """Compute integer lead time in hours from Initial/Valid Time."""
    return (
        ((df["Valid Time"] - df["Initial Time"]).dt.total_seconds() / 3600.0)
        .round()
        .astype("Int64")
    )


def _per_lead_scores(
    gt_df: pd.DataFrame, pred_df: pd.DataFrame, allowed_leads=None
) -> pd.DataFrame:
    """
    Align prediction to GT keys, coerce RI to booleans, optionally treat missing as negative,
    and compute TP/FP/FN/TN, CSI, PSS per lead_hours.
    Returns DataFrame indexed by lead_hours with columns [CSI, PSS, N, RI_pos].
    """
    merged = gt_df.merge(pred_df, on=KEYS, how="left")

    # Make it explicitly nullable boolean, then fill/drop per flag
    pred = _to_bool(merged["RI_pred"]).astype("boolean")
    if TREAT_MISSING_AS_NEGATIVE:
        pred = pred.fillna(False)
    merged["RI_pred"] = pred
    if not TREAT_MISSING_AS_NEGATIVE:
        merged = merged.dropna(subset=["RI_pred"])

    merged = merged.dropna(subset=["Initial Time", "Valid Time"]).copy()
    merged["lead_hours"] = _lead_hours(merged)

    # Optionally restrict to a subset of leads (e.g., for post-processing models)
    if allowed_leads is not None:
        # ensure numeric comparison
        try:
            allowed_leads = set(int(l) for l in allowed_leads)
        except Exception:
            allowed_leads = set(allowed_leads)
        merged = merged[merged["lead_hours"].isin(list(allowed_leads))].copy()

    m = merged["RI_pred"].astype(bool)
    t = merged["RI_true"].astype(bool)
    grp = merged["lead_hours"]

    TP = (m & t).groupby(grp).sum().astype(int)
    FP = (m & ~t).groupby(grp).sum().astype(int)
    FN = (~m & t).groupby(grp).sum().astype(int)
    TN = (~m & ~t).groupby(grp).sum().astype(int)

    N = (
        grp.groupby(grp).size().astype(int)
    )  # same as merged.groupby("lead_hours").size()
    RI_pos = t.groupby(grp).sum().astype(int)

    CSI = TP / (TP + FP + FN).replace(0, np.nan)
    TPR = TP / (TP + FN).replace(0, np.nan)
    FPR = FP / (FP + TN).replace(0, np.nan)
    PSS = TPR - FPR

    out = pd.DataFrame({"CSI": CSI, "PSS": PSS, "N": N, "RI_pos": RI_pos}).sort_index()
    out.index.name = "lead_hours"
    return out


def _is_google_label(lbl: str) -> bool:
    return (
        _is_google(lbl)
        or ("fnv3" in str(lbl).lower())
        or ("weatherlab" in str(lbl).lower())
    )


# Build per-model curves
ALLOWED_PP_LEADS = (6, 12, 18, 24, 48, 72, 96, 120)
curves = {}  # model -> per-lead DataFrame
for fn in sorted(model_files):
    try:
        pred = _load_model_ri(os.path.join(EVAL_DIR, fn))
        label = _pretty_name(os.path.splitext(fn)[0])
        # If this is a post-processing model, only evaluate at the specified leads
        allowed = ALLOWED_PP_LEADS if ("postprocessing" in fn.lower()) else None
        curves[label] = _per_lead_scores(gt, pred, allowed_leads=allowed)
    except Exception as e:
        print(f"⚠️ Skipping {fn}: {e}")


# ===================== FIGURE A — OVERALL SKILL BY MODEL (BARS) =====================
# Uses `scores_df` built above
def _is_google_label(lbl: str) -> bool:
    return (
        _is_google(lbl)
        or ("fnv3" in str(lbl).lower())
        or ("weatherlab" in str(lbl).lower())
    )


# Order by CSI and set consistent colors
disp = scores_df.sort_values("CSI", ascending=False).reset_index(drop=True)
labels_bar = disp["model"].tolist()
ypos = np.arange(len(disp))

cmap = mpl.colormaps.get("tab10")
bar_colors = []
for m in labels_bar:
    ml = str(m).lower()
    if "persistence" in ml:
        bar_colors.append("black")
    elif _is_google(m) or "fnv3" in ml:
        bar_colors.append(color_for("$\\it{FNV3}$"))
    else:
        bar_colors.append(color_for(m))  # cmap(labels_bar.index(m) % 10))

figA, (ax_csi_bar, ax_pss_bar) = plt.subplots(
    1, 2, figsize=(12, max(4.8, 0.6 * len(disp))), sharey=True
)

# CSI bars
bars_csi = ax_csi_bar.barh(ypos, disp["CSI"].values, color=bar_colors, edgecolor="none")
ax_csi_bar.set_yticks(ypos)
ax_csi_bar.set_yticklabels([pretty_curve_label(m) for m in labels_bar], fontsize=10)
ax_csi_bar.invert_yaxis()
ax_csi_bar.set_xlabel("Critical Success Index (CSI)")
ax_csi_bar.set_title("RI — CSI (Overall)")
xmax_csi = (
    float(np.nanmax(disp["CSI"].values))
    if np.isfinite(np.nanmax(disp["CSI"].values))
    else 0.0
)
ax_csi_bar.set_xlim(0, max(0.05, xmax_csi * 1.15))
for y, val in enumerate(disp["CSI"].values):
    if np.isfinite(val):
        ax_csi_bar.text(
            val + ax_csi_bar.get_xlim()[1] * 0.01,
            y,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=9,
        )
ax_csi_bar.grid(True, axis="x", alpha=0.25, linestyle="--", linewidth=0.8)

# Hatch Google/FNV3/GENC/PANGU_POST bars (in‑plot) to match legend
_hatch_pat = globals().get("_HATCH_PATTERN", "/")
for i, rect in enumerate(bars_csi.patches):
    m = labels_bar[i]
    if _is_google(m) or ("fnv3" in str(m).lower()) or _is_genc(m) or _is_pangu_post(m):
        rect.set_hatch(_hatch_pat)
        rect.set_edgecolor("black")
        rect.set_linewidth(0.8)

# PSS bars
ax_pss_bar.axvline(0.0, color="0.35", lw=1)
bars_pss = ax_pss_bar.barh(ypos, disp["PSS"].values, color=bar_colors, edgecolor="none")
ax_pss_bar.set_yticks(ypos)
ax_pss_bar.set_yticklabels([])  # names on CSI panel
ax_pss_bar.invert_yaxis()
ax_pss_bar.set_xlabel("Peirce Skill Score (PSS = TPR − FPR)")
ax_pss_bar.set_title("RI — PSS (Overall)")
vmax = (
    float(np.nanmax(np.abs(disp["PSS"].values)))
    if np.isfinite(np.nanmax(np.abs(disp["PSS"].values)))
    else 0.1
)
ax_pss_bar.set_xlim(-0.05, vmax * 1.2)
for y, val in enumerate(disp["PSS"].values):
    if np.isfinite(val):
        ha = "left" if val >= 0 else "right"
        pad = ax_pss_bar.get_xlim()[1] * 0.01 * (1 if val >= 0 else -1)
        ax_pss_bar.text(val + pad, y, f"{val:+.3f}", va="center", ha=ha, fontsize=9)
# Hatch Google/FNV3/GENC/PANGU_POST bars (in‑plot) to match legend
for i, rect in enumerate(bars_pss.patches):
    m = labels_bar[i]
    if _is_google(m) or ("fnv3" in str(m).lower()) or _is_genc(m) or _is_pangu_post(m):
        rect.set_hatch(_hatch_pat)
        rect.set_edgecolor("black")
        rect.set_linewidth(0.8)

# --- Build legend directly from plotted bars so styles (color/hatch) match exactly
legend_handles = []
legend_labels = []
for i, rect in enumerate(bars_csi.patches):
    m = labels_bar[i]
    pretty = pretty_curve_label(m)
    # Create a proxy patch with the same facecolor and hatch as the bar
    fc = rect.get_facecolor()
    ec = rect.get_edgecolor()
    hatch = getattr(rect, "get_hatch", lambda: None)()
    proxy = mpl.patches.Patch(facecolor=fc, edgecolor=ec, hatch=hatch, label=pretty)
    legend_handles.append(proxy)
    legend_labels.append(pretty)

# Reorder legend so that Persistence comes first
_pairs = list(zip(legend_handles, legend_labels))
_persist = [(h, l) for (h, l) in _pairs if "persistence" in str(l).lower()]
_others = [(h, l) for (h, l) in _pairs if "persistence" not in str(l).lower()]
_ordered = (_persist + _others) if (_persist or _others) else _pairs
legend_handles, legend_labels = (
    zip(*_ordered) if _ordered else (legend_handles, legend_labels)
)

figA.legend(
    legend_handles,
    legend_labels,
    loc="upper center",
    ncols=min(4, len(legend_labels)),
    frameon=False,
    bbox_to_anchor=(0.5, 0.01),
    fontsize=9,
)

# figA.suptitle("Rapid Intensification — Overall Skill by Model", fontsize=14, y=0.98)
figA.tight_layout(rect=[0, 0.08, 1, 0.88])
figA.savefig("TCBench_RI_skill_overall.pdf", format="pdf", bbox_inches="tight")
plt.show()


# ===================== FIGURE B — SKILL BY LEAD TIME (CURVES, WITH CAPS) =====================
# Uses `curves` dict that we already built above

if not curves:
    raise RuntimeError("No per-lead RI curves were computed.")

# Consistent colors per model
labels_curve = list(curves.keys())
cmap = mpl.colormaps.get("tab10")
color_map_curve = {lab: cmap(i % 10) for i, lab in enumerate(labels_curve)}

figB, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
ax_csi, ax_pss = axes

# CSI by lead (cap at 0.2)
for lab in labels_curve:
    df = curves[lab]
    x = df.index.values.astype(float)
    y = df["CSI"].values
    pretty = pretty_curve_label(lab)
    plot_kwargs = dict(
        marker="o",
        linewidth=1.6,
        markersize=3.5,
        label=pretty,
        color=color_for(lab),
        # color=color_map_curve.get(lab, None),
    )
    # Force Persistence style (black dashed, thicker line)
    if "persistence" in str(lab).lower():
        plot_kwargs.update(color="black", linestyle="--", linewidth=3.0)
    if _is_google_label(lab):
        plot_kwargs.update(
            dict(color="#7a7a7a", linestyle=":", marker="s", linewidth=2.0)
        )
    elif _is_genc(lab) or _is_pangu_post(lab):
        plot_kwargs.update(dict(linestyle=":", marker="s", linewidth=2.0))
    ax_csi.plot(x, y, **plot_kwargs)
ax_csi.set_title("RI — Critical Success Index (CSI) by lead")
ax_csi.set_xlabel("Lead time (hours)")
ax_csi.set_ylabel("CSI")
ax_csi.set_xlim(6, 120)
ax_csi.xaxis.set_major_locator(mticker.MultipleLocator(24))
ax_csi.xaxis.set_minor_locator(mticker.MultipleLocator(6))
ax_csi.set_ylim(0, 0.15)  # cap at 0.15 as requested
ax_csi.grid(True, alpha=0.3)


# PSS by lead (cap at ±0.2)
for lab in labels_curve:
    df = curves[lab]
    x = df.index.values.astype(float)
    y = df["PSS"].values
    pretty = pretty_curve_label(lab)
    plot_kwargs = dict(
        marker="o",
        linewidth=1.6,
        markersize=3.5,
        label=pretty,
        color=color_for(lab),
        # color=color_map_curve.get(lab, None),
    )
    # Force Persistence style (black dashed, thicker line)
    if "persistence" in str(lab).lower():
        plot_kwargs.update(color="black", linestyle="--", linewidth=3.0)
    if _is_google_label(lab):
        plot_kwargs.update(
            dict(color="#7a7a7a", linestyle=":", marker="s", linewidth=2.0)
        )
    elif _is_genc(lab) or _is_pangu_post(lab):
        plot_kwargs.update(dict(linestyle=":", marker="s", linewidth=2.0))
    ax_pss.plot(x, y, **plot_kwargs)
ax_pss.axhline(0.0, color="0.35", lw=1)
ax_pss.set_title("RI — Peirce Skill Score (PSS) by lead")
ax_pss.set_xlabel("Lead time (hours)")
ax_pss.set_ylabel("PSS = TPR − FPR")
ax_pss.set_xlim(6, 120)
ax_pss.xaxis.set_major_locator(mticker.MultipleLocator(24))
ax_pss.xaxis.set_minor_locator(mticker.MultipleLocator(6))
ax_pss.set_ylim(-0.15, 0.2)
ax_pss.grid(True, alpha=0.3)

# Put the legend below the plots, using the pretty labels already set above
handles, labels_ = ax_csi.get_legend_handles_labels()
# Reorder legend so that Persistence comes first
_pairs = list(zip(handles, labels_))
_persist = [(h, l) for (h, l) in _pairs if "persistence" in str(l).lower()]
_others = [(h, l) for (h, l) in _pairs if "persistence" not in str(l).lower()]
_ordered = (_persist + _others) if (_persist or _others) else _pairs
handles_ord, labels_ord = zip(*_ordered) if _ordered else (handles, labels_)

figB.legend(
    handles_ord,
    labels_ord,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.01),
    ncols=min(4, len(labels_curve)),
    frameon=False,
    fontsize=9,
)
# Make room at the bottom for the legend
figB.subplots_adjust(bottom=0.02)
"""
figB.suptitle(
    "Rapid Intensification skill by lead time (vs ibtracs_RI_gt.csv)",
    fontsize=14,
    y=0.99,
)
"""
figB.tight_layout(rect=[0, 0.08, 1, 0.88])
figB.savefig("TCBench_RI_skill_by_lead.pdf", format="pdf", bbox_inches="tight")
plt.show()


def plot_ri_overall(ax, results_dict, pretty_curve_label, color_for, _legend_sort_key):
    """
    Plot RI CSI curves for all models on a single axes.
    Ensure:
      - persistence is drawn in black dashed and labeled 'Persistence [BASE]'
      - persistence is the FIRST entry in the legend
    """
    plotted = []  # list of tuples: (handle, label, is_persist)

    for model_name, series in results_dict.items():
        name_str = str(model_name)
        is_persist = "persistence" in name_str.lower()
        display_label = (
            "Persistence [BASE]" if is_persist else pretty_curve_label(name_str)
        )

        plot_kwargs = dict(marker="o", linewidth=1.8)
        if is_persist:
            plot_kwargs.update(dict(color="black", linestyle="--", linewidth=3.0))
        else:
            plot_kwargs.setdefault("color", color_for(display_label))

        (h_line,) = ax.plot(
            series.index, series.values, label=display_label, **plot_kwargs
        )
        plotted.append((h_line, display_label, is_persist))

    # ---- Legend: put persistence first, then all others sorted by the provided key
    if plotted:
        persist_items = [(h, lbl) for (h, lbl, isp) in plotted if isp]
        other_items = [(h, lbl) for (h, lbl, isp) in plotted if not isp]

        # Sort "others" by legend sort key if provided
        if other_items:
            other_items.sort(
                key=lambda hl: _legend_sort_key(hl[1]) if _legend_sort_key else hl[1]
            )

        ordered = persist_items + other_items

        # Deduplicate labels preserving first occurrence (defensive)
        seen = set()
        ordered_unique = []
        for h, lbl in ordered:
            if lbl in seen:
                continue
            seen.add(lbl)
            ordered_unique.append((h, lbl))

        handles_sorted, labels_sorted = zip(*ordered_unique)
        ax.legend(
            handles_sorted, labels_sorted, ncols=2, frameon=False, loc="upper left"
        )
    else:
        ax.legend(frameon=False, loc="upper left")


def plot_ri_by_lead(ax, results_dict, pretty_curve_label, color_for, _legend_sort_key):
    """
    Plot RI CSI for each lead (lines with markers), ensuring:
      - persistence is black dashed and labeled 'Persistence [BASE]'
      - persistence appears FIRST in the legend
    """
    plotted = []  # list of tuples: (handle, label, is_persist)

    for model_name, series in results_dict.items():
        name_str = str(model_name)
        is_persist = "persistence" in name_str.lower()
        display_label = (
            "Persistence [BASE]" if is_persist else pretty_curve_label(name_str)
        )

        plot_kwargs = dict(marker="o", linewidth=1.8)
        if is_persist:
            plot_kwargs.update(dict(color="black", linestyle="--", linewidth=3.0))
        else:
            plot_kwargs.setdefault("color", color_for(display_label))

        (h_line,) = ax.plot(
            series.index, series.values, label=display_label, **plot_kwargs
        )
        plotted.append((h_line, display_label, is_persist))

    # ---- Legend: persistence first, then others sorted
    if plotted:
        persist_items = [(h, lbl) for (h, lbl, isp) in plotted if isp]
        other_items = [(h, lbl) for (h, lbl, isp) in plotted if not isp]

        if other_items:
            other_items.sort(
                key=lambda hl: _legend_sort_key(hl[1]) if _legend_sort_key else hl[1]
            )

        ordered = persist_items + other_items

        seen = set()
        ordered_unique = []
        for h, lbl in ordered:
            if lbl in seen:
                continue
            seen.add(lbl)
            ordered_unique.append((h, lbl))

        handles_sorted, labels_sorted = zip(*ordered_unique)
        ax.legend(
            handles_sorted, labels_sorted, ncols=2, frameon=False, loc="upper left"
        )
    else:
        ax.legend(frameon=False, loc="upper left")


# %%
