"""Summary tables and figures of the TCBench evaluation (revision).

Settings (run year, folders, lead times, model names and styles, colors) are
read from the YAML file at CONFIG_PATH.

Pipeline (main block):
  1. Inputs, in result_folder: the per-case results written by
     evaluate_tracks_rev, one file per model and kind, named
     '{model}_{det|prob|ri}_{year}[_{basin}].csv', and the IBTrACS ground
     truth ('ibtracs_gt...{year}...csv'). With a basin set, every table is
     restricted to the forecast cases (sid, initial_time) whose storm is in
     that basin at the initial time (_basin_cases), and output names are
     tagged '{year}_{basin}'.
  2. Per-case metrics (det_columns, prob_columns) are summarized per lead time
     over each model's own cases (not a homogeneous sample), plus a 'mean' row
     pooling all lead times (_summarize_metric). Ensemble members are first
     averaged per case. The statistic depends on the metric (_aggregate):
       SquErr_*                -> RMSE (root of the mean squared error)
       AlongTrack, CrossTrack  -> mean absolute value (the errors are signed)
       others                  -> mean (AbsErr_* gives the MAE)
  3. Aggregate metrics score a set of cases at once, per lead time and pooled:
     the RI critical success index (CSI), Peirce skill score (PSS), Brier
     skill score (BSS, against the mean tendency climatology) and ROC AUC, and
     the R^2 of the intensity forecasts.
  4. Coverage (_coverage_flags): each valid time of a case that is observed or
     predicted is exactly one of
       hit                (observed and predicted; 'comparable' in the tables)
       miss               (observed; the model predicts no valid time of the case)
       early dissipation  (observed, not predicted; other valid times are)
       late dissipation   (predicted, not observed)
     The hit fraction is hits / all four categories.
  5. Outputs, in plotting_folder: summary_{metric}_{tag}.csv,
     coverage_{tag}.csv and detail_{metric}_{tag}.csv tables, and figures:
       deterministic_summary, probabilistic_summary: scores by lead time
         (lines) and pooled (bars), with the coverage counts
       deterministic_scorecard, probabilistic_scorecard: score per model and
         lead time, shaded relative to persistence, with the hit fraction
       ri_summary (off by default): RI CSI and PSS

Terms: 'hit' and 'hit fraction' always refer to the coverage categories above.
The probability circle score ProbCircle_hit70 is the frequency at which the
observed position falls inside the ensemble's 70% circle; its per-case 'hit'
flags come from metrics_rev and are unrelated to coverage hits.
"""

# %% imports
import sys
import os
import re
import shutil
from itertools import cycle
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.collections import PolyCollection

_DEV_FOLDER = "/home/mgomezd1/repos/iop_tcbench/dev"
if _DEV_FOLDER not in sys.path:
    sys.path.insert(0, _DEV_FOLDER)
os.chdir(_DEV_FOLDER)

import metrics_rev as metrics
from utils import toolbox

# %% Constants
# Plotting configuration (model names and styles, metric columns, colors),
# editable in a YAML file
CONFIG_PATH = os.path.join(_DEV_FOLDER, "plot_errors_config.yaml")

with open(CONFIG_PATH) as f:
    _config = yaml.safe_load(f)

pretty_names = _config["pretty_names"]
# Line style per model (raw names) in the lead-time panels; others are solid
MODEL_LINESTYLES = _config.get("model_linestyles") or {}
MODEL_HATCH = _config["model_hatch"]
SCORECARD_HATCH = _config["scorecard_hatch"]
# Models (raw names) excluded from all summaries, coverage and plots
IGNORED_MODELS = _config.get("ignored_models") or {}

det_columns = _config["det_columns"]
prob_columns = _config["prob_columns"]
summary_columns = [*det_columns, *prob_columns]

MAX_DISTINCT = _config["max_distinct"]
PERSIST_COLOR = _config["persist_color"]
CLIM_COLOR = _config["clim_color"]

SCORECARD_BETTER = _config["scorecard_better"]
SCORECARD_WORSE = _config["scorecard_worse"]
SCORECARD_COVERAGE = _config["scorecard_coverage"]
SCORECARD_MISSING = _config["scorecard_missing"]
SCORECARD_SCORE_LOW = _config["scorecard_score_low"]
SCORECARD_SCORE_HIGH = _config["scorecard_score_high"]
SCORECARD_UNDERDISPERSIVE = _config["scorecard_underdispersive"]
SCORECARD_OVERDISPERSIVE = _config["scorecard_overdispersive"]

# Variable whose coverage applies to each metric
METRIC_COVERAGE_VARIABLE = _config["metric_coverage_variable"]

# Columns identifying a forecast case (one storm, one initialization) and one
# valid time of that case
CASE_KEYS = ["sid", "initial_time"]
KEYS = [*CASE_KEYS, "valid_time"]

# IBTrACS basin codes accepted by the 'basin' setting
BASINS = ["NA", "EP", "WP", "NI", "SI", "SP", "SA"]

# Millimeters to inches, for figure sizes
MM = 1 / 25.4


# %% helper functions
def _model_colors(model_names) -> dict:
    """Color lookup for plotting, keyed by both raw and pretty model names.

    The persistence and climatology baselines get PERSIST_COLOR and CLIM_COLOR;
    the other models are assigned MAX_DISTINCT colors in the given order,
    cycling if there are more models than colors.
    """
    colors = {}
    palette = cycle(MAX_DISTINCT)
    for name in model_names:
        if name.startswith("PERSIST"):
            color = PERSIST_COLOR
        elif name.startswith("CLIM"):
            color = CLIM_COLOR
        else:
            color = next(palette)
        colors[name] = color
        colors[pretty_names.get(name, name)] = color
    return colors


def _model_hatch(model) -> str:
    """Bar hatch of a raw model name: MODEL_HATCH for the models with a
    MODEL_LINESTYLES entry, none otherwise."""
    return MODEL_HATCH if model in MODEL_LINESTYLES else ""


def _baseline(model_names, prefix):
    """First model name starting with prefix ('PERSIST' or 'CLIM'), or None."""
    return next((m for m in model_names if m.startswith(prefix)), None)


def _lead_hours(df: pd.DataFrame) -> pd.Series:
    """Lead time in hours (valid_time - initial_time) of each row."""
    return (df["valid_time"] - df["initial_time"]).dt.total_seconds() / 3600


def _filter_files(file_list, year, suffixes):
    """Filter a list of files to only include those that match the specified year and suffixes."""
    filtered_files = []
    for file in file_list:
        if (
            any(suffix in file for suffix in suffixes)
            and str(year) in file
            and file.endswith(".csv")
        ):
            filtered_files.append(file)
    return filtered_files


def _model_names(file_list, year=None, kinds=("ri", "det", "prob")):
    """Unique model names from results files named
    '{model}_{kind}_{year}[_{basin}].csv', as written by evaluate_tracks.

    Files that do not match the pattern (or the given year) are ignored.
    Returns the names sorted alphabetically.
    """
    year_re = str(year) if year is not None else r"\d{4}"
    pattern = re.compile(
        rf"^(?P<model>.+)_(?:{'|'.join(kinds)})_{year_re}(?:_[A-Za-z]{{2}})?\.csv$"
    )
    names = set()
    for file in file_list:
        match = pattern.match(os.path.basename(file))
        if match:
            names.add(match.group("model"))
    return sorted(names)


def _model_name(file, year=None):
    """Model name of a single results file, or None if it does not match."""
    names = _model_names([file], year)
    return names[0] if names else None


def _basin_cases(ibtracs_tracks_path: str, year: int, basin: str) -> pd.MultiIndex:
    """Forecast cases (sid, initial_time) of the given basin and year: the
    IBTrACS records in that basin, read with the same loader as
    evaluate_tracks_rev. A case belongs to the basin the storm is in at the
    initial time, so a storm that crosses basins contributes its forecasts to
    each basin from the time it is in it."""
    ib = toolbox.read_hist_track_file(tracks_path=ibtracs_tracks_path)
    in_basin = (ib["BASIN"] == basin) & (ib["ISO_TIME"].dt.year == int(year))
    cases = ib.loc[in_basin, ["SID", "ISO_TIME"]].drop_duplicates()
    return pd.MultiIndex.from_arrays(
        [cases["SID"].astype(str), cases["ISO_TIME"]], names=CASE_KEYS
    )


def _in_cases(df: pd.DataFrame, cases: pd.MultiIndex) -> pd.DataFrame:
    """Rows of df whose (sid, initial_time) is one of the cases."""
    return df[pd.MultiIndex.from_frame(df[CASE_KEYS]).isin(cases)]


def _read_results(path):
    """Read a results / ground-truth CSV with string sids and datetime keys."""
    df = pd.read_csv(path, dtype={"sid": str})
    for col in ["initial_time", "valid_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    if "lead_hours" in df.columns:
        df["lead_hours"] = df["lead_hours"].astype(int)
    return df


def _aggregate(values: pd.Series, metric: str) -> float:
    """Summary statistic of one metric over a set of cases: RMSE for squared
    errors, mean absolute value for the signed along/cross-track errors, and the
    mean otherwise."""
    v = values.dropna()
    if v.empty:
        return np.nan
    if metric.startswith("SquErr"):
        return np.sqrt(v.mean())
    if metric in ("AlongTrack", "CrossTrack"):
        return v.abs().mean()
    return v.mean()


def _summarize_metric(df: pd.DataFrame, metric: str, lead_hours) -> pd.Series:
    """Per-lead summary of one metric over the model's own cases, plus a 'mean'
    entry pooling all cases in lead_hours.

    Ensemble members are first averaged per case (mean member error), so each
    case counts once.
    """
    if "ensemble_idx" in df.columns:
        df = df.groupby([*KEYS, "lead_hours"], as_index=False)[metric].mean()
    df = df[df["lead_hours"].isin(lead_hours)]
    summary = (
        df.groupby("lead_hours")[metric]
        .agg(lambda v: _aggregate(v, metric))
        .reindex(lead_hours)
    )
    summary.loc["mean"] = _aggregate(df[metric], metric)
    return summary


def _coverage_flags(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    variable: str = "wind_max",
    lead_hours=None,
    ens_fraction: float = 0.3,
) -> pd.DataFrame:
    """Classify forecast coverage per (sid, initial_time, valid_time).

    A valid time is observed when the ground truth has the variable, and
    predicted when the forecast has it. variable is 'position' (lat and lon) or
    a column name such as 'wind_max'. For ensembles (ensemble_idx column), the
    TC is predicted at a valid time when at least ens_fraction of the ensemble
    members have it; the denominator is the number of unique ensemble indices.

    Only (sid, initial_time) cases observed at one or more of the lead_hours are
    kept, and within them only valid times that are observed or predicted. Each
    such valid time falls in exactly one category:
      - comparable: observed and predicted
      - miss: observed, not predicted, and the model predicts no valid time of
        that case
      - early_dissipation: observed, not predicted, but the model predicts other
        valid times of that case
      - late_dissipation: predicted but not observed
    """

    def present(df):
        if variable == "position":
            return df["lat"].notna() & df["lon"].notna()
        return df[variable].notna()

    obs = gt_df[KEYS].assign(obs=present(gt_df).to_numpy()).drop_duplicates(subset=KEYS)
    pred = pred_df[KEYS].assign(pred=present(pred_df).to_numpy())
    if "ensemble_idx" in pred_df.columns:
        # TC exists at a valid time if enough members have it
        n_members = pred_df["ensemble_idx"].nunique()
        pred = pred.groupby(KEYS, as_index=False)["pred"].sum()
        pred["pred"] = pred["pred"] / n_members >= ens_fraction

    m = obs.merge(pred, on=KEYS, how="left")
    m["pred"] = m["pred"].fillna(False).astype(bool)
    if lead_hours is not None:
        m = m[_lead_hours(m).isin(lead_hours)]

    m = m.sort_values(KEYS)
    case = m.groupby(CASE_KEYS)
    case_n_obs = case["obs"].transform("sum")
    case_n_pred = case["pred"].transform("sum")
    m = m[(case_n_obs > 0) & (m["obs"] | m["pred"])].copy()
    case_n_pred = case_n_pred.loc[m.index]

    # Mutually exclusive categories per valid time
    m["comparable"] = m["obs"] & m["pred"]
    m["miss"] = m["obs"] & ~m["pred"] & (case_n_pred == 0)
    m["early_dissipation"] = m["obs"] & ~m["pred"] & (case_n_pred > 0)
    m["late_dissipation"] = ~m["obs"] & m["pred"]
    return m.reset_index(drop=True)


def _coverage_summary(
    pred_df, gt_df, variable="wind_max", lead_hours=None, ens_fraction=0.3
) -> dict:
    """Counts and fractions (of n_cases) of the _coverage_flags categories for
    one model, counted per (sid, initial_time, valid_time).
    n_comparable + n_miss + n_early_dissipation + n_late_dissipation = n_cases."""
    flags = _coverage_flags(
        pred_df,
        gt_df,
        variable=variable,
        lead_hours=lead_hours,
        ens_fraction=ens_fraction,
    )
    n = len(flags)
    out = {"n_cases": n}
    for col in ["comparable", "miss", "early_dissipation", "late_dissipation"]:
        count = int(flags[col].sum())
        out[f"n_{col}"] = count
        out[f"frac_{col}"] = round(count / n, 4) if n else np.nan
    return out


def _coverage_by_lead(
    pred_df, gt_df, variable="wind_max", lead_hours=None, ens_fraction=0.3
) -> pd.Series:
    """Hit fraction (comparable / all _coverage_flags categories) per lead
    hour for one model. Leads with no flagged valid times are NaN."""
    flags = _coverage_flags(
        pred_df,
        gt_df,
        variable=variable,
        lead_hours=lead_hours,
        ens_fraction=ens_fraction,
    )
    hits = flags.groupby(_lead_hours(flags).astype(int))["comparable"].mean()
    return hits.reindex(lead_hours) if lead_hours is not None else hits


# %% plotting functions
COVERAGE_HATCHES = {
    "miss": ("Miss", ""),
    "early_dissipation": ("Early dissipation", "////"),
    "late_dissipation": ("Late dissipation", "...."),
}


def _tex_rc(usetex: bool = True) -> dict:
    """rcParams for small serif figures in the default LaTeX font (Computer
    Modern). Uses LaTeX when it is installed, else matplotlib's bundled
    Computer Modern via mathtext."""
    rc = {
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "hatch.linewidth": 0.6,
    }
    if usetex and shutil.which("latex"):
        rc.update({"text.usetex": True, "font.serif": ["Computer Modern Roman"]})
    else:
        if usetex:
            print("[PLOT] LaTeX not found; using matplotlib's Computer Modern.")
        rc.update(
            {
                "mathtext.fontset": "cm",
                "font.serif": ["cmr10"],
                "axes.formatter.use_mathtext": True,
                "axes.unicode_minus": False,  # cmr10 has no unicode minus glyph
            }
        )
    return rc


def _tex_safe(label: str) -> str:
    """Escape characters that LaTeX treats specially in plain-text labels."""
    if not mpl.rcParams["text.usetex"]:
        return label
    for char in ["_", "%", "&", "#"]:
        label = label.replace(char, "\\" + char)
    return label


def _escape_pct(label: str) -> str:
    """Escape only '%' for LaTeX, for labels that also hold math ($...$)."""
    return label.replace("%", r"\%") if mpl.rcParams["text.usetex"] else label


def _pretty(model: str) -> str:
    """Display name of a raw model name, safe for LaTeX."""
    return _tex_safe(pretty_names.get(model, model))


def _cmap(name, colors, under=None):
    """Linear colormap through colors, with SCORECARD_MISSING for missing values
    and, if given, the under color for values below the norm's range."""
    cmap = mpl.colors.LinearSegmentedColormap.from_list(name, colors)
    cmap.set_bad(SCORECARD_MISSING)
    if under is not None:
        cmap.set_under(under)
    return cmap


def _save_or_show(fig, out_path=None, show=False):
    """Draw the figure, then save it to out_path and/or show it. Call inside the
    rc context so every label (including lazily created tick labels) uses its
    fonts."""
    fig.canvas.draw()
    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()


def _plot_lead_panel(
    ax,
    table,
    models,
    model_colors,
    lead_ticks,
    ylabel,
    title,
    mean_bars=True,
    ybottom=0,
):
    """Lines of a per-lead summary for each model, with the pooled 'mean' as
    bars to the right of the lead-time axis (unless mean_bars is False).
    ybottom is the lower y limit, or None to leave it free. Returns the models
    plotted."""
    leads = [i for i in table.index if i != "mean"]
    x = np.asarray(leads, dtype=float)
    plotted = []
    for model in models:
        if model not in table.columns:
            continue
        values = table.loc[leads, model].to_numpy(dtype=float)
        valid = ~np.isnan(values)
        if not valid.any():
            continue
        ax.plot(
            x[valid],
            values[valid],
            color=model_colors[model],
            ls=MODEL_LINESTYLES.get(model, "-"),
            lw=0.6,
            marker="o",
            ms=1.0,
        )
        plotted.append(model)

    if mean_bars:
        # Mean across lead times as one bar per model, right of a divider
        bar_start, bar_span = x.max() + 12, 36
        width = bar_span / max(len(plotted), 1)
        for i, model in enumerate(plotted):
            ax.bar(
                bar_start + width * (i + 0.5),
                float(table.loc["mean", model]),
                width=0.85 * width,
                color=model_colors[model],
            )
        ax.axvline(x.max() + 6, color="0.6", lw=0.6, ls="--")
        ax.set_xticks([*lead_ticks, bar_start + bar_span / 2])
        ax.set_xticklabels([*(str(h) for h in lead_ticks), "Mean"])
        ax.set_xlim(0, bar_start + bar_span + 3)
    else:
        ax.set_xticks(list(lead_ticks))
        ax.set_xlim(0, x.max() + 6)

    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    if ybottom is not None:
        ax.set_ylim(bottom=ybottom)
    return plotted


def _plot_overall_bars(ax, values, model_colors, xlabel, title, signed=False):
    """Horizontal bars of one overall score per model (values: Series indexed
    by raw model name, in top-to-bottom order), annotated with their values.
    signed scores get a symmetric x range around a zero line."""
    y = np.arange(len(values))
    v = values.to_numpy(dtype=float)
    bars = ax.barh(
        y,
        np.nan_to_num(v),
        height=0.75,
        color=[model_colors[m] for m in values.index],
        edgecolor="black",
        linewidth=0.4,
    )
    for bar, model in zip(bars, values.index):
        bar.set_hatch(_model_hatch(model))
    vmax = np.nanmax(np.abs(v)) if np.isfinite(v).any() else 0.0
    vmax = vmax if vmax > 0 else 0.01
    if signed:
        ax.axvline(0, color="0.35", lw=0.6)
        ax.set_xlim(-1.4 * vmax, 1.4 * vmax)
    else:
        ax.set_xlim(0, 1.25 * vmax)
    pad = 0.02 * vmax
    for yi, val in zip(y, v):
        if not np.isfinite(val):
            continue
        text = f"${val:+.3f}$" if signed else f"${val:.3f}$"
        ax.text(
            val + (pad if val >= 0 else -pad),
            yi,
            text,
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=6,
        )
    ax.set_yticks(y)
    ax.set_ylim(len(values) - 0.5, -0.5)  # first model on top
    ax.grid(True, axis="x", color="0.85", lw=0.4, ls="--")
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")


def _plot_coverage_panel(ax, coverage_df, models, model_colors, title):
    """Number of hits (comparable predictions) per model as bars (left), and
    stacked columns of misses, early and late dissipation per model (right), in
    thousands. Returns the models plotted."""
    plotted = [m for m in models if m in coverage_df.index]
    n = len(plotted)
    width = 0.8
    left = np.arange(n)
    right = n + 1 + np.arange(n)

    for i, model in enumerate(plotted):
        color = model_colors[model]
        row = coverage_df.loc[model]
        ax.bar(left[i], row["n_comparable"] / 1000, width=width, color=color)
        bottom = 0.0
        for cat, (_, hatch) in COVERAGE_HATCHES.items():
            height = float(row[f"n_{cat}"]) / 1000
            ax.bar(
                right[i],
                height,
                bottom=bottom,
                width=width,
                color=color,
                hatch=hatch,
                edgecolor="white",
                linewidth=0.4,
            )
            bottom += height

    ax.axvline(n, color="0.6", lw=0.6, ls="--")
    ax.set_xticks([(n - 1) / 2, n + 1 + (n - 1) / 2])
    ax.set_xticklabels(["Hits", "Not comparable"])
    ax.set_xlim(-0.8, 2 * n + 0.8)
    ax.set_ylabel(r"Predictions ($\times 10^3$)")
    ax.set_title(title, loc="left")
    return plotted


def _plot_order(model_names) -> list:
    """Model names with the baselines (climatology, then persistence) last."""
    baselines = [m for m in model_names if m.startswith("CLIM")] + [
        m for m in model_names if m.startswith("PERSIST")
    ]
    return [m for m in model_names if m not in baselines] + baselines


def _finish_figure(
    fig,
    models,
    model_colors,
    title,
    out_path=None,
    show=False,
    hatches=COVERAGE_HATCHES,
    hatch_models=False,
):
    """Add the shared legend (model colors + hatches, by default the coverage
    ones; an empty dict for none) and the suptitle, then draw, save and/or show
    (_save_or_show). With hatch_models, the model entries carry their
    _model_hatch, matching hatched bars."""
    model_handles = [
        Patch(
            facecolor=model_colors[m],
            label=_pretty(m),
            **(
                {"hatch": _model_hatch(m), "edgecolor": "black", "linewidth": 0.4}
                if hatch_models
                else {}
            ),
        )
        for m in models
    ]
    hatch_handles = [
        Patch(
            facecolor="0.55",
            edgecolor="white",
            linewidth=0.4,
            hatch=hatch,
            label=label,
        )
        for label, hatch in hatches.values()
    ]
    # Legend columns fill top to bottom with one row per hatch entry; pad the
    # model entries with blanks to whole columns so the hatches always form the
    # last column on their own
    n_rows = len(hatch_handles) or 3
    n_model_cols = max(-(-len(model_handles) // n_rows), 1)  # ceiling division
    blank = Patch(facecolor="none", edgecolor="none", label="")
    model_handles += [blank] * (n_model_cols * n_rows - len(model_handles))
    fig.legend(
        handles=model_handles + hatch_handles,
        loc="outside lower center",
        ncol=n_model_cols + (1 if hatch_handles else 0),
        frameon=False,
        handlelength=1.4,
        columnspacing=1.0,
    )
    fig.suptitle(title, fontsize=10)
    _save_or_show(fig, out_path, show)


def plot_deterministic_summary(
    summary_dict: dict,
    coverage_df: pd.DataFrame,
    model_colors: dict,
    model_names: list,
    out_path: str | None = None,
    lead_ticks=(24, 48, 72, 96, 120),
    usetex: bool = True,
    show: bool = False,
):
    """Landscape 2 x 3 figure of the deterministic metrics, sized to fit half an
    A4 page with margins (170 x 120 mm).

    Row 1: wind_max MAE | coverage | direct position error
    Row 2: pressure_min MAE | along-track error | cross-track error

    The error panels show each model's per-lead summary as lines, and the mean
    across lead times as bars to the right. The coverage panel shows the number
    of hits per model (left) and stacked misses / early / late dissipation
    (right). A shared legend gives the model colors and the coverage hatches.
    The intensity panels show the MAE, whereas the deterministic scorecard
    shows the RMSE.

    Args:
        summary_dict: per-metric summary tables (index lead hours + 'mean',
            columns raw model names), as built in the main block.
        coverage_df: coverage counts indexed by raw model name.
        model_colors: color lookup from _model_colors.
        model_names: raw model names, in plotting order.
        out_path: if given, the figure is saved there (e.g. a .pdf).
        show: display the figure (e.g. in the notebook output).
    """
    panels = [
        ("AbsErr_wind_max", r"MAE $V_\mathrm{max}$ (kt)", "(a) Maximum wind"),
        (None, None, "(b) Coverage"),
        ("DirPosErr", "DPE (km)", "(c) Direct position error"),
        ("AbsErr_pressure_min", r"MAE $P_\mathrm{min}$ (hPa)", "(d) Minimum pressure"),
        ("AlongTrack", "ATE (km)", "(e) Along-track error"),
        ("CrossTrack", "CTE (km)", "(f) Cross-track error"),
    ]
    model_names = _plot_order(model_names)

    with mpl.rc_context(_tex_rc(usetex)):
        fig, axes = plt.subplots(
            2, 3, figsize=(170 * MM, 120 * MM), layout="constrained"
        )
        shown = set()
        for ax, (metric, ylabel, title) in zip(axes.flat, panels):
            if metric is None:
                shown.update(
                    _plot_coverage_panel(
                        ax, coverage_df, model_names, model_colors, title
                    )
                )
            elif metric in summary_dict:
                shown.update(
                    _plot_lead_panel(
                        ax,
                        summary_dict[metric],
                        model_names,
                        model_colors,
                        lead_ticks,
                        ylabel,
                        title,
                    )
                )
            else:
                ax.set_axis_off()
            ax.tick_params(length=2.5, pad=2)

        _finish_figure(
            fig,
            [m for m in model_names if m in shown],
            model_colors,
            "",  # "Deterministic Evaluation Summary",
            out_path,
            show,
        )
    return fig


def plot_probabilistic_summary(
    summary_dict: dict,
    coverage_df: pd.DataFrame,
    model_colors: dict,
    model_names: list,
    out_path: str | None = None,
    lead_ticks=(24, 48, 72, 96, 120),
    usetex: bool = True,
    show: bool = False,
):
    """Landscape 2 x 2 figure of the probabilistic metrics, sized to fit half an
    A4 page with margins (170 x 120 mm).

    Row 1: wind_max fair CRPS | coverage
    Row 2: pressure_min fair CRPS | track CRPS

    The CRPS panels show each model's per-lead summary as lines, and the mean
    across lead times as bars to the right. Only models with probabilistic
    scores are shown, so the coverage panel is restricted to them (ensemble
    coverage uses the ens_fraction threshold of _coverage_flags). A shared
    legend gives the model colors and the coverage hatches.

    Args: as plot_deterministic_summary.
    """
    panels = [
        ("FairCRPS_wind_max", r"CRPS $V_\mathrm{max}$ (kt)", "(a) Maximum wind"),
        (
            "FairCRPS_pressure_min",
            r"CRPS $P_\mathrm{min}$ (hPa)",
            "(c) Minimum pressure",
        ),
        ("TrackCRPS", "Track CRPS (km)", "(d) Track"),
    ]
    model_names = _plot_order(model_names)

    with mpl.rc_context(_tex_rc(usetex)):
        fig, axes = plt.subplots(
            2, 2, figsize=(170 * MM, 120 * MM), layout="constrained"
        )
        coverage_ax = axes[0, 1]
        shown = set()
        for ax, (metric, ylabel, title) in zip(
            [axes[0, 0], axes[1, 0], axes[1, 1]], panels
        ):
            if metric in summary_dict:
                shown.update(
                    _plot_lead_panel(
                        ax,
                        summary_dict[metric],
                        model_names,
                        model_colors,
                        lead_ticks,
                        ylabel,
                        title,
                    )
                )
            else:
                ax.set_axis_off()

        # Coverage for the models that have probabilistic scores
        prob_models = [m for m in model_names if m in shown]
        _plot_coverage_panel(
            coverage_ax, coverage_df, prob_models, model_colors, "(b) Coverage"
        )
        for ax in axes.flat:
            ax.tick_params(length=2.5, pad=2)

        _finish_figure(
            fig,
            prob_models,
            model_colors,
            "",  # "Probabilistic Evaluation Summary",
            out_path,
            show,
        )
    return fig


def plot_ri_summary(
    aggregate_summary: dict,
    model_colors: dict,
    model_names: list,
    out_path: str | None = None,
    lead_ticks=(24, 48, 72, 96, 120),
    usetex: bool = True,
    show: bool = False,
):
    """Landscape 2 x 2 figure of the rapid intensification skill, sized to fit
    half an A4 page with margins (170 x 120 mm). Not drawn by default (its call
    in the main block is commented out); the scorecard shows the RI BSS.

    Row 1: overall CSI | overall PSS (horizontal bars, one per model)
    Row 2: CSI by lead time | PSS by lead time (lines)

    The overall scores are the pooled 'mean' rows of the RI_CSI and RI_PSS
    summaries (all lead times scored together, not an average of the per-lead
    scores). Models are ordered by overall CSI, best on top, in both bar panels;
    model names label the CSI panel. Only models with an overall CSI are shown.

    Args:
        aggregate_summary: per-score summary tables (index lead hours + 'mean',
            columns raw model names), as built in the main block; must hold
            'RI_CSI' and 'RI_PSS'.
        model_colors: color lookup from _model_colors.
        model_names: raw model names (baselines are moved last in the legend).
        out_path: if given, the figure is saved there (e.g. a .pdf).
        show: display the figure (e.g. in the notebook output).
    """
    csi, pss = aggregate_summary["RI_CSI"], aggregate_summary["RI_PSS"]
    models = [m for m in _plot_order(model_names) if m in csi.columns]
    overall_csi = csi.loc["mean", models].astype(float).dropna()
    bar_order = overall_csi.sort_values(ascending=False).index.tolist()
    models = [m for m in models if m in bar_order]

    with mpl.rc_context(_tex_rc(usetex)):
        fig, axes = plt.subplots(
            2, 2, figsize=(170 * MM, 120 * MM), layout="constrained"
        )
        _plot_overall_bars(
            axes[0, 0],
            overall_csi[bar_order],
            model_colors,
            "CSI",
            "(a) Critical success index (overall)",
        )
        axes[0, 0].set_yticklabels([_pretty(m) for m in bar_order])
        _plot_overall_bars(
            axes[0, 1],
            pss.loc["mean", bar_order].astype(float),
            model_colors,
            "PSS (TPR $-$ FPR)",
            "(b) Peirce skill score (overall)",
            signed=True,
        )
        axes[0, 1].set_yticklabels([])

        _plot_lead_panel(
            axes[1, 0],
            csi,
            models,
            model_colors,
            lead_ticks,
            "CSI",
            "(c) Critical success index by lead time",
            mean_bars=False,
        )
        _plot_lead_panel(
            axes[1, 1],
            pss,
            models,
            model_colors,
            lead_ticks,
            "PSS",
            "(d) Peirce skill score by lead time",
            mean_bars=False,
            ybottom=None,
        )
        axes[1, 1].axhline(0, color="0.35", lw=0.6)
        for ax in axes.flat:
            ax.tick_params(length=2.5, pad=2)

        _finish_figure(
            fig,
            models,
            model_colors,
            "Rapid Intensification Skill Summary",
            out_path,
            show,
            hatches={},
            hatch_models=True,
        )
    return fig


def _wrap_two_lines(label: str, models) -> str:
    """Break label at the space nearest its middle when it is longer than the
    longest word of any model name (so no line is wider than necessary and no
    word is split), giving at most two lines."""
    width = max(len(word) for m in models for word in _pretty(m).split())
    spaces = [i for i, c in enumerate(label) if c == " "]
    if len(label) <= width or not spaces:
        return label
    cut = min(spaces, key=lambda i: abs(i - len(label) / 2))
    return label[:cut] + "\n" + label[cut + 1 :]


# Colorbar label of the hit fraction shading in the scorecards
HIT_FRACTION_LABEL = "Hit fraction\n(hits / observed or predicted)"


def _plot_scorecard_panel(
    ax,
    table,
    coverage,
    models,
    baseline,
    kind,
    score_cmap,
    score_norm,
    hit_cmap,
    target=None,
):
    """One scorecard panel: models as rows, the table's lead times as columns.

    Each cell shows the model's score as text, and its upper-left triangle is
    colored by
      - kind 'error' (lower is better): the % difference from the baseline
        model at that lead (negative: lower error than the baseline);
      - kind 'skill' (higher is better, e.g. BSS): the score itself;
      - kind 'calibration' (a frequency that should equal target, e.g. how
        often the observed position falls inside a probability circle): the
        difference from target in percentage points, with the score in %.
    The lower-right triangle is colored by the hit fraction (coverage table);
    with coverage None, cells are not split and the upper-left color fills
    them. Missing values take the colormaps' 'bad' color.

    Returns the score and hit fraction PolyCollections (the latter None without
    coverage), for the colorbars.
    """
    leads = list(table.index)
    split = coverage is not None
    upper, lower, shade, hits = [], [], [], []
    for i, model in enumerate(models):
        for j, lead in enumerate(leads):
            if split:
                upper.append([(j, i), (j + 1, i), (j, i + 1)])
                lower.append([(j + 1, i), (j + 1, i + 1), (j, i + 1)])
            else:
                upper.append([(j, i), (j + 1, i), (j + 1, i + 1), (j, i + 1)])
            value = table.at[lead, model] if model in table.columns else np.nan
            if kind == "skill":
                shade.append(value)
            elif kind == "calibration":
                shade.append(100 * (value - target))
            else:
                base = table.at[lead, baseline]
                shade.append(100 * (value - base) / base if base else np.nan)
            hits.append(
                coverage.at[lead, model]
                if split and model in coverage.columns
                else np.nan
            )

            if not np.isfinite(value):
                text = "--"
            elif kind == "skill":
                text = f"{value:.2f}"
            elif kind == "calibration":
                text = f"{100 * value:.0f}"
            else:
                text = f"{value:.0f}" if abs(value) >= 100 else f"{value:.1f}"
            # The text straddles the diagonal (two background colors), so it is
            # black on a white box to stay legible on both
            ax.text(
                j + 0.5,
                i + 0.5,
                text,
                ha="center",
                va="center",
                fontsize=4.5,
                color="black",
                bbox=dict(
                    boxstyle="square,pad=0.2", facecolor="white", edgecolor="none"
                ),
            )

    scored = PolyCollection(
        upper,
        array=np.ma.masked_invalid(shade),
        cmap=score_cmap,
        norm=score_norm,
        edgecolor="none",
    )
    ax.add_collection(scored)
    covered = None
    if split:
        covered = PolyCollection(
            lower,
            array=np.ma.masked_invalid(hits),
            cmap=hit_cmap,
            norm=mpl.colors.Normalize(0, 1),
            edgecolor="none",
        )
        ax.add_collection(covered)

    # Hatch the rows of the models drawn dashed / hatched in the other figures
    hatched = [
        [(j, i), (j + 1, i), (j + 1, i + 1), (j, i + 1)]
        for i, model in enumerate(models)
        if _model_hatch(model)
        for j in range(len(leads))
    ]
    if hatched:
        ax.add_collection(
            PolyCollection(
                hatched,
                facecolor="none",
                edgecolor=(0, 0, 0, 0.3),
                linewidth=0,
                hatch=SCORECARD_HATCH,
            )
        )

    # Cell borders
    for j in range(len(leads) + 1):
        ax.axvline(j, color="white", lw=0.8)
    for i in range(len(models) + 1):
        ax.axhline(i, color="white", lw=0.8)

    ax.set_xlim(0, len(leads))
    ax.set_ylim(len(models), 0)  # first model on top
    ax.set_xticks(np.arange(len(leads)) + 0.5)
    ax.set_xticklabels([str(h) for h in leads])
    ax.set_yticks(np.arange(len(models)) + 0.5)
    ax.set_xlabel("Lead time (h)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=2)
    return scored, covered


def _scorecard_models(model_names, keep=lambda model: True) -> list:
    """Scorecard rows: the persistence baseline on top, then the other models
    (in plotting order) for which keep(model) is true. The climatological
    baseline is left out."""
    baseline = _baseline(model_names, "PERSIST")
    if baseline is None:
        raise ValueError("No persistence model found for the scorecard baseline.")
    return [baseline] + [
        m
        for m in _plot_order(model_names)
        if not m.startswith(("PERSIST", "CLIM")) and keep(m)
    ]


def _error_panel(title, table, coverage, pct_limit, label) -> dict:
    """Scorecard panel spec of an 'error' metric (lower is better), shaded by
    the % difference from persistence, clipped at +-pct_limit."""
    return dict(
        title=title,
        table=table,
        coverage=coverage,
        kind="error",
        cmap=_cmap(
            "scorecard_improvement", [SCORECARD_BETTER, "white", SCORECARD_WORSE]
        ),
        norm=mpl.colors.Normalize(-pct_limit, pct_limit),
        label=label,
        extend="both",
    )


def _scorecard_figure(
    panels, models, out_path=None, usetex=True, show=False, cbar_label_size=6
):
    """Draw a scorecard, 170 mm wide: one _plot_scorecard_panel per panel spec
    side by side, models as rows (persistence, models[0], is the baseline),
    with model names on the first panel only and each panel's colorbars
    underneath (score, then hit fraction).

    panels: dicts with the keys title, table, coverage, kind, cmap, norm, label
    and extend (of the score colorbar), plus target for 'calibration' panels.
    A '%' in titles and labels is escaped for LaTeX here.
    """
    hit_cmap = _cmap("scorecard_hit_fraction", ["white", SCORECARD_COVERAGE])

    with mpl.rc_context(_tex_rc(usetex)):
        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=(170 * MM, (6 * len(models) + 55) * MM),
            layout="constrained",
            squeeze=False,
        )
        axes = axes[0]
        colorbars = []  # (mappable, ax, label, extend)
        for ax, panel in zip(axes, panels):
            scored, covered = _plot_scorecard_panel(
                ax,
                panel["table"],
                panel["coverage"],
                models,
                models[0],
                panel["kind"],
                panel["cmap"],
                panel["norm"],
                hit_cmap,
                target=panel.get("target"),
            )
            # Colorbars added first sit lowest, so the score colorbar ends up
            # directly under the panel and the hit fraction one below it
            if covered is not None:
                colorbars.append((covered, ax, HIT_FRACTION_LABEL, "neither"))
            colorbars.append((scored, ax, _escape_pct(panel["label"]), panel["extend"]))
            ax.set_title(_escape_pct(panel["title"]), loc="left")
            ax.set_yticklabels([])
        axes[0].set_yticklabels([_wrap_two_lines(_pretty(m), models) for m in models])
        axes[0].tick_params(axis="y", labelsize=5.5)

        for mappable, ax, label, extend in colorbars:
            cbar = fig.colorbar(
                mappable,
                ax=ax,
                location="bottom",
                shrink=0.9,
                aspect=20,
                extend=extend,
            )
            cbar.set_label(label, fontsize=cbar_label_size)
            cbar.ax.tick_params(length=2, pad=1, labelsize=5.5)

        _save_or_show(fig, out_path, show)
    return fig


def plot_deterministic_scorecard(
    summary_dict: dict,
    coverage_by_lead: dict,
    model_names: list,
    metrics_to_show=(
        ("SquErr_wind_max", r"(a) RMSE $V_\mathrm{max}$ (kt)", "error"),
        ("SquErr_pressure_min", r"(b) RMSE $P_\mathrm{min}$ (hPa)", "error"),
        ("RI_BSS", "(c) RI BSS (vs Mean Tendency)", "skill"),
    ),
    leads=(12, 24, 36, 48, 72, 96, 120),
    pct_limit: float = 100.0,
    out_path: str | None = None,
    usetex: bool = True,
    show: bool = False,
):
    """Scorecard of deterministic scores (_scorecard_figure). Every cell holds
    the model's score.
      - 'error' panels (lower is better, e.g. RMSE): cells are split along the
        diagonal. The upper-left triangle is colored by the % difference from
        persistence at that lead (honolulu blue: lower error, vivid rose:
        higher, centered on white, clipped at +-pct_limit). The lower-right
        triangle is colored by the hit fraction of the variable the metric is
        computed from, from white (no hits) to bamboo (all hits).
      - 'skill' panels (higher is better, e.g. BSS): cells are not split and
        are colored by the score as written (2 decimals), from white (0) to
        ocean green (the panel's maximum); negative scores (worse than the
        reference, the mean tendency climatology for the BSS) are vivid rose.
    Persistence is the top row; the climatological baseline is not shown.

    Args:
        summary_dict: per-metric summary tables (index lead hours + 'mean',
            columns raw model names), as built in the main block; for the RI
            panel, merge in the aggregate_summary tables.
        coverage_by_lead: hit fraction tables keyed by variable ('wind_max',
            'pressure_min', 'position'; index lead hours, columns raw model
            names), as built in the main block.
        model_names: raw model names, in plotting order.
        metrics_to_show: (metric, panel title, 'error' or 'skill') triples.
        leads: lead hours shown as columns.
        pct_limit: color scale limit of the % difference ('error' panels).
        out_path: if given, the figure is saved there (e.g. a .pdf).
        show: display the figure (e.g. in the notebook output).
    """
    models = _scorecard_models(model_names)
    panels = []
    for metric, title, kind in metrics_to_show:
        if metric not in summary_dict:
            continue
        table = summary_dict[metric].loc[list(leads)].astype(float)
        if kind == "error":
            panels.append(
                _error_panel(
                    title,
                    table,
                    coverage_by_lead.get(METRIC_COVERAGE_VARIABLE.get(metric)),
                    pct_limit,
                    r"% difference with persistence ($\downarrow$ is better)",
                )
            )
            continue
        # 'skill': colored by the score as written, so e.g. -0.004 is a white
        # 0.00 rather than a red -0.00 (+ 0.0 turns -0.0 into 0.0)
        table = table.round(2) + 0.0
        values = table.reindex(columns=models).to_numpy()
        vmax = np.nanmax(values, initial=0)
        panels.append(
            dict(
                title=title,
                table=table,
                coverage=None,
                kind="skill",
                cmap=_cmap(
                    "scorecard_score",
                    [SCORECARD_SCORE_LOW, SCORECARD_SCORE_HIGH],
                    under=SCORECARD_WORSE,
                ),
                norm=mpl.colors.Normalize(0, vmax if vmax > 0 else 1),
                label=f"{metric.split('_')[-1]} (higher is better)",
                # Negative scores take the colormap's 'under' color
                extend="min" if np.nanmin(values, initial=0) < 0 else "neither",
            )
        )
    return _scorecard_figure(panels, models, out_path, usetex, show, cbar_label_size=6)


def plot_probabilistic_scorecard(
    summary_dict: dict,
    coverage_by_lead: dict,
    model_names: list,
    metrics_to_show=(
        (
            "FairCRPS_wind_max",
            r"(a) CRPS $V_\mathrm{max}$ (kt)",
            "error",
            "AbsErr_wind_max",
        ),
        (
            "FairCRPS_pressure_min",
            r"(b) CRPS $P_\mathrm{min}$ (hPa)",
            "error",
            "AbsErr_pressure_min",
        ),
        ("TrackCRPS", "(c) Track CRPS (km)", "error", "DirPosErr"),
        (
            "ProbCircle_hit70",
            "(d) Track circle reliability",
            "calibration",
            None,
        ),
    ),
    leads=(12, 24, 36, 48, 72, 96, 120),
    pct_limit: float = 100.0,
    nominal: float = 0.7,
    pp_limit: float = 30.0,
    out_path: str | None = None,
    usetex: bool = True,
    show: bool = False,
):
    """Scorecard of probabilistic scores (_scorecard_figure). Every cell holds
    the model's score, and is split along the diagonal with the lower-right
    triangle colored by the hit fraction of the variable the metric is computed
    from.
      - 'error' panels (CRPS, lower is better): the upper-left triangle is
        colored by the % difference from persistence at that lead, as in
        plot_deterministic_scorecard. Persistence is a point forecast, whose
        CRPS reduces to its absolute error (track CRPS: direct position
        error), so its row shows the baseline_metric of the panel.
      - 'calibration' panels (frequency of the observed position inside the
        probability circle, in %): the upper-left triangle is colored by that
        frequency minus nominal, in percentage points (purple: below,
        underdispersive; amber: above, overdispersive; clipped at
        +-pp_limit). Persistence has no ensemble, so its cells are blank.
    Persistence is the top row, followed by the models with any probabilistic
    score; the climatological baseline is not shown.

    Args:
        summary_dict: per-metric summary tables (index lead hours + 'mean',
            columns raw model names), as built in the main block.
        coverage_by_lead: hit fraction tables keyed by variable, as built in
            the main block.
        model_names: raw model names, in plotting order.
        metrics_to_show: (metric, panel title, 'error' or 'calibration',
            baseline_metric) tuples; baseline_metric is the summary_dict table
            whose persistence column is the baseline ('error' panels only).
        leads: lead hours shown as columns.
        pct_limit: color scale limit of the % difference ('error' panels).
        nominal: nominal probability of the 'calibration' panels.
        pp_limit: color scale limit, in percentage points ('calibration' panels).
        out_path: if given, the figure is saved there (e.g. a .pdf).
        show: display the figure (e.g. in the notebook output).
    """
    shown = [p for p in metrics_to_show if p[0] in summary_dict]
    models = _scorecard_models(
        model_names,
        keep=lambda m: any(summary_dict[p[0]][m].notna().any() for p in shown),
    )
    baseline = models[0]
    panels = []
    for metric, title, kind, baseline_metric in shown:
        table = summary_dict[metric].loc[list(leads)].astype(float).copy()
        coverage = coverage_by_lead.get(METRIC_COVERAGE_VARIABLE.get(metric))
        if kind == "error":
            # Persistence CRPS = its absolute error (point forecast)
            table[baseline] = (
                summary_dict[baseline_metric].loc[list(leads), baseline]
                if baseline_metric in summary_dict
                else np.nan
            )
            panels.append(
                _error_panel(
                    title,
                    table,
                    coverage,
                    pct_limit,
                    "% difference with\n" r"persistence ($\downarrow$ is better)",
                )
            )
            continue
        # 'calibration': persistence has no ensemble, so its row is left blank
        if coverage is not None:
            coverage = coverage.drop(columns=baseline, errors="ignore")
        panels.append(
            dict(
                title=title,
                table=table,
                coverage=coverage,
                kind="calibration",
                cmap=_cmap(
                    "scorecard_calibration",
                    [SCORECARD_UNDERDISPERSIVE, "white", SCORECARD_OVERDISPERSIVE],
                ),
                norm=mpl.colors.Normalize(-pp_limit, pp_limit),
                label=f"Obs. inside circle $-$ {100 * nominal:.0f}% (pp)\n"
                r"$<0$ underdispersive, $>0$ overdispersive",
                extend="both",
                target=nominal,
            )
        )
    return _scorecard_figure(panels, models, out_path, usetex, show, cbar_label_size=4)


# %%
if __name__ == "__main__":
    # Run settings from the configuration file (CONFIG_PATH)
    year = _config["year"]
    basin = _config.get("basin") or None
    if basin is not None and basin not in BASINS:
        raise ValueError(f"Unknown basin '{basin}'; expected one of {BASINS} or null.")
    # Output file names are tagged '{year}' (all basins) or '{year}_{basin}'
    tag = f"{year}_{basin}" if basin else f"{year}"
    ri_suffix = f"_ri_{year}"
    det_suffix = f"_det_{year}"
    prob_suffix = f"_prob_{year}"
    suffixes = [ri_suffix, det_suffix, prob_suffix]
    plotting_folder = _config["plotting_folder"]
    result_folder = _config["result_folder"]
    _leads = _config["lead_hours"]
    lead_hours = np.arange(_leads["start"], _leads["stop"] + 1, _leads["step"])

    # Aggregate-statistic metrics: each scores a set of cases at once (one value
    # per lead time plus a pooled row) rather than one value per case.
    #   source: suffix of the results files the metric is computed from
    #   scores: output columns reported in the per-lead summary tables
    #   prefix: tag prepended to the output file names
    #   kwargs: extra arguments passed to the metric (optional)
    aggregate_metrics = [
        {
            "metric": metrics.CSI,
            "source": ri_suffix,
            "scores": ["CSI"],
            "prefix": "RI_",
        },
        {
            "metric": metrics.PeirceSS,
            "source": ri_suffix,
            "scores": ["PSS"],
            "prefix": "RI_",
        },
        {
            "metric": metrics.BrierSS,
            "source": ri_suffix,
            "scores": ["BSS"],
            "prefix": "RI_",
        },
        {
            "metric": metrics.RocAUC,
            "source": ri_suffix,
            "scores": ["AUC"],
            "prefix": "RI_",
        },
        {
            "metric": metrics.RSquared,
            "source": det_suffix,
            "scores": ["RSquared_wind_max", "RSquared_pressure_min"],
            "prefix": "",
        },
    ]

    files = os.listdir(result_folder)

    ibtracs_file = None
    for file in files.copy():
        # find the ibtracs file
        if "ibtracs_gt" in file and str(year) in file and file.endswith(".csv"):
            ibtracs_file = file
            files.remove(file)
            break

    if ibtracs_file is None:
        raise FileNotFoundError(
            "No IBTrACS ground truth found in the specified directory."
        )

    candidate_files = _filter_files(files, year, suffixes)
    candidate_files = sorted(
        f for f in candidate_files if _model_name(f, year) not in IGNORED_MODELS
    )
    for model, reason in IGNORED_MODELS.items():
        print(f"[INFO] Ignoring {model}: {reason}")

    # The BSS reference is the mean tendency climatology's RI forecast, scored on
    # the same cases as each model
    clim_ri_file = next(
        (
            f
            for f in candidate_files
            if ri_suffix in f and (_model_name(f, year) or "").startswith("CLIM")
        ),
        None,
    )
    if clim_ri_file is None:
        raise FileNotFoundError(
            "No climatology RI results found for the BSS reference."
        )
    clim_ri_df = _read_results(os.path.join(result_folder, clim_ri_file))

    # Basin evaluation: the all-basin results restricted to the forecast cases
    # initialized in the basin (ground truth and forecasts alike)
    gt_df = _read_results(os.path.join(result_folder, ibtracs_file))
    if basin:
        basin_cases = _basin_cases(_config["ibtracs_tracks_path"], year, basin)
        gt_df = _in_cases(gt_df, basin_cases)
        clim_ri_df = _in_cases(clim_ri_df, basin_cases)
        n_cases = gt_df[CASE_KEYS].drop_duplicates()
        print(
            f"[INFO] Evaluating {len(n_cases)} forecast cases of "
            f"{n_cases['sid'].nunique()} storms initialized in the {basin} basin."
        )

    for spec in aggregate_metrics:
        if spec["metric"] is metrics.BrierSS:
            spec["kwargs"] = {"reference_forecast": clim_ri_df}

    # Make a dictionary that includes a dataframe for each metric to summarize,
    # and populate it so that the lead_hours are the rows (plus another row for the overall
    # mean across all lead_hours), and the columns are the models.
    # Each model is summarized over its own cases (non-homogeneous); coverage
    # problems are reported separately in coverage_df.
    model_names = _model_names(candidate_files, year)
    model_colors = _model_colors(model_names)
    summary_dict = {}
    for metric in summary_columns:
        summary_dict[metric] = pd.DataFrame(
            index=[*lead_hours.tolist(), "mean"], columns=model_names, dtype=float
        )

    coverage_rows = []
    # Per-lead hit fraction for each forecast variable (scorecard)
    coverage_by_lead = {
        variable: pd.DataFrame(index=lead_hours, columns=model_names, dtype=float)
        for variable in ["wind_max", "pressure_min", "position"]
    }

    # Aggregate metrics: one per-lead summary per score (models as columns) and
    # one detail table per metric (all outputs, long format)
    aggregate_summary = {
        f"{spec['prefix']}{score}": pd.DataFrame(
            index=[*lead_hours.tolist(), "mean"], columns=model_names, dtype=float
        )
        for spec in aggregate_metrics
        for score in spec["scores"]
    }
    aggregate_details = {
        f"{spec['prefix']}{spec['metric'].__name__}": [] for spec in aggregate_metrics
    }

    for candidate_file in candidate_files:
        model = _model_name(candidate_file, year)
        df = _read_results(os.path.join(result_folder, candidate_file))
        if basin:
            df = _in_cases(df, basin_cases)
        # Restrict to the evaluated lead times so pooled rows match the per-lead rows
        df = df[_lead_hours(df).isin(lead_hours)]

        for spec in aggregate_metrics:
            if spec["source"] not in candidate_file:
                continue
            name = spec["metric"].__name__
            try:
                # reference first, forecast second; pooled row labelled "mean"
                out = spec["metric"](
                    gt_df, df, overall_label="mean", **spec.get("kwargs", {})
                )
            except ValueError as e:
                print(f"[AGG] {name} failed for {model}: {e}")
                continue
            aggregate_details[f"{spec['prefix']}{name}"].append(out.assign(model=model))
            by_lead = out.set_index("lead_hours")
            for score in spec["scores"]:
                if score in by_lead.columns:  # e.g. no pressure_min forecast
                    aggregate_summary[f"{spec['prefix']}{score}"][model] = by_lead[
                        score
                    ]

        if ri_suffix in candidate_file:
            continue  # RI files only feed the aggregate metrics

        # find the intersection of the columns in df and summary_columns
        columns_to_plot = [col for col in df.columns if col in summary_columns]
        for metric in columns_to_plot:
            summary_dict[metric][model] = _summarize_metric(df, metric, lead_hours)

        # Coverage from the deterministic file, which holds the forecast values
        # for every ground-truth (sid, initial_time, valid_time) key
        # (wind_max, so intensity-only models such as the postprocessing ones
        # are included)
        # Only the lead times the model forecasts at all are used, so models with
        # a sparse lead set (e.g. postprocessing at 6-24, 48, 72, 96, 120 h) are
        # not flagged for leads they never produce.
        if det_suffix in candidate_file and "wind_max" in df.columns:
            model_leads = sorted(
                set(df.loc[df["wind_max"].notna(), "lead_hours"]) & set(lead_hours)
            )
            coverage_rows.append(
                {
                    "model": model,
                    **_coverage_summary(
                        df, gt_df, variable="wind_max", lead_hours=model_leads
                    ),
                    "n_leads": len(model_leads),
                    "coverage_leads": " ".join(str(h) for h in model_leads),
                }
            )

        if det_suffix in candidate_file:
            for variable, table in coverage_by_lead.items():
                cols = ["lat", "lon"] if variable == "position" else [variable]
                if not set(cols) <= set(df.columns):
                    continue
                has_var = df[cols].notna().all(axis=1)
                var_leads = sorted(set(df.loc[has_var, "lead_hours"]) & set(lead_hours))
                table[model] = _coverage_by_lead(
                    df, gt_df, variable=variable, lead_hours=var_leads
                ).reindex(lead_hours)

    coverage_df = pd.DataFrame(coverage_rows).set_index("model")

    os.makedirs(plotting_folder, exist_ok=True)
    for metric, table in summary_dict.items():
        table.rename(columns=pretty_names).to_csv(
            os.path.join(plotting_folder, f"summary_{metric}_{tag}.csv")
        )
    coverage_df.rename(index=pretty_names).to_csv(
        os.path.join(plotting_folder, f"coverage_{tag}.csv")
    )

    for score, table in aggregate_summary.items():
        table.rename(columns=pretty_names).to_csv(
            os.path.join(plotting_folder, f"summary_{score}_{tag}.csv")
        )
    for name, outs in aggregate_details.items():
        if outs:
            detail = pd.concat(outs, ignore_index=True)
            detail["model"] = detail["model"].replace(pretty_names)
            detail.to_csv(
                os.path.join(plotting_folder, f"detail_{name}_{tag}.csv"),
                index=False,
            )
    # %%
    fig = plot_deterministic_summary(
        summary_dict,
        coverage_df,
        model_colors,
        model_names,
        out_path=os.path.join(plotting_folder, f"deterministic_summary_{tag}.pdf"),
        show=True,
    )

    fig = plot_probabilistic_summary(
        summary_dict,
        coverage_df,
        model_colors,
        model_names,
        out_path=os.path.join(plotting_folder, f"probabilistic_summary_{tag}.pdf"),
        show=True,
    )

    # fig = plot_ri_summary(
    #     aggregate_summary,
    #     model_colors,
    #     model_names,
    #     out_path=os.path.join(plotting_folder, f"ri_summary_{tag}.pdf"),
    #     show=True,
    # )

    fig = plot_deterministic_scorecard(
        {**summary_dict, **aggregate_summary},
        coverage_by_lead,
        model_names,
        out_path=os.path.join(plotting_folder, f"deterministic_scorecard_{tag}.pdf"),
        show=True,
    )

    fig = plot_probabilistic_scorecard(
        summary_dict,
        coverage_by_lead,
        model_names,
        out_path=os.path.join(plotting_folder, f"probabilistic_scorecard_{tag}.pdf"),
        show=True,
    )


# %%
