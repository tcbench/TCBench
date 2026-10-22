# %% imports
import sys
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

_DEV_FOLDER = "/home/mgomezd1/repos/iop_tcbench/dev"
if _DEV_FOLDER not in sys.path:
    sys.path.insert(0, _DEV_FOLDER)
os.chdir(_DEV_FOLDER)

import metrics_rev as metrics

# %% Constants
pretty_names = {
    "2023_PANGU": "Pangu",
    "2023_TIGGE_GEFS": "GEFS",
    "2023_TIGGE_IFS": "IFS",
    "2023_aifs": "AIFS",
    "2023_fcnet": "FCNet",
    "2023_fnv3": "FNV3",
    "2023_postprocessing_panguweather_ANN_LeakyReLU,_M": "Postprocessed Pangu (ANN)",
    "2023_postprocessing_panguweather_MLR_Masked": "Postprocessed Pangu (MLR)",
    "2023_postprocessing_panguweather_UNetv2_dout_0.33": "Postprocessed Pangu (UNet)",
    "CLIM-intensity-y2023": "Mean Tendency (CLIM)",
    "PERSIST-intensity-y2023": "Persistence",
    "gencast": "GenCast",
}

det_columns = [
    "AbsErr_wind_max",
    "AbsErr_pressure_min",
    "SquErr_wind_max",
    "SquErr_pressure_min",
    "DirPosErr",
    "AlongTrack",
    "CrossTrack",
]

prob_columns = [
    "FairCRPS_wind_max",
    "FairCRPS_pressure_min",
    "TrackCRPS",
]

summary_columns = [*det_columns, *prob_columns]


# %% helper functions
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
    keys = ["sid", "initial_time", "valid_time", "lead_hours"]
    if "ensemble_idx" in df.columns:
        df = df.groupby(keys, as_index=False)[metric].mean()
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
    """Flag forecast coverage problems per case (sid, initial_time).

    A valid time is observed when the ground truth has the variable, and
    predicted when the forecast has it. variable is 'position' (lat and lon) or
    a column name such as 'wind_max'. For ensembles (ensemble_idx column), the
    TC is predicted at a valid time when at least ens_fraction of the ensemble
    members have it; the denominator is the number of unique ensemble indices.
    Only cases observed at one or more valid times are kept. Flags (not
    mutually exclusive):
      - miss: observed at every valid time, predicted at none
      - early_dissipation: predicted at the first valid time, but not predicted
        at one or more observed valid times
      - late_dissipation: observed at some but not all valid times, and
        predicted at one or more valid times that were not observed
      - no_prediction: predicted at no valid time, regardless of observations
    """

    def present(df):
        if variable == "position":
            return df["lat"].notna() & df["lon"].notna()
        return df[variable].notna()

    keys = ["sid", "initial_time"]

    obs = (
        gt_df[[*keys, "valid_time"]]
        .assign(obs=present(gt_df).to_numpy())
        .drop_duplicates(subset=[*keys, "valid_time"])
    )
    pred = pred_df[[*keys, "valid_time"]].assign(pred=present(pred_df).to_numpy())
    if "ensemble_idx" in pred_df.columns:
        # TC exists at a valid time if enough members have it
        n_members = pred_df["ensemble_idx"].nunique()
        pred = pred.groupby([*keys, "valid_time"], as_index=False)["pred"].sum()
        pred["pred"] = pred["pred"] / n_members >= ens_fraction

    m = obs.merge(pred, on=[*keys, "valid_time"], how="left")
    m["pred"] = m["pred"].fillna(False).astype(bool)
    if lead_hours is not None:
        lead = (m["valid_time"] - m["initial_time"]).dt.total_seconds() / 3600
        m = m[lead.isin(lead_hours)]

    m = m.sort_values([*keys, "valid_time"])
    m["obs_not_pred"] = m["obs"] & ~m["pred"]
    m["pred_not_obs"] = m["pred"] & ~m["obs"]

    u = m.groupby(keys).agg(
        n_steps=("obs", "size"),
        n_obs=("obs", "sum"),
        n_pred=("pred", "sum"),
        pred_first=("pred", "first"),
        obs_not_pred=("obs_not_pred", "any"),
        pred_not_obs=("pred_not_obs", "any"),
    )
    u = u[u["n_obs"] > 0]

    all_obs = u["n_obs"] == u["n_steps"]
    u["miss"] = all_obs & (u["n_pred"] == 0)
    u["early_dissipation"] = u["pred_first"] & u["obs_not_pred"]
    u["late_dissipation"] = ~all_obs & u["pred_not_obs"]
    u["no_prediction"] = u["n_pred"] == 0
    return u.reset_index()


def _coverage_summary(
    pred_df, gt_df, variable="wind_max", lead_hours=None, ens_fraction=0.3
) -> dict:
    """Counts and fractions of the _coverage_flags categories for one model."""
    flags = _coverage_flags(
        pred_df,
        gt_df,
        variable=variable,
        lead_hours=lead_hours,
        ens_fraction=ens_fraction,
    )
    n = len(flags)
    out = {"n_cases": n}
    for col in ["miss", "early_dissipation", "late_dissipation", "no_prediction"]:
        count = int(flags[col].sum())
        out[f"n_{col}"] = count
        out[f"frac_{col}"] = round(count / n, 4) if n else np.nan
    return out


# %%
if __name__ == "__main__":
    year = 2023
    ri_suffix = f"_ri_{year}"
    det_suffix = f"_det_{year}"
    prob_suffix = f"_prob_{year}"
    suffixes = [ri_suffix, det_suffix, prob_suffix]
    plotting_folder = "/home/mgomezd1/repos/idea_testing/outputs_rev/plotting_csvs"
    result_folder = "/home/mgomezd1/repos/idea_testing/outputs_rev"
    lead_hours = np.arange(6, 121, 6)
    # lead_hours = [6, 12, 18, 24, 48, 72, 96, 120] For post-processing RI comparison

    RI_metrics = [
        metrics.CSI,
        metrics.BrierSS,
        metrics.RocAUC,
    ]
    # Headline score of each RI metric, reported in the per-lead summary tables
    RI_score_cols = {"CSI": "CSI", "BrierSS": "BSS", "RocAUC": "AUC"}

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
    candidate_files.sort()

    # Make a dictionary that includes a dataframe for each metric to summarize,
    # and populate it so that the lead_hours are the rows (plus another row for the overall
    # mean across all lead_hours), and the columns are the models.
    # Each model is summarized over its own cases (non-homogeneous); coverage
    # problems are reported separately in coverage_df.
    model_names = _model_names(candidate_files, year)
    summary_dict = {}
    for metric in summary_columns:
        summary_dict[metric] = pd.DataFrame(
            index=[*lead_hours.tolist(), "mean"], columns=model_names, dtype=float
        )

    gt_df = _read_results(os.path.join(result_folder, ibtracs_file))
    coverage_rows = []

    # RI: one per-lead summary (headline score, models as columns) and one
    # detail table (all outputs, long format) per RI metric
    ri_summary = {
        m.__name__: pd.DataFrame(
            index=[*lead_hours.tolist(), "mean"], columns=model_names, dtype=float
        )
        for m in RI_metrics
    }
    ri_details = {m.__name__: [] for m in RI_metrics}

    for candidate_file in candidate_files:
        model = _model_name(candidate_file, year)
        df = _read_results(os.path.join(result_folder, candidate_file))

        if ri_suffix in candidate_file:
            # Restrict to the evaluated lead times so the pooled row matches
            lead = (df["valid_time"] - df["initial_time"]).dt.total_seconds() / 3600
            df = df[lead.isin(lead_hours)]
            for ri_metric in RI_metrics:
                name = ri_metric.__name__
                try:
                    # reference first, forecast second; pooled row labelled "mean"
                    out = ri_metric(gt_df, df, overall_label="mean")
                except ValueError as e:
                    print(f"[RI] {name} failed for {model}: {e}")
                    continue
                ri_details[name].append(out.assign(model=model))
                ri_summary[name][model] = out.set_index("lead_hours")[
                    RI_score_cols[name]
                ]
            continue

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

    coverage_df = pd.DataFrame(coverage_rows).set_index("model")

    os.makedirs(plotting_folder, exist_ok=True)
    for metric, table in summary_dict.items():
        table.rename(columns=pretty_names).to_csv(
            os.path.join(plotting_folder, f"summary_{metric}_{year}.csv")
        )
    coverage_df.rename(index=pretty_names).to_csv(
        os.path.join(plotting_folder, f"coverage_{year}.csv")
    )

    for name, table in ri_summary.items():
        table.rename(columns=pretty_names).to_csv(
            os.path.join(plotting_folder, f"summary_RI_{name}_{year}.csv")
        )
        if ri_details[name]:
            detail = pd.concat(ri_details[name], ignore_index=True)
            detail["model"] = detail["model"].replace(pretty_names)
            detail.to_csv(
                os.path.join(plotting_folder, f"detail_RI_{name}_{year}.csv"),
                index=False,
            )


# %%
