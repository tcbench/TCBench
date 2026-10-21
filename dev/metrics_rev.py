import pandas as pd
import numpy as np


# %% helper functions
def _get_refkeys(col_dict: dict, df: pd.DataFrame) -> tuple:
    column_keys = col_dict.copy()

    ref_keys = [*column_keys.values()]
    if column_keys.get("ensemble_idx", None):
        ref_keys.remove(column_keys.get("ensemble_idx", None))

    return ref_keys


def haversine(lat1, lon1, lat2, lon2, radius=6371.0):
    """Great-circle distance in km; inputs in degrees, broadcastable."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def to_lon360(df: pd.DataFrame, lon_cols) -> pd.DataFrame:
    """Return a copy of df with the given longitude columns wrapped to [0, 360).

    Accepts a single column name or a list of names. NaNs are preserved.
    """
    if isinstance(lon_cols, str):
        lon_cols = [lon_cols]
    df = df.copy()
    for col in lon_cols:
        df[col] = np.mod(df[col], 360.0)
    return df


# %% Deterministic Metrics
def AbsErr(
    reference_df: pd.DataFrame, forecast_df: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    """Absolute Error (AbsErr) metric.
    Args:
        reference_df (pd.DataFrame): DataFrame containing the reference values.
        forecast_df (pd.DataFrame): DataFrame containing the forecast values.
        **kwargs: Additional keyword arguments

        Additional arguments:
        column_keys (dict): Dictionary mapping columns containing the initial time,
        valid time, and storm ID.


        target_cols (list): List with the target columns to be evaluated.
        Default is ['wind_max', 'pressure_min']

    Returns:
        pd.DataFrame: DataFrame containing the computed metric values.
    """

    ref_keys = _get_refkeys(
        kwargs.get(
            "column_keys",
            {
                "storm_id": "sid",
                "initial_time": "initial_time",
                "valid_time": "valid_time",
                "ensemble_idx": "ensemble_idx",
                "lead_hours": "lead_hours",
            },
        ),
        forecast_df,
    )

    # ref_keys = [*column_keys.values()]
    # if column_keys.get("ensemble_idx", None):
    #     ref_keys.remove(column_keys.get("ensemble_idx", None))

    # if (
    #     column_keys.get("ensemble_idx")
    #     and column_keys.get("ensemble_idx") not in forecast_df.columns
    # ):
    #     column_keys.pop("ensemble_idx", None)

    target_cols = kwargs.get("target_cols", ["wind_max", "pressure_min"])

    eval_df = forecast_df.merge(
        reference_df[[*ref_keys, *target_cols]],
        on=ref_keys,
        how="left",
        suffixes=("_fcst", "_ref"),
        validate="many_to_one",
    )

    decimals = kwargs.get("decimals", 2)
    for col in target_cols:
        eval_df[f"AbsErr_{col}"] = (
            (eval_df[f"{col}_ref"] - eval_df[f"{col}_fcst"]).abs().round(decimals)
        )

    # drop the reference column, rename the forecast column to the original name, and return only the relevant columns
    eval_df = eval_df.drop(columns=[f"{col}_ref" for col in target_cols])
    eval_df = eval_df.rename(columns={f"{col}_fcst": col for col in target_cols})

    return eval_df


def SquErr(
    reference_df: pd.DataFrame, forecast_df: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    """Squared Error (SquErr) metric.
    Args:
        reference_df (pd.DataFrame): DataFrame containing the reference values.
        forecast_df (pd.DataFrame): DataFrame containing the forecast values.
        **kwargs: Additional keyword arguments

        Additional arguments:
        column_keys (dict): Dictionary mapping columns containing the initial time,
        valid time, and storm ID.

        target_cols (list): List with the target columns to be evaluated.
        Default is ['wind_max', 'pressure_min']

    Returns:
        pd.DataFrame: DataFrame containing the computed metric values.
    """

    ref_keys = _get_refkeys(
        kwargs.get(
            "column_keys",
            {
                "storm_id": "sid",
                "initial_time": "initial_time",
                "valid_time": "valid_time",
                "ensemble_idx": "ensemble_idx",
                "lead_hours": "lead_hours",
            },
        ),
        forecast_df,
    )

    target_cols = kwargs.get("target_cols", ["wind_max", "pressure_min"])

    eval_df = forecast_df.merge(
        reference_df[[*ref_keys, *target_cols]],
        on=ref_keys,
        how="left",
        suffixes=("_fcst", "_ref"),
    )

    decimals = kwargs.get("decimals", 2)
    for col in target_cols:
        eval_df[f"SquErr_{col}"] = (
            (eval_df[f"{col}_fcst"] - eval_df[f"{col}_ref"]).pow(2)
        ).round(decimals)

    # drop the reference column, rename the forecast column to the original name, and return only the relevant columns
    eval_df = eval_df.drop(columns=[f"{col}_ref" for col in target_cols])
    eval_df = eval_df.rename(columns={f"{col}_fcst": col for col in target_cols})

    return eval_df


def FairCRPS(
    reference_df: pd.DataFrame, forecast_df: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    """Fair CRPS (Leutbecher, 2019) for ensemble forecasts.

    CRPS_fair = mean_i |x_i - y| - sum_{i,j} |x_i - x_j| / (2 M (M - 1))

    The pairwise term is computed from sorted members in O(M log M):
    sum_{i,j} |x_i - x_j| = 2 * sum_k (2k - M - 1) x_(k).

    Args:
        reference_df (pd.DataFrame): DataFrame containing the reference values.
        forecast_df (pd.DataFrame): DataFrame containing the ensemble forecast values.
        **kwargs: Additional keyword arguments

        Additional arguments:
        column_keys (dict): Dictionary mapping columns containing the initial time,
        valid time, storm ID, and ensemble index.

        target_cols (list): List with the target columns to be evaluated.
        Default is ['wind_max', 'pressure_min']

    Returns:
        pd.DataFrame: One row per forecast case, with the storm ID, initial time and
        valid time columns followed by a FairCRPS_<col> column per target. NaN where
        fewer than 2 members or no reference value.
    """
    col_keys = kwargs.get(
        "column_keys",
        {
            "storm_id": "sid",
            "initial_time": "initial_time",
            "valid_time": "valid_time",
            "ensemble_idx": "ensemble_idx",
        },
    )
    ref_keys = _get_refkeys(col_keys, forecast_df)
    ens_col = col_keys.get("ensemble_idx", None)
    if ens_col is None or ens_col not in forecast_df.columns:
        raise ValueError(
            "FairCRPS requires an ensemble index column. Please provide it in column_keys."
        )

    target_cols = kwargs.get("target_cols", ["wind_max", "pressure_min"])

    eval_df = forecast_df[[*ref_keys, ens_col, *target_cols]].merge(
        reference_df[[*ref_keys, *target_cols]].drop_duplicates(subset=ref_keys),
        on=ref_keys,
        how="left",
        suffixes=("_fcst", "_ref"),
    )

    decimals = kwargs.get("decimals", 2)
    result_df = eval_df[ref_keys].drop_duplicates().reset_index(drop=True)
    for col in target_cols:
        df = (
            eval_df[ref_keys]
            .assign(x=eval_df[f"{col}_fcst"], y=eval_df[f"{col}_ref"])
            .dropna(subset=["x"])
            .sort_values([*ref_keys, "x"])
        )
        grp = df.groupby(ref_keys, sort=False)
        m = grp["x"].transform("size")
        rank = grp.cumcount() + 1
        df["abs_err"] = (df["x"] - df["y"]).abs()
        df["spread"] = (2 * rank - m - 1) * df["x"]

        agg = df.groupby(ref_keys).agg(
            abs_err=("abs_err", "mean"),
            spread=("spread", "sum"),
            m=("x", "size"),
            y=("y", "first"),
        )
        crps = agg["abs_err"] - agg["spread"] / (agg["m"] * (agg["m"] - 1))
        crps[(agg["m"] < 2) | agg["y"].isna()] = np.nan

        result_df = result_df.merge(
            crps.round(decimals).rename(f"FairCRPS_{col}").reset_index(),
            on=ref_keys,
            how="left",
        )

    return result_df


def DirPosErr(
    reference_df: pd.DataFrame, forecast_df: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    pass


def CartPosErr(
    reference_df: pd.DataFrame, forecast_df: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    pass


def TrackCRPS(
    reference_df: pd.DataFrame, forecast_df: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    """Track CRPS - CRPS with haversine distance (hd) in lieu of Euclidean
    distance. Based on the FairCRPS formulation.

    TCRPS = mean_i hd(x_i, y) - 1 / (2 M (M - 1)) * sum_{i!=j} hd(x_i, x_j)

    Additional arguments:
    min_member_frac (float): Minimum fraction of ensemble members with a valid
    position required to score a case; cases below it are NaN. The denominator
    is the number of unique ensemble indices in forecast_df. Default 30%.
    """
    radius = kwargs.get("radius", 6371.0)  # km
    col_keys = kwargs.get(
        "column_keys",
        {
            "storm_id": "sid",
            "initial_time": "initial_time",
            "valid_time": "valid_time",
            "ensemble_idx": "ensemble_idx",
        },
    )
    ref_keys = _get_refkeys(col_keys, forecast_df)
    ens_col = col_keys.get("ensemble_idx", None)
    if ens_col is None or ens_col not in forecast_df.columns:
        raise ValueError(
            "TrackCRPS requires an ensemble index column. Please provide it in column_keys."
        )

    lat_col, lon_col = kwargs.get("position_cols", ("lat", "lon"))
    decimals = kwargs.get("decimals", 2)
    chunk_size = kwargs.get("chunk_size", 2000)
    min_member_frac = kwargs.get("ensemble_fraction", 0.3)
    ensemble_size = forecast_df[ens_col].nunique()

    eval_df = forecast_df[[*ref_keys, ens_col, lat_col, lon_col]].merge(
        reference_df[[*ref_keys, lat_col, lon_col]].drop_duplicates(subset=ref_keys),
        on=ref_keys,
        how="left",
        suffixes=("_fcst", "_ref"),
    )
    df = eval_df.dropna(subset=[f"{lat_col}_fcst", f"{lon_col}_fcst"])

    grp = df.groupby(ref_keys, sort=False)
    case = grp.ngroup().to_numpy()
    slot = grp.cumcount().to_numpy()
    n_cases, M = case.max() + 1, slot.max() + 1

    # Padded (case, member) arrays; NaN marks absent/missing members
    lat = np.full((n_cases, M), np.nan)
    lon = np.full((n_cases, M), np.nan)
    lat[case, slot] = df[f"{lat_col}_fcst"].to_numpy()
    lon[case, slot] = df[f"{lon_col}_fcst"].to_numpy()

    keys = grp.size().reset_index()[ref_keys]
    y_lat = grp[f"{lat_col}_ref"].first().to_numpy()
    y_lon = grp[f"{lon_col}_ref"].first().to_numpy()

    m = np.sum(~np.isnan(lat), axis=1)
    member_frac = m / ensemble_size
    abs_err = np.nanmean(
        haversine(lat, lon, y_lat[:, None], y_lon[:, None], radius=radius), axis=1
    )

    pair_sum = np.empty(n_cases)
    for s in range(0, n_cases, chunk_size):
        la, lo = lat[s : s + chunk_size], lon[s : s + chunk_size]
        d = haversine(
            la[:, :, None],
            lo[:, :, None],
            la[:, None, :],
            lo[:, None, :],
            radius=radius,
        )
        pair_sum[s : s + chunk_size] = np.nansum(
            d, axis=(1, 2)
        )  # ordered pairs, diag = 0

    with np.errstate(divide="ignore", invalid="ignore"):
        score = abs_err - pair_sum / (2 * m * (m - 1))
    score[
        (m < 2) | (member_frac < min_member_frac) | np.isnan(y_lat) | np.isnan(y_lon)
    ] = np.nan

    keys["TrackCRPS"] = np.round(score, decimals)
    keys["member_frac"] = np.round(member_frac, decimals)

    return eval_df[ref_keys].drop_duplicates().merge(keys, on=ref_keys, how="left")
