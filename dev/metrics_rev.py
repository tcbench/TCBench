# %%
import warnings

import pandas as pd
import numpy as np


# %% helper functions
def _get_refkeys(col_dict: dict, df: pd.DataFrame) -> tuple:
    column_keys = col_dict.copy()

    ref_keys = [*column_keys.values()]
    if column_keys.get("ensemble_idx", None):
        ref_keys.remove(column_keys.get("ensemble_idx", None))

    if column_keys.get("lead_hours", None):
        ref_keys.remove(column_keys.get("lead_hours", None))

    return ref_keys


def _add_lead_hours(df: pd.DataFrame, col_keys: dict | None = None) -> pd.DataFrame:
    """Return a copy of df with lead hours computed as valid time - initial time,
    placed immediately to the right of the valid time column. Any existing lead
    hours column is replaced."""
    col_keys = col_keys or {}
    it_col = col_keys.get("initial_time", "initial_time")
    vt_col = col_keys.get("valid_time", "valid_time")
    lead_col = col_keys.get("lead_hours", None) or "lead_hours"

    lead = (
        pd.to_datetime(df[vt_col]) - pd.to_datetime(df[it_col])
    ).dt.total_seconds() / 3600

    df = df.drop(columns=lead_col, errors="ignore")
    df.insert(df.columns.get_loc(vt_col) + 1, lead_col, lead.to_numpy().astype(int))
    return df


def haversine(lat1, lon1, lat2, lon2, radius=6371.0):
    """Great-circle distance in km; inputs in degrees, broadcastable."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing from point 1 to point 2, in radians clockwise
    from north; inputs in degrees, broadcastable."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    return np.arctan2(
        np.sin(dlon) * np.cos(lat2),
        np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon),
    )


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


def _to_xyz(lat, lon):
    """Unit vectors (..., 3) from latitude/longitude in degrees; NaN propagates."""
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack(
        [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1
    )


def _to_latlon(v):
    """Latitude/longitude in degrees of (..., 3) vectors; zero vectors give NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.linalg.norm(v, axis=-1)
        lat = np.degrees(np.arcsin(np.clip(v[..., 2] / norm, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(v[..., 1], v[..., 0]))
    return lat, np.where(np.isnan(lat), np.nan, lon)


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

    return _add_lead_hours(eval_df, kwargs.get("column_keys"))


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

    return _add_lead_hours(eval_df, kwargs.get("column_keys"))


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

    return _add_lead_hours(result_df, col_keys)


def DirPosErr(
    reference_df: pd.DataFrame, forecast_df: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    """Direct Position Error (DirPosErr) metric: great-circle (haversine) distance
    between the forecast and reference positions, in km.
    Args:
        reference_df (pd.DataFrame): DataFrame containing the reference values.
        forecast_df (pd.DataFrame): DataFrame containing the forecast values.
        **kwargs: Additional keyword arguments

        Additional arguments:
        column_keys (dict): Dictionary mapping columns containing the initial time,
        valid time, and storm ID.

        position_cols (tuple): Latitude and longitude columns. Default ('lat', 'lon').

        radius (float): Earth radius in km. Default 6371.

    Returns:
        pd.DataFrame: The forecast DataFrame with a DirPosErr column, in km.
    """
    radius = kwargs.get("radius", 6371.0)  # km
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

    lat_col, lon_col = kwargs.get("position_cols", ("lat", "lon"))
    decimals = kwargs.get("decimals", 2)

    eval_df = forecast_df.merge(
        reference_df[[*ref_keys, lat_col, lon_col]].drop_duplicates(subset=ref_keys),
        on=ref_keys,
        how="left",
        suffixes=("_fcst", "_ref"),
    )

    eval_df["DirPosErr"] = np.round(
        haversine(
            eval_df[f"{lat_col}_ref"].to_numpy(dtype=float),
            eval_df[f"{lon_col}_ref"].to_numpy(dtype=float),
            eval_df[f"{lat_col}_fcst"].to_numpy(dtype=float),
            eval_df[f"{lon_col}_fcst"].to_numpy(dtype=float),
            radius=radius,
        ),
        decimals,
    )

    # drop the reference columns, restore the forecast column names
    eval_df = eval_df.drop(columns=[f"{lat_col}_ref", f"{lon_col}_ref"])
    eval_df = eval_df.rename(
        columns={f"{lat_col}_fcst": lat_col, f"{lon_col}_fcst": lon_col}
    )

    return _add_lead_hours(eval_df, kwargs.get("column_keys"))


def CartPosErr(
    reference_df: pd.DataFrame, forecast_df: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    """Along-track / cross-track position error (CARTE successor).

    The forecast position is projected onto a local azimuthal-equidistant plane
    centred on the reference position E at valid time, with the y-axis along the
    reference storm motion (from the reference position S at valid time -
    motion_hours to E). The forecast offset is decomposed into:
      - along-track: positive ahead of the storm (too fast), negative behind
      - cross-track: positive right of the motion, negative left

    The projection preserves distance from E, so along^2 + cross^2 equals the
    squared haversine direct position error. Longitudes need no wrapping.

    Args:
        reference_df (pd.DataFrame): DataFrame containing the reference values.
        forecast_df (pd.DataFrame): DataFrame containing the forecast values.
        **kwargs: Additional keyword arguments

        Additional arguments:
        column_keys (dict): Dictionary mapping columns containing the initial time,
        valid time, and storm ID.

        position_cols (tuple): Latitude and longitude columns. Default ('lat', 'lon').

        track_df (pd.DataFrame): Reference track used to find S, with the storm ID,
        valid time and position columns. Default is reference_df.

        motion_hours (float): Lookback defining the storm motion. Default 6.

        min_motion_km (float): Cases where the storm moved less than this over
        motion_hours have an undefined heading and are NaN. Default 1 km.

    Returns:
        pd.DataFrame: The forecast DataFrame with CartPosErr_along and
        CartPosErr_cross columns, in km.
    """
    radius = kwargs.get("radius", 6371.0)  # km
    col_keys = kwargs.get(
        "column_keys",
        {
            "storm_id": "sid",
            "initial_time": "initial_time",
            "valid_time": "valid_time",
            "ensemble_idx": "ensemble_idx",
            "lead_hours": "lead_hours",
        },
    )
    ref_keys = _get_refkeys(col_keys, forecast_df)
    sid_col, vt_col = col_keys["storm_id"], col_keys["valid_time"]

    lat_col, lon_col = kwargs.get("position_cols", ("lat", "lon"))
    track_df = kwargs.get("track_df", reference_df)
    motion_hours = kwargs.get("motion_hours", 6)
    min_motion_km = kwargs.get("min_motion_km", 1.0)
    decimals = kwargs.get("decimals", 2)

    eval_df = forecast_df.merge(
        reference_df[[*ref_keys, lat_col, lon_col]].drop_duplicates(subset=ref_keys),
        on=ref_keys,
        how="left",
        suffixes=("_fcst", "_ref"),
    )

    # Reference position motion_hours before valid time, looked up per storm
    prev = (
        track_df[[sid_col, vt_col, lat_col, lon_col]]
        .dropna()
        .drop_duplicates(subset=[sid_col, vt_col])
        .rename(columns={vt_col: "_t_prev", lat_col: "_lat_prev", lon_col: "_lon_prev"})
    )
    eval_df["_t_prev"] = eval_df[vt_col] - pd.Timedelta(hours=motion_hours)
    eval_df = eval_df.merge(prev, on=[sid_col, "_t_prev"], how="left")

    lat_p = eval_df[f"{lat_col}_fcst"].to_numpy(dtype=float)
    lon_p = eval_df[f"{lon_col}_fcst"].to_numpy(dtype=float)
    lat_e = eval_df[f"{lat_col}_ref"].to_numpy(dtype=float)
    lon_e = eval_df[f"{lon_col}_ref"].to_numpy(dtype=float)
    lat_s = eval_df["_lat_prev"].to_numpy(dtype=float)
    lon_s = eval_df["_lon_prev"].to_numpy(dtype=float)

    # Heading of the reference motion at E (back-bearing to S, reversed)
    heading = bearing(lat_e, lon_e, lat_s, lon_s) + np.pi
    rel = bearing(lat_e, lon_e, lat_p, lon_p) - heading
    dist = haversine(lat_e, lon_e, lat_p, lon_p, radius=radius)
    motion = haversine(lat_s, lon_s, lat_e, lon_e, radius=radius)

    along = dist * np.cos(rel)
    cross = dist * np.sin(rel)
    undefined = ~(motion >= min_motion_km)  # also catches missing S or E
    along[undefined] = np.nan
    cross[undefined] = np.nan

    eval_df["AlongTrack"] = np.round(along, decimals)
    eval_df["CrossTrack"] = np.round(cross, decimals)

    # drop the reference and helper columns, restore the forecast column names
    eval_df = eval_df.drop(
        columns=[
            f"{lat_col}_ref",
            f"{lon_col}_ref",
            "_t_prev",
            "_lat_prev",
            "_lon_prev",
        ]
    )
    eval_df = eval_df.rename(
        columns={f"{lat_col}_fcst": lat_col, f"{lon_col}_fcst": lon_col}
    )

    return _add_lead_hours(eval_df, kwargs.get("column_keys"))


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

    return _add_lead_hours(
        eval_df[ref_keys].drop_duplicates().merge(keys, on=ref_keys, how="left"),
        col_keys,
    )


def ProbCircle(
    reference_df: pd.DataFrame, forecast_df: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    """Ensemble probability circles for reliability verification.

    For each case the circle is centred on the ensemble mean position (mean of
    member unit vectors, so dateline-safe). The radius for nominal probability p
    is the p-quantile of the member distances to the centre, and a case is a hit
    if the reference position lies within that radius. Averaging the hit columns
    over cases gives the observed frequency to plot against p.

    With leave_one_out, each member's distance is measured to the mean of the
    other members. Members otherwise pull the mean towards themselves and sit
    closer to it than the truth does, which makes a perfect ensemble look
    over-confident, increasingly so for small ensembles.

    Additional arguments:
    levels (sequence): Nominal probabilities. Default 0.1, 0.2, ..., 0.9.
    leave_one_out (bool): Leave-one-out member distances. Default True.
    ensemble_fraction (float): Minimum fraction of ensemble members with a valid
    position required to score a case; cases below it are NaN. The denominator
    is the number of unique ensemble indices in forecast_df. Default 30%.

    Returns:
        pd.DataFrame: One row per case with the circle centre (ProbCircle_lat,
        ProbCircle_lon), the reference distance to the centre (ProbCircle_dist),
        per level the radius ProbCircle_r<pct> in km and hit flag
        ProbCircle_hit<pct> (1/0, NaN if not scored), the fraction of member
        distances below the reference distance (ProbCircle_pit), and member_frac.
    """
    radius = kwargs.get("radius", 6371.0)  # km
    col_keys = kwargs.get(
        "column_keys",
        {
            "storm_id": "sid",
            "initial_time": "initial_time",
            "valid_time": "valid_time",
            "ensemble_idx": "ensemble_idx",
            "lead_hours": "lead_hours",
        },
    )
    ref_keys = _get_refkeys(col_keys, forecast_df)
    ens_col = col_keys.get("ensemble_idx", None)
    if ens_col is None or ens_col not in forecast_df.columns:
        raise ValueError(
            "ProbCircle requires an ensemble index column. Please provide it in column_keys."
        )

    lat_col, lon_col = kwargs.get("position_cols", ("lat", "lon"))
    decimals = kwargs.get("decimals", 2)
    levels = np.asarray(kwargs.get("levels", np.round(np.arange(0.1, 1.0, 0.1), 2)))
    leave_one_out = kwargs.get("leave_one_out", True)
    min_member_frac = kwargs.get("ensemble_fraction", 0.3)
    ensemble_size = forecast_df[ens_col].nunique()

    eval_df = forecast_df.merge(
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

    # Ensemble mean position as the normalised mean of member unit vectors
    xyz = _to_xyz(lat, lon)
    total = np.nansum(xyz, axis=1)
    c_lat, c_lon = _to_latlon(total)
    obs_dist = haversine(c_lat, c_lon, y_lat, y_lon, radius=radius)

    if leave_one_out:
        l_lat, l_lon = _to_latlon(total[:, None, :] - np.nan_to_num(xyz))
        mem_dist = haversine(lat, lon, l_lat, l_lon, radius=radius)
    else:
        mem_dist = haversine(lat, lon, c_lat[:, None], c_lon[:, None], radius=radius)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cases
        radii = np.nanquantile(mem_dist, levels, axis=1)  # (n_levels, n_cases)
    hits = (obs_dist[None, :] <= radii).astype(float)
    pit = np.sum(mem_dist < obs_dist[:, None], axis=1) / m

    invalid = (m < 2) | (member_frac < min_member_frac) | np.isnan(obs_dist)
    radii[:, invalid] = np.nan
    hits[:, invalid] = np.nan
    pit[invalid] = np.nan

    keys["ProbCircle_lat"] = np.round(c_lat, decimals)
    keys["ProbCircle_lon"] = np.round(c_lon, decimals)
    keys["ProbCircle_dist"] = np.round(obs_dist, decimals)
    for p, r, h in zip(levels, radii, hits):
        pct = int(round(p * 100))
        keys[f"ProbCircle_r{pct}"] = np.round(r, decimals)
        keys[f"ProbCircle_hit{pct}"] = h
    keys["ProbCircle_pit"] = np.round(pit, decimals)
    keys["member_frac"] = np.round(member_frac, decimals)

    return _add_lead_hours(
        eval_df[ref_keys].drop_duplicates().merge(keys, on=ref_keys, how="left"),
        col_keys,
    )


# %% Binary event (RI) metrics
def _binary_eval_df(
    reference_df: pd.DataFrame, forecast_df: pd.DataFrame, kwargs: dict
) -> tuple[pd.DataFrame, str]:
    """Merge event probabilities with observed outcomes, keeping only cases where
    both are defined. Returns the merged DataFrame and the probability column used.

    Columns: keys, lead hours, _p (probability), _o (outcome), and _m (number of
    valid members) when the count column is present.

    Probabilistic forecasts use 'RI_fraction'; deterministic forecasts fall back
    to their 0/1 'RI' column. An ensemble summary (count column present) without
    'RI_fraction' raises, rather than being scored as deterministic.
    """
    col_keys = kwargs.get(
        "column_keys",
        {
            "storm_id": "sid",
            "initial_time": "initial_time",
            "valid_time": "valid_time",
        },
    )
    ref_keys = _get_refkeys(col_keys, forecast_df)
    target_col = kwargs.get("target_col", "RI")
    count_col = kwargs.get("count_col", "n_valid")

    prob_col = kwargs.get("prob_col", None)
    if prob_col is None:
        if "RI_fraction" in forecast_df.columns:
            prob_col = "RI_fraction"
        elif count_col in forecast_df.columns:
            raise ValueError(
                f"Ensemble forecast ('{count_col}' present) has no 'RI_fraction' "
                "column; pass prob_col explicitly or keep RI_fraction "
                "(keep_intensification=True in _compute_RI)."
            )
        else:
            prob_col = "RI"
    if prob_col not in forecast_df.columns:
        raise ValueError(f"Missing probability column '{prob_col}' in forecast_df.")

    fc_cols = [*ref_keys, prob_col]
    renames = {prob_col: "_p"}
    if count_col in forecast_df.columns and count_col != prob_col:
        fc_cols.append(count_col)
        renames[count_col] = "_m"

    df = (
        forecast_df[fc_cols]
        .rename(columns=renames)
        .merge(
            reference_df[[*ref_keys, target_col]]
            .drop_duplicates(subset=ref_keys)
            .rename(columns={target_col: "_o"}),
            on=ref_keys,
            how="left",
        )
    )
    df = _add_lead_hours(df, col_keys)
    df = df[df["_p"].notna() & df["_o"].notna()]

    p = df["_p"].to_numpy(dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError(
            f"Values in '{prob_col}' fall outside [0, 1]; it is not a probability."
        )

    return df, prob_col


def _score_groups(
    df: pd.DataFrame, group_cols, score_fn, kwargs: dict | None = None
) -> pd.DataFrame:
    """Apply score_fn (DataFrame -> dict) per group, or once if group_cols is empty.

    With include_overall (default True), a final row scores all cases pooled,
    with its group columns set to overall_label (default 'all')."""
    kwargs = kwargs or {}
    if not group_cols:
        return pd.DataFrame([score_fn(df)])
    rows = []
    for key, g in df.groupby(group_cols, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(group_cols, key)), **score_fn(g)})
    if kwargs.get("include_overall", True):
        label = kwargs.get("overall_label", "all")
        rows.append({**{c: label for c in group_cols}, **score_fn(df)})
    return pd.DataFrame(rows)


def BrierSS(
    reference_df: pd.DataFrame, forecast_df: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    """Brier score and Brier skill score for a binary event (default: RI).

    BS = mean_i (p_i - o_i)^2,  BSS = 1 - BS / BS_ref

    Works for probabilistic (RI_fraction) and deterministic (0/1 RI) forecasts;
    for the latter BS is the fraction of wrong forecasts. The reference forecast
    is climatology: by default the sample base rate of each group, for which
    BS_ref = o_bar (1 - o_bar). Cases where either the probability or the
    outcome is NaN are excluded.

    Additional arguments:
    prob_col (str): Forecast probability column. Default 'RI_fraction' if present,
    else 'RI' (deterministic 0/1).
    target_col (str): Observed outcome column (1/0/NaN). Default 'RI'.
    group_cols (list): Columns to score separately. Default ['lead_hours'];
    an empty list scores all cases together.
    include_overall (bool): Append a row scoring all cases pooled, with the group
    columns set to overall_label. Default True.
    overall_label: Group value of the pooled row. Default 'all'; np.nan keeps
    the group columns numeric.
    climatology (float): Fixed reference probability, e.g. a long-term base rate,
    in lieu of the sample base rate. Default None.
    fair (bool): Use the fair Brier score for finite ensembles (Ferro, 2014),
    (p - o)^2 - p (1 - p) / (m - 1), with m from count_col (default 'n_valid').
    Ensembles only; cases with m < 2 are excluded. Default False.

    Returns:
        pd.DataFrame: One row per group with n, n_events, base_rate, BS, BS_ref,
        BSS and the probability column scored. BSS is NaN where BS_ref is 0
        (no events or all events).
    """
    group_cols = kwargs.get("group_cols", ["lead_hours"])
    climatology = kwargs.get("climatology", None)
    fair = kwargs.get("fair", False)
    decimals = kwargs.get("decimals", 4)

    df, prob_col = _binary_eval_df(reference_df, forecast_df, kwargs)
    if fair:
        if "_m" not in df.columns:
            raise ValueError(
                "Fair Brier score requires an ensemble member count column (count_col)."
            )
        df = df[df["_m"] >= 2]

    def score(g):
        p = g["_p"].to_numpy(dtype=float)
        o = g["_o"].to_numpy(dtype=float)
        n = len(o)
        bs_i = (p - o) ** 2
        if fair:
            bs_i = bs_i - p * (1 - p) / (g["_m"].to_numpy(dtype=float) - 1)
        base_rate = o.mean() if n else np.nan
        if climatology is None:
            bs_ref = base_rate * (1 - base_rate)
        else:
            bs_ref = np.mean((climatology - o) ** 2) if n else np.nan
        bs = bs_i.mean() if n else np.nan
        bss = 1 - bs / bs_ref if bs_ref > 0 else np.nan
        return {
            "n": n,
            "n_events": int(o.sum()),
            "base_rate": np.round(base_rate, decimals),
            "BS": np.round(bs, decimals),
            "BS_ref": np.round(bs_ref, decimals),
            "BSS": np.round(bss, decimals),
        }

    out = _score_groups(df, group_cols, score, kwargs)
    out["prob_col"] = prob_col
    return out


def RocAUC(
    reference_df: pd.DataFrame, forecast_df: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    """Area under the ROC curve for a binary event (default: RI).

    Computed as the Mann-Whitney statistic: the probability that a randomly
    chosen event case has a higher forecast probability than a randomly chosen
    non-event case, with ties counted as 1/2. This equals the trapezoidal area
    under the ROC curve over all probability thresholds, so it is exact for the
    discrete probabilities of a finite ensemble. For deterministic 0/1 forecasts
    it reduces to (POD + 1 - POFD) / 2. 0.5 is no skill, 1 is perfect
    discrimination. Cases where either the probability or the outcome is NaN
    are excluded.

    Additional arguments:
    prob_col (str): Forecast probability column. Default 'RI_fraction' if present,
    else 'RI' (deterministic 0/1).
    target_col (str): Observed outcome column (1/0/NaN). Default 'RI'.
    group_cols (list): Columns to score separately. Default ['lead_hours'];
    an empty list scores all cases together.
    include_overall (bool): Append a row scoring all cases pooled, with the group
    columns set to overall_label. Default True.
    overall_label: Group value of the pooled row. Default 'all'; np.nan keeps
    the group columns numeric.

    Returns:
        pd.DataFrame: One row per group with n, n_events, AUC and the probability
        column scored. AUC is NaN where a group has no events or no non-events.
    """
    group_cols = kwargs.get("group_cols", ["lead_hours"])
    decimals = kwargs.get("decimals", 4)

    df, prob_col = _binary_eval_df(reference_df, forecast_df, kwargs)

    def score(g):
        o = g["_o"].to_numpy(dtype=float) == 1
        n1 = int(o.sum())
        n0 = len(o) - n1
        if n1 == 0 or n0 == 0:
            auc = np.nan
        else:
            ranks = g["_p"].rank(method="average").to_numpy()
            auc = (ranks[o].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        return {"n": len(o), "n_events": n1, "AUC": np.round(auc, decimals)}

    out = _score_groups(df, group_cols, score, kwargs)
    out["prob_col"] = prob_col
    return out


def CSI(
    reference_df: pd.DataFrame, forecast_df: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    """Critical Success Index (threat score) for a binary event (default: RI).

    CSI = hits / (hits + misses + false alarms)

    A case is forecast as an event when its probability is >= threshold. For
    deterministic 0/1 forecasts any threshold in (0, 1] gives the same result;
    for ensembles the default matches the ens_fraction used for the RI flag.
    Correct negatives do not enter the CSI, so it is not inflated by the many
    non-RI cases, but it does depend on the base rate. POD, FAR and frequency
    bias are returned alongside it, e.g. for a performance diagram. Cases where
    either the probability or the outcome is NaN are excluded.

    Additional arguments:
    threshold (float or sequence): Probability threshold(s) for a yes forecast.
    Several thresholds give one row per threshold and group. Default 0.3.
    prob_col (str): Forecast probability column. Default 'RI_fraction' if present,
    else 'RI' (deterministic 0/1).
    target_col (str): Observed outcome column (1/0/NaN). Default 'RI'.
    group_cols (list): Columns to score separately. Default ['lead_hours'];
    an empty list scores all cases together.
    include_overall (bool): Append a row scoring all cases pooled, with the group
    columns set to overall_label. Default True.
    overall_label: Group value of the pooled row. Default 'all'; np.nan keeps
    the group columns numeric.

    Returns:
        pd.DataFrame: One row per group and threshold with the contingency counts
        (hits, misses, false_alarms, correct_negatives), POD, FAR, bias, CSI and
        the probability column scored. Ratios are NaN where their denominator is 0.
    """
    group_cols = kwargs.get("group_cols", ["lead_hours"])
    thresholds = np.atleast_1d(kwargs.get("threshold", 0.3)).astype(float)
    decimals = kwargs.get("decimals", 4)

    df, prob_col = _binary_eval_df(reference_df, forecast_df, kwargs)

    def ratio(num, den):
        return np.round(num / den, decimals) if den > 0 else np.nan

    outs = []
    for thr in thresholds:

        def score(g):
            yes_f = g["_p"].to_numpy(dtype=float) >= thr
            yes_o = g["_o"].to_numpy(dtype=float) == 1
            hits = int(np.sum(yes_f & yes_o))
            misses = int(np.sum(~yes_f & yes_o))
            false_alarms = int(np.sum(yes_f & ~yes_o))
            return {
                "n": len(yes_o),
                "hits": hits,
                "misses": misses,
                "false_alarms": false_alarms,
                "correct_negatives": int(np.sum(~yes_f & ~yes_o)),
                "POD": ratio(hits, hits + misses),
                "FAR": ratio(false_alarms, hits + false_alarms),
                "bias": ratio(hits + false_alarms, hits + misses),
                "CSI": ratio(hits, hits + misses + false_alarms),
            }

        res = _score_groups(df, group_cols, score, kwargs)
        res.insert(min(len(group_cols or []), res.shape[1]), "threshold", thr)
        outs.append(res)

    out = pd.concat(outs, ignore_index=True)
    out["prob_col"] = prob_col
    return out


# %%
