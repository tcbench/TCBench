import pandas as pd


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

    column_keys = kwargs.get(
        "column_keys",
        {
            "storm_id": "sid",
            "initial_time": "initial_time",
            "valid_time": "valid_time",
            "ensemble_idx": "ensemble_idx",
        },
    )

    if (
        column_keys.get("ensemble_idx")
        and column_keys.get("ensemble_idx") not in forecast_df.columns
    ):
        column_keys.pop("ensemble_idx", None)

    ref_keys = [*column_keys.values()]
    ref_keys.remove(column_keys.get("ensemble_idx", None))

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
    """Squared Error (RMeanSquErr) metric.
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

    column_keys = kwargs.get(
        "column_keys",
        {
            "storm_id": "sid",
            "initial_time": "initial_time",
            "valid_time": "valid_time",
            "ensemble_idx": "ensemble_idx",
        },
    )

    if (
        column_keys.get("ensemble_idx")
        and column_keys.get("ensemble_idx") not in forecast_df.columns
    ):
        column_keys.pop("ensemble_idx", None)

    ref_keys = [*column_keys.values()]
    ref_keys.remove(column_keys.get("ensemble_idx", None))

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
    pass


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
    pass
