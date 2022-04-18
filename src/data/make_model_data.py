import pickle as pkl
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    StandardScaler,
)
from src.constants import (
    model_data_dir,
    models_dir,
    raw_data_dir,
    raw_data_name,
)


def scale_fit(df: pd.DataFrame, method="standard", scale_func=None):
    if scale_func is not None:
        return df.apply(scale_func)

    scalers = {
        "min_max": MinMaxScaler(),
        "standard": StandardScaler(),
        "yeo_johnson": PowerTransformer(method="yeo-johnson"),
        "quantile_uniform": QuantileTransformer(),
        "quantile_normal": QuantileTransformer(output_distribution="normal"),
    }

    scaler = scalers.get(method)
    if scaler is None:
        raise NotImplementedError(
            "`method` must be either 'standard', 'yeo_johnson', or 'min_max'"
        )

    scaler = scaler.fit(df)
    df_scaled = scaler.transform(df)
    return pd.DataFrame(df_scaled, index=df.index, columns=df.columns), scaler


def save_scaler(scaler, name: str, save_dir: str = models_dir):
    with open(f"{save_dir}/{name}", "wb") as f:
        pkl.dump(scaler, f)


def import_data(scale_method="yoe_johnson"):
    factors = pd.read_excel(raw_data_dir / raw_data_name, sheet_name=1)
    factors.set_index("Date", inplace=True)
    factors, scaler = scale_fit(factors, scale_method)

    outcomes = pd.read_excel(raw_data_dir / raw_data_name, sheet_name=3)
    outcomes["Date"] = pd.to_datetime(outcomes["Date"])
    categories = pd.read_excel(raw_data_dir / raw_data_name, sheet_name=2)

    return factors, outcomes, categories, scaler


def lag_return(x: pd.Series, lag: int = 1) -> pd.Series:
    return (x - x.shift(lag)) / x.shift(lag)


def group_sc(categories) -> Dict:
    sc_group = (
        categories.groupby("Subcategory")["Variable"]
        .agg("unique")
        .reset_index()
        .to_dict("list")
    )
    return sc_group


def extract_sc(
    factors: pd.DataFrame,
    sc_group: Dict,
    exclude=["Policy Uncertainty", "Sentiment", "Inflation"],
) -> List[tuple]:
    """
    returns 2-element tuple of (df_subcategory, List[column])
    """

    return [
        (factors.loc[:, sc_group["Variable"][idx]], sc_group["Variable"][idx])
        for idx, sc in enumerate(sc_group["Subcategory"])
        if sc not in exclude
    ]


def clean_imputed_df(
    df: pd.DataFrame,
    cols_to_fill=["Global Inflation-linked debt", "S&P 500 VRP"],
) -> pd.DataFrame:
    """
    final clean up of imputed dataframe
    """
    # convert inf as na
    df = df.replace([-np.inf, np.inf], np.nan)
    # drop first and last rows
    # forward fill remaining nas
    df = df.loc[~df.index.isin(["2000-05-30", "2021-06-30"])].fillna(
        method="ffill"
    )
    for col in cols_to_fill:
        df.loc[df[col].isna(), col] = (
            df.drop([col], axis=1).loc[df[col].isna(), :].mean(axis=1)
        )

    return df


def impute(
    factors: pd.DataFrame,
    categories: pd.DataFrame,
    scale_method="min_max",
    return_func=None,
):
    def impute_col(col: str) -> pd.Series:
        """
        Impute column with mean return values in its subcategory
        """
        if (has_return_func := return_func) is None:
            mean_returns = df_sc.pct_change().median(axis=1)
        else:
            mean_returns = df_sc.apply(return_func).median(axis=1)
        df_col = df_sc.loc[:, col].copy()
        idx_non_missing = df_col.notnull()
        df_col.loc[~idx_non_missing] = mean_returns.loc[~idx_non_missing]
        if has_return_func:
            df_col.loc[idx_non_missing] = return_func(df_col).loc[
                idx_non_missing
            ]
        else:
            df_col.loc[idx_non_missing] = df_col.pct_change().loc[
                idx_non_missing
            ]
        return df_col

    sc_group = group_sc(categories)
    sc_col_pairs = extract_sc(factors, sc_group)

    cols_imputed = []

    for df_sc, cols in sc_col_pairs:
        for col in cols:
            col_imputed = impute_col(col)
            cols_imputed.append(col_imputed)

    df_imputed = pd.concat(cols_imputed, axis=1)
    df_clean = clean_imputed_df(df_imputed)

    df_clean, scaler = scale_fit(df_clean, scale_method)

    return df_clean, scaler


def train_test_split(
    df: pd.DataFrame,
    outcomes: pd.DataFrame,
    mode: str = "regression",
    split_date: str = "2013-12-31",
) -> pd.DataFrame:
    if mode == "regression":
        usecols = [
            "Date",
            "sc_1d_fwd_rel_ret",
            "mom_1d_fwd_rel_ret",
            "value_1d_fwd_rel_ret",
        ]
    elif mode == "classification":
        usecols = [
            "Date",
            "sc_1d_fwd_rel_d",
            "mom_1d_fwd_rel_d",
            "value_1d_fwd_rel_d",
        ]
    else:
        raise NotImplementedError(
            "`mode` should either be 'regression' or 'classification'"
        )

    outcomes = outcomes[usecols].iloc[1:, :]

    df_all = df.merge(outcomes, on="Date")

    return df_all.query(f"Date <= '{split_date}'"), df_all.query(
        f"Date > '{split_date}'"
    )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "regression"
    scale_method_first = "quantile_normal"
    scale_method_second = "quantile_uniform"
    factors, outcomes, categories, scaler_first = import_data(
        scale_method=scale_method_first
    )
    df_clean, scaler_second = impute(
        factors, categories, scale_method=scale_method_second
    )
    df_clean = df_clean.round(4).reset_index()
    train, test = train_test_split(df_clean, outcomes, mode)

    train.to_csv(model_data_dir / f"train_{mode}.csv", index=False)
    test.to_csv(model_data_dir / f"test_{mode}.csv", index=False)
    save_scaler(scaler_first, f"scaler_first_{scale_method_first}.pkl")
    save_scaler(scaler_second, f"scaler_second_{scale_method_second}.pkl")
