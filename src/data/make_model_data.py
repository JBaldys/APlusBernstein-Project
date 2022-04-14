import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from src.constants import model_data_dir, raw_data_dir, raw_data_name


def import_data(normalize_factors: bool = True) -> Tuple[pd.DataFrame]:
    factors = pd.read_excel(raw_data_dir / raw_data_name, sheet_name=1)
    factors.set_index("Date", inplace=True)
    if normalize_factors:
        factors = factors.apply(lambda x: (x - x.mean()) / x.std())
    outcomes = pd.read_excel(raw_data_dir / raw_data_name, sheet_name=3)
    outcomes["Date"] = pd.to_datetime(outcomes["Date"])
    categories = pd.read_excel(raw_data_dir / raw_data_name, sheet_name=2)

    return factors, outcomes, categories


def lag_return(x: pd.Series, lag: int = 1) -> pd.Series:
    """
    Calculate factor returns, in percent scale
    """
    r = (x - x.shift(lag)) / x.shift(lag)
    return pd.Series(np.round(r * 100, 3))


def impute_col(
    df_sc: pd.DataFrame,
    col: str,
) -> pd.DataFrame:
    """
    impute column with mean return in its subcategory
    """
    mean_returns = df_sc.apply(lag_return).mean(axis=1)
    df_col = df_sc.loc[:, col].copy()
    idx_non_missing = df_col.notnull()
    df_col.loc[~idx_non_missing] = mean_returns.loc[~idx_non_missing]
    df_col.loc[idx_non_missing] = lag_return(df_col).loc[idx_non_missing]
    return df_col


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


def clean_imputed_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    final clean up of imputed dataframe
    """
    # convert inf as na
    df = df.reset_index().replace([-np.inf, np.inf], np.nan)
    # drop first and last rows
    # forward fill remaining nas
    df = df.loc[~df["Date"].isin(["2000-05-30", "2021-06-30"])].fillna(
        method="ffill"
    )
    for col in ["Global Inflation-linked debt", "S&P 500 VRP"]:
        df.loc[df[col].isna(), col] = (
            df.drop(["Date", col], axis=1).loc[df[col].isna(), :].mean(axis=1)
        )

    return df


def impute(factors: pd.DataFrame, categories: pd.DataFrame) -> pd.DataFrame:
    sc_group = group_sc(categories)
    sc_col_pair = extract_sc(factors, sc_group)

    cols_imputed = []

    for df_sc, cols in sc_col_pair:
        for col in cols:
            col_imputed = impute_col(df_sc, col)
            cols_imputed.append(col_imputed)

    df_imputed = pd.concat(cols_imputed, axis=1)
    df_clean = clean_imputed_df(df_imputed)
    return df_clean


def train_test_split(
    df: pd.DataFrame, outcomes: pd.DataFrame, mode: str = "regression"
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

    outcomes = outcomes[usecols].query("Date != '2000-05-30'")

    if mode == "regression":
        outcomes = outcomes.apply(
            lambda x: x * 100 if x.dtype == np.float64 else x
        )

    df_all = df.merge(outcomes, on="Date")

    return df_all.query("Date <= '2013-12-31'"), df_all.query(
        "Date > '2013-12-31'"
    )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "regression"
    factors, outcomes, categories = import_data()
    df_clean = impute(factors, categories)
    train, test = train_test_split(df_clean, outcomes, mode)

    train.to_csv(model_data_dir / f"train_{mode}.csv", index=False)
    test.to_csv(model_data_dir / f"test_{mode}.csv", index=False)
