import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from src.constants import model_data_dir, raw_data_dir, raw_data_name


def scale_cols(df: pd.DataFrame, method="mean_std", func=None) -> pd.DataFrame:
    if func is not None:
        return df.apply(func)
    else:
        if method == "mean_std":
            return df.apply(
                lambda x: (x - x.mean()) / x.std()
                if pd.api.types.is_numeric_dtype(x)
                else x
            )
        elif method == "min_max":
            return df.apply(
                lambda x: (x - x.min()) / (x.max() - x.min())
                if pd.api.types.is_numeric_dtype(x)
                else x
            )
        else:
            raise NotImplementedError(
                "`method` must be either `mean_std` or `min_max`"
            )


def import_data(
    scale: bool = True, scale_method="mean_std"
) -> Tuple[pd.DataFrame]:
    factors = pd.read_excel(raw_data_dir / raw_data_name, sheet_name=1)
    factors.set_index("Date", inplace=True)
    if scale:
        factors = scale_cols(factors, method=scale_method)
    outcomes = pd.read_excel(raw_data_dir / raw_data_name, sheet_name=3)
    outcomes["Date"] = pd.to_datetime(outcomes["Date"])
    categories = pd.read_excel(raw_data_dir / raw_data_name, sheet_name=2)

    return factors, outcomes, categories


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


def impute(
    factors: pd.DataFrame,
    categories: pd.DataFrame,
    scale: bool = True,
    scale_method="mean_std",
    return_func=None,
) -> pd.DataFrame:
    def impute_col(col: str) -> pd.DataFrame:
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

    if scale:
        df_clean = scale_cols(df_clean, method=scale_method)
    return df_clean


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
    factors, outcomes, categories = import_data(
        scale=True, scale_method="mean_std"
    )
    df_clean = impute(factors, categories, scale=True, scale_method="min_max")
    df_clean = df_clean.round(4)
    train, test = train_test_split(df_clean, outcomes, mode)

    train.to_csv(model_data_dir / f"train_{mode}.csv", index=False)
    test.to_csv(model_data_dir / f"test_{mode}.csv", index=False)
