import pickle as pkl
import sys
from random import choice
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    StandardScaler,
)
from src.constants import model_data_dir, models_dir, raw_data_dir, raw_data_name
from src.utils import add_seasonal_features


def scale_fit(df: pd.DataFrame, method="standard", scale_func=None):
    if scale_func is not None:
        return df.apply(scale_func), None

    if method is None:
        return df, None

    scalers = {
        "min_max": MinMaxScaler(),
        "standard": StandardScaler(),
        "yeo_johnson": PowerTransformer(method="yeo-johnson"),
        "quantile_uniform": QuantileTransformer(output_distribution="uniform"),
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


def ffill_factors(factors: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Fill missing values with forward-filling
    """
    factors = factors.copy(deep=True)
    for col in cols:
        factors[col] = factors[col].fillna(method="ffill")
    return factors


def drop_missing_cols(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """
    Drop columns with missing values
    """
    missing_pcts = factors.apply(np.isnan).mean()
    cols_to_drop = missing_pcts[missing_pcts > threshold].index.tolist()
    return df.drop(cols_to_drop, axis=1)


def import_data(
    add_seasonality: bool = False,
):
    factors = pd.read_excel(raw_data_dir / raw_data_name, sheet_name=1)
    factors = factors.set_index("Date")

    outcomes = pd.read_excel(raw_data_dir / raw_data_name, sheet_name=3)
    outcomes["Date"] = pd.to_datetime(outcomes["Date"])
    if add_seasonality:
        factors = add_seasonal_features(factors, factors.index)
        outcomes = add_seasonal_features(outcomes, outcomes["Date"])

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
    exclude=[
        "Policy Uncertainty",
        "Sentiment",
        "Inflation",
    ],
) -> List[tuple]:
    """
    returns list of 2-element tuple of (df_subcategory, List[column_name])
    """

    ret = []
    for idx, sc in enumerate(sc_group["Subcategory"]):
        if sc not in exclude:
            cols = [col for col in factors.columns if col in sc_group["Variable"][idx]]
            df_subcategory = factors.loc[:, cols]
            ret.append((df_subcategory, cols))

    return ret


def use_returns(df: pd.DataFrame, return_func=None, **kwargs) -> pd.DataFrame:
    df = df.copy(deep=True)
    if return_func is None:
        return df.pct_change(fill_method=None)
    else:
        return df.apply(return_func, **kwargs)


def impute(
    factors: pd.DataFrame,
    categories: pd.DataFrame,
    random: bool = True,  # random selection of imputed values
):
    def impute_col(df_sc: pd.DataFrame, col: str) -> pd.Series:
        """
        Impute column with median return values in its subcategory
        """
        if random:
            values = df_sc.apply(choice, axis=1)
        else:
            values = df_sc.median(axis=1)
        values.update(df_sc[col])
        return pd.Series(values, name=col)

    sc_group = group_sc(categories)
    sc_col_pairs = extract_sc(factors, sc_group)
    cols_imputed = []

    for df_sc, cols in sc_col_pairs:
        for col in cols:
            if col in factors.columns:
                col_imputed = impute_col(df_sc, col)
                cols_imputed.append(col_imputed)

    return pd.concat(cols_imputed, axis=1)


def replace_inf(df: pd.DataFrame, factor: int = 2) -> pd.DataFrame:
    def replace_inf_col(col):
        m = np.max(col[~np.isinf(col)])
        return col.replace([np.inf, -np.inf], [m * factor, -m * factor])

    return df.apply(replace_inf_col)


# def clean_imputed_df(
#     df: pd.DataFrame,
#     # cols_to_fill=[
#     #     "S&P 500 VRP",
#     #     "S&P 500 Price-to-Earnings",
#     #     "P/B",
#     #     "US Value P/E over Growth P/E",
#     #     "US Value P/B over Growth P/B",
#     #     "EquityBond premia",
#     # ],
#     replace_inf: bool = True,
# ) -> pd.DataFrame:
#     """
#     final clean up of imputed dataframe
#     """
#     # drop first and last rows
#     # forward fill remaining nas
#     df = df.loc[~df.index.isin(["2000-05-30", "2021-06-30"])].fillna(method="ffill")
#     # for col in cols_to_fill:
#     #     df.loc[df[col].isna(), col] = df.loc[df[col].isna(), :].median(axis=1)

#     return df


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

    return df_all.query(f"Date <= '{split_date}'"), df_all.query(f"Date > '{split_date}'")


if __name__ == "__main__":
    # configuration options
    mode = sys.argv[1] if len(sys.argv) > 1 else "regression"
    use_return = True
    scale_method_first = None
    scale_method_second = "quantile_normal"
    ffill = True
    factors_ffill = [
        "Global Inflation-linked debt",
        "Citi US Inflation Surprise Index",
        "Global Economic Policy Uncertainty",
        "US Trade Policy Uncertainty",
        "US Monetary Policy Uncertainty",
        "US Economic Policy Uncertainty",
        "EU Economic Policy Uncertainty",
        "China Economic Policy Uncertainty",
        "US ISM Service PMI",
        "US ISM Manufacturing PMI",
        "Conference Board Consumer Confidence",
        "University of Michigan Consumer Sentiment",
        "EU Consumer Confidence",
    ]
    add_seasonality = True

    factors, outcomes, categories = import_data(add_seasonality)
    if ffill:
        factors = ffill_factors(factors, factors_ffill)
    factors = drop_missing_cols(factors, threshold=0.5)

    factors, scaler_first = scale_fit(factors, method=scale_method_first)

    if use_return:
        factors = use_returns(factors)

    factors_imputed = impute(factors, categories)
    factors.update(factors_imputed)
    factors = replace_inf(factors)

    # drop first and last rows
    # forward fill remaining nas
    factors = factors.iloc[1 : (factors.shape[0] - 1)].fillna(method="ffill")
    factors = factors.fillna(method="bfill")
    factors, scaler_second = scale_fit(factors, method=scale_method_second)
    factors = factors.round(4).reset_index()
    train, test = train_test_split(factors, outcomes, mode)

    train.to_csv(model_data_dir / f"train_{mode}.csv", index=False)
    test.to_csv(model_data_dir / f"test_{mode}.csv", index=False)

    # if scaler_first is not None:
    #     save_scaler(scaler_first, f"scaler_first_{scale_method_first}.pkl")
    # if scaler_second is not None:
    #     save_scaler(scaler_second, f"scaler_second_{scale_method_second}.pkl")
