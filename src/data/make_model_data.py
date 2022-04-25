import pickle as pkl
import sys
from random import choice
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from bleach import clean
from janitor import clean_names
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    StandardScaler,
)
from src.constants import model_data_dir, models_dir, raw_data_dir, raw_data_name
from src.utils import (
    add_moving_averages,
    add_returns,
    add_rolling_vols,
    add_seasonal_features,
    group_columns,
)


def scale_fit(df: pd.DataFrame, method="standard", scale_func=None):
    if scale_func is not None:
        for col in df.columns:
            df[col] = scale_func(df[col])
            return df

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


def fill_factors(factors: pd.DataFrame, cols: List[str], method="ffill") -> pd.DataFrame:
    """
    Fill missing values with forward-filling
    """
    for col in cols:
        factors[col] = factors[col].fillna(method=method)
    return factors


def drop_missing_cols(factors: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Drop columns with missing values
    """
    missing_pcts = np.isnan(factors).mean()
    cols_to_drop = missing_pcts[missing_pcts > threshold].index.tolist()
    return factors.drop(cols_to_drop, axis=1)


def select_rows(factors: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    return factors.iloc[start : end + 1]


def import_data():
    factors = pd.read_excel(raw_data_dir / raw_data_name, sheet_name=1)
    factors = factors.set_index("Date")

    outcomes = pd.read_excel(raw_data_dir / raw_data_name, sheet_name=3)
    outcomes["Date"] = pd.to_datetime(outcomes["Date"])

    categories = pd.read_excel(raw_data_dir / raw_data_name, sheet_name=2)

    return factors, outcomes, categories


def lag_return(x: pd.Series, lag: int = 1) -> pd.Series:
    return (x - x.shift(lag)) / x.shift(lag)


def impute(
    factors: pd.DataFrame,
    columns_by_subcategory: Dict[str, List[str]],
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

    cols_imputed = []

    for cols in columns_by_subcategory.values():
        df_sc = factors.loc[:, cols]
        for col in cols:
            col_imputed = impute_col(df_sc, col)
            cols_imputed.append(col_imputed)

    return pd.concat(cols_imputed, axis=1)


def replace_inf(df: pd.DataFrame, factor: int = 2) -> pd.DataFrame:
    def replace_inf_col(col):
        m = np.max(col[~np.isinf(col)])
        return col.replace([np.inf, -np.inf], [m * factor, -m * factor])

    return df.apply(replace_inf_col)


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
    add_return = True
    scale_method_first = None
    scale_method_second = "standard"
    ffill = True
    # monthly factors
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
    moving_average_windows = [10, 40]
    vol_windows = [10, 40]

    factors, outcomes, categories = import_data()
    if ffill:
        factors = fill_factors(factors, factors_ffill, "ffill")

    factors = drop_missing_cols(factors, threshold=0.5)
    all_factors = factors.columns.tolist()
    columns_by_category = group_columns(categories, "Category", all_factors)
    columns_by_subcategory = group_columns(categories, "Subcategory", all_factors)
    market_factors = columns_by_category["Market Variables"]
    vol_factors = columns_by_category["Options & Volatility"]
    factors_imputed = impute(factors, columns_by_subcategory)
    factors.update(factors_imputed)
    factors = replace_inf(factors)
    factors = fill_factors(factors, factors.columns, "ffill")
    factors = fill_factors(factors, factors.columns, "bfill")

    if add_return:
        factors = add_returns(factors, market_factors)

    if moving_average_windows:
        factors = add_moving_averages(factors, moving_average_windows, market_factors)

    if vol_windows:
        factors = add_rolling_vols(factors, vol_windows, vol_factors)

    rows = factors.shape[0]
    factors = select_rows(
        factors,
        start=np.max(moving_average_windows + vol_windows),
        end=rows - 1,
    )

    # remaining NAs are caused by consecutive missing values on top

    factors, scaler_second = scale_fit(factors, method=scale_method_second)
    if add_seasonality:
        factors = add_seasonal_features(factors, factors.index)
        outcomes = add_seasonal_features(outcomes, outcomes["Date"])
    factors = factors.round(4).reset_index()
    train, test = train_test_split(factors, outcomes, mode)
    train = clean_names(train)
    test = clean_names(test)
    train.to_csv(model_data_dir / f"train_{mode}.csv", index=False)
    test.to_csv(model_data_dir / f"test_{mode}.csv", index=False)

    # if scaler_first is not None:
    #     save_scaler(scaler_first, f"scaler_first_{scale_method_first}.pkl")
    # if scaler_second is not None:
    #     save_scaler(scaler_second, f"scaler_second_{scale_method_second}.pkl")
