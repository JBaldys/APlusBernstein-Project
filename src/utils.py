import itertools as it
from typing import Dict, List, Union

import numpy as np
import pandas as pd


def group_columns(categories: pd.DataFrame, by: str, all_factors: List[str]) -> Dict:
    """
    group column names by subcategory or category
    returns Dict[category_name, List[factor_names]]
    """

    ret = categories.groupby(by)["Variable"].unique().to_dict()
    for cat in ret:
        cols = [col.strip() for col in ret[cat] if col.strip() in all_factors]
        ret[cat] = cols

    return ret


def add_returns(
    df: pd.DataFrame, cols: List[str] = None, return_func=None, **kwargs
) -> pd.DataFrame:
    df = df.copy(deep=True)
    cols = df.columns if cols is None else cols

    if return_func is None:
        returns = [
            pd.Series(df[col].pct_change(fill_method=None, **kwargs), name=f"{col}_ret")
            for col in cols
        ]
    else:
        returns = [return_func(df[col], **kwargs) for col in cols]

    return pd.concat([df] + returns, axis=1)


def add_seasonal_features(
    df: pd.DataFrame, dates: Union[pd.Series, pd.Index], dummy: bool = True
):
    df = df.copy(deep=True)
    dates = pd.Series(pd.to_datetime(dates))
    # `values`` ignore index alignment
    df["month"] = dates.dt.month.values
    df["day"] = dates.dt.day.values
    df["weekday"] = dates.dt.weekday.values

    return df


def use_dummy_seasonal(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy(deep=True)
    df = pd.get_dummies(
        df.astype(
            {
                "year": "category",
                "month": "category",
                "day": "category",
                "weekday": "category",
            }
        ),
        drop_first=True,
    )

    return df


def add_moving_averages(
    df: pd.DataFrame,
    windows: List[int],
    cols: List[str] = None,
):
    df = df.copy(deep=True)
    cols = df.columns if cols is None else cols
    moving_averages = [
        pd.Series(df[col].rolling(w).mean(), name=f"{col}_mv_{w}")
        for col in cols
        for w in windows
    ]

    return pd.concat([df] + moving_averages, axis=1)


def add_rolling_vols(
    df: pd.DataFrame, windows: List[int], cols: List[str], days_in_year: int = 252
) -> pd.DataFrame:
    df = df.copy(deep=True)
    vols = [
        pd.Series(
            df[col].rolling(w).std(ddof=1) * np.sqrt(days_in_year),
            name=f"{col}_rolling_{w}",
        )
        for w in windows
        for col in cols
    ]

    return pd.concat([df] + vols, axis=1)


def combine_classes(
    df: pd.DataFrame,
    cols: List[str] = ["sc_1d_fwd_rel_d", "mom_1d_fwd_rel_d", "value_1d_fwd_rel_d"],
    target_name: str = "y",
    drop: bool = True,
) -> pd.DataFrame:
    if not pd.Serise(cols).isin(df.columns).all():
        raise ValueError(f"not all {cols} are in dataframe `df`")

    def combine_classes_row(row: pd.Series) -> int:
        combo = row["sc_1d_fwd_rel_d"], row["mom_1d_fwd_rel_d"], row["value_1d_fwd_rel_d"]
        return combos_map.get(combo)

    df = df.copy(deep=True)
    combos = list(it.product([0, 1], repeat=3))
    combos.reverse()
    combos_map = {combos[k]: k for k in range(0, len(combos))}
    df[target_name] = df.apply(combine_classes_row, axis=1)
    if drop:
        df = df.drop(cols, axis=1)

    return df


def use_target(
    df: pd.DataFrame, col: str, mode: str, next_day: bool = False, split: bool = True
) -> pd.DataFrame:
    if mode == "regression":
        cols = [
            "sc_1d_fwd_rel_ret",
            "mom_1d_fwd_rel_ret",
            "value_1d_fwd_rel_ret",
        ]
    elif mode == "classification":
        cols = [
            "sc_1d_fwd_rel_d",
            "mom_1d_fwd_rel_d",
            "value_1d_fwd_rel_d",
        ]
    else:
        raise NotImplementedError(
            "`mode` should either be 'regression' or 'classification'"
        )

    if col not in cols:
        raise ValueError(f"`col` should be one of {cols}")

    df = df.copy(deep=True)
    cols_to_drop = [c for c in cols if c != col and c in df.columns]
    df = df.drop(cols_to_drop, axis=1)

    if next_day:
        df = use_next_day(df, col)

    if split:
        new_col_name = f"{col}_next" if next_day else col
        return df.pop(new_col_name), df
    else:
        return df


def use_next_day(
    df: pd.DataFrame,
    col: str,
    drop_original_col=True,
    drop_last_row=True,
) -> pd.DataFrame:
    df = df.copy(deep=True)
    df[f"{col}_next"] = df[col].shift(-1)
    if drop_original_col:
        df = df.drop(col, axis=1)
    if drop_last_row:
        df = df.drop(df.index[-1])
    return df


class BlockingTimeSeriesSplit:
    """(ordered) time-series cross-validation iterator"""

    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.5 * (stop - start)) + start
            yield indices[start:mid], indices[mid + margin : stop]


def split_folds(df: pd.DataFrame, n_folds: int = 5):
    n_samples = df.shape[0]
    fold_size = n_samples // n_folds
    indices = np.arange(n_samples)
    for i in range(n_folds):
        start = i * fold_size
        if (start + fold_size) > (n_samples - 1):
            stop = n_samples - 1
        else:
            stop = start + fold_size

        out_indices = indices[start:stop].tolist()
        in_indices = np.concatenate([indices[0:start], indices[stop:]]).tolist()
        yield in_indices, out_indices
