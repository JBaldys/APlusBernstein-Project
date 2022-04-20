from typing import Union

import pandas as pd


def add_seasonal_features(df: pd.DataFrame, dates: Union[pd.Series, pd.Index]):
    df = df.copy(deep=True)
    dates = pd.Series(pd.to_datetime(dates))
    # `values`` ignore index alignment
    df["month"] = dates.dt.month.values
    df["year"] = dates.dt.year.values
    df["day"] = dates.dt.day.values
    df["weekday"] = dates.dt.weekday.values

    return df
