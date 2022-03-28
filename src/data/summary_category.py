import itertools as it
from pathlib import Path

import pandas as pd
from dask import compute, delayed
from src.constants import processed_data_dir

summary_dir = processed_data_dir / "summary"
category_split_dir = Path(processed_data_dir) / "split" / "category"
subcategory_split_dir = Path(processed_data_dir) / "split" / "subcategory"


def list_category_files():

    files = it.chain(
        category_split_dir.glob("*.parquet"),
        subcategory_split_dir.glob("*.parquet"),
    )

    return files


@delayed
def summarize_category(file: str):
    aggs = pd.read_parquet(file).agg(
        [
            "min",
            "max",
            "mean",
            "median",
            "std",
            "skew",
            "kurt",
            lambda col: col.isnull().mean(),
        ]
    )

    aggs.index = [
        "min",
        "max",
        "mean",
        "median",
        "std",
        "skewness",
        "kurtosis",
        "pct_missing",
    ]

    group = file.parents[0].name
    c = file.stem
    return aggs.to_csv(summary_dir / f"{group}/{c}.csv")


if __name__ == "__main__":
    out = [summarize_category(file) for file in list_category_files()]
    compute(**out)
