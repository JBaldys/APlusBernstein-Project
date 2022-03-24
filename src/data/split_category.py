import re
import sys
from typing import Literal, Union

import dask as dk
import dask.dataframe as dd
import pandas as pd
import pyarrow
from src.constants import processed_data_dir, raw_data_dir, raw_data_name


def read_sheet(sheet: int):
    return dk.delayed(pd.read_excel)(raw_data_dir / raw_data_name, sheet_name=sheet)


def extract_category(ddf_all, category: str, by: str):
    p = re.compile(r" |&|/")
    ddf = ddf_all.loc[ddf_all[by] == category, ["Date", "Variable", "value"]]
    ddf = ddf.pivot(index="Date", columns="Variable", values="value")
    category = p.sub("-", category, 1).replace("& ", "").lower()
    return ddf.to_parquet(
        processed_data_dir / "split" / by.lower() / f"{category}.parquet"
    )


def split_df(by: Union[Literal["Category"], Literal["Subcategory"]] = "Subcategory"):
    if by not in ["Category", "Subcategory"]:
        print("Invalid split option, must be either 'Category' or 'Subcategory'")
    ddf_factors = read_sheet(1)
    ddf_categories = read_sheet(2)
    ddf_factors_long = ddf_factors.melt(
        id_vars=["Date"], var_name="Variable", value_name="value"
    )
    ddf_all = ddf_factors_long.merge(ddf_categories, on="Variable", how="left")
    categories = ddf_categories[by].unique().compute()
    ddfs = [extract_category(ddf_all, category, by) for category in categories]

    dk.compute(*ddfs)
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        split_df(sys.argv[1])
    else:
        split_df()
