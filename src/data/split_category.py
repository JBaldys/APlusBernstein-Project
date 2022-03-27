import re
import sys
from typing import Literal, Union

import dask as dk
import pandas as pd
from src.constants import processed_data_dir, raw_data_dir, raw_data_name


def read_sheet(sheet: int):
    return dk.delayed(pd.read_excel)(
        raw_data_dir / raw_data_name, sheet_name=sheet
    )


def extract_category(ddf_factors, ddf_categories, category: str, by: str):
    p = re.compile(r" |&|/")
    factors = ddf_categories.loc[ddf_categories[by] == category, "Variable"]
    ddf = ddf_factors[factors]
    category = (
        p.sub("-", category, 1).replace("& ", "").replace(" ", "-").lower()
    )
    return ddf.to_parquet(
        processed_data_dir / "split" / by.lower() / f"{category}.parquet"
    )


def split_df(
    by: Union[Literal["Category"], Literal["Subcategory"]] = "Subcategory"
):
    if by not in ["Category", "Subcategory"]:
        raise Exception(
            "Invalid split option, must be either 'Category' or 'Subcategory'"
        )
    ddf_factors = read_sheet(1)
    ddf_categories = read_sheet(2)
    categories = ddf_categories[by].unique().compute()
    ddfs = [
        extract_category(ddf_factors, ddf_categories, category, by)
        for category in categories
    ]

    dk.compute(*ddfs)
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        split_df(sys.argv[1])
    else:
        split_df()
