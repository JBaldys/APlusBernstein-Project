import pandas as pd
from src.constants import processed_data_dir
from src.data.split_category import read_sheet


def remove_missing(df: pd.DataFrame) -> pd.DataFrame:
    df_missing = df.isna().mean().sort_values(ascending=False).reset_index()
    df_missing.columns = ["factor", "missing_pct"]

    names_to_remove = set(df_missing[df_missing["missing_pct"] > 0.9]["factor"])
    return df[list(set(df.columns) - names_to_remove)]


if __name__ == "__main__":
    df = read_sheet(1).compute()
    df = remove_missing(df)
    df.to_csv(processed_data_dir / "df_removed_missing.csv")
