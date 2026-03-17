import pandas as pd

from src.io_utils import ensure_columns, load_csv
from src.text_normalization import build_category_match_text


REQUIRED_CATEGORY_COLUMNS = ["master_category_id", "category_name"]


def load_categories(filepath: str) -> pd.DataFrame:
    """
    Load master category reference data.
    """
    df = load_csv(filepath)
    ensure_columns(df, REQUIRED_CATEGORY_COLUMNS, df_name="Category reference")

    if "category_path" not in df.columns:
        df["category_path"] = ""

    if "aliases" not in df.columns:
        df["aliases"] = ""

    df["master_category_id"] = df["master_category_id"].astype(str)
    df["category_name"] = df["category_name"].fillna("").astype(str)
    df["category_path"] = df["category_path"].fillna("").astype(str)
    df["aliases"] = df["aliases"].fillna("").astype(str)

    return df


def build_category_index(categories_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a category index DataFrame with normalized text
    ready for vectorization / similarity ranking.
    """
    working_df = categories_df.copy()

    working_df["match_text"] = working_df.apply(
        lambda row: build_category_match_text(
            category_name=row["category_name"],
            category_path=row["category_path"],
            aliases=row["aliases"],
        ),
        axis=1,
    )

    return working_df[
        ["master_category_id", "category_name", "category_path", "aliases", "match_text"]
    ].copy()