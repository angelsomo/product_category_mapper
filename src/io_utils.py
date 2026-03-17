from pathlib import Path
from typing import Iterable

import pandas as pd


def load_csv(filepath: str | Path, **kwargs) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.

    Parameters
    ----------
    filepath : str | Path
        Path to the CSV file.
    **kwargs
        Extra keyword arguments passed to pandas.read_csv().

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the loaded file is empty.
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path, **kwargs)

    if df.empty:
        raise ValueError(f"CSV file is empty: {path}")

    return df


def save_csv(df: pd.DataFrame, filepath: str | Path, index: bool = False) -> None:
    """
    Save a DataFrame to CSV, creating parent directories if needed.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def ensure_columns(df: pd.DataFrame, required_columns: Iterable[str], df_name: str = "DataFrame") -> None:
    """
    Validate that a DataFrame contains all required columns.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{df_name} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )