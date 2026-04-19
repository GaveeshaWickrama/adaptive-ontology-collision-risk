from pathlib import Path
import pandas as pd


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_csv(path: str) -> pd.read_csv:
    return pd.read_csv(path)


def write_csv(df, path: str, index: bool = False) -> None:
    ensure_parent_dir(path)
    df.to_csv(path, index=index)