# src/data_loader.py
import pandas as pd

def parse_connections(x):
    if isinstance(x, str) and "500+" in x:
        return 500
    try:
        return int(x)
    except:
        return 0

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["job_title_n"] = df["job_title"].str.lower().fillna("")
    df["connections_num"] = df["connection"].apply(parse_connections)

    df["connections_norm"] = (
        (df["connections_num"] - df["connections_num"].min()) /
        (df["connections_num"].max() - df["connections_num"].min())
    )

    return df