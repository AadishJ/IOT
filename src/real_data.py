"""Load the Intel Berkeley Lab Data text file."""
from pathlib import Path
import pandas as pd

COLUMNS = ["date", "time", "epoch", "moteid", "temperature", "humidity", "light", "voltage"]

def load_intel_data(path: str | Path, sensor_id: int | None = None,
                    variable: str = "temperature") -> pd.DataFrame:
    """Load whitespace-separated Intel Lab readings, clean them, and sort by time.

    The raw format is date time epoch moteid temperature humidity light voltage.
    Missing sensor readings are commonly represented by NA.
    """
    path = Path(path)
    if variable not in {"temperature", "humidity", "light", "voltage"}:
        raise ValueError("variable must be temperature, humidity, light, or voltage")
    df = pd.read_csv(
        path, sep=r"\s+", header=None, names=COLUMNS,
        na_values=["NA", "NaN", "nan"], on_bad_lines="skip",
        dtype={"date": "string", "time": "string"},
        engine="python"
    )
    df["timestamp"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        errors="coerce"
    )
    for col in ["epoch", "moteid", "temperature", "humidity", "light", "voltage"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "moteid", variable])
    df["moteid"] = df["moteid"].astype(int)
    if sensor_id is not None:
        df = df[df["moteid"] == sensor_id]
    df = df.sort_values(["moteid", "timestamp"]).drop_duplicates(
        subset=["moteid", "timestamp"], keep="last"
    )
    return df[["timestamp", "moteid", variable, "voltage"]].rename(
        columns={variable: "value"}
    ).reset_index(drop=True)
