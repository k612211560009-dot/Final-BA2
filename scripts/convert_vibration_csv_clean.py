"""
Clean `vibration_dataset.csv` and write to data/processed/vibration_dataset_clean.csv
This script normalizes timestamp column names and attempts to infer sensor/equipment ids.
"""
import os
import pandas as pd
import pathlib


def ensure_dir(d):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)


def main():
    base = os.path.join(os.path.dirname(__file__), "..")
    src = os.path.join(base, "vibration_dataset.csv")
    out_dir = os.path.join(base, "data", "processed")
    ensure_dir(out_dir)
    if not os.path.exists(src):
        print("vibration_dataset.csv not found at", src)
        return
    df = pd.read_csv(src)
    # Find timestamp-like column
    ts_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    if ts_cols:
        df["timestamp"] = pd.to_datetime(df[ts_cols[0]], errors="coerce")
    else:
        # try common names
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        else:
            # if no timestamp, create index based time
            df["timestamp"] = pd.NaT

    # If sensor_id column exists, keep it; else try to create from columns
    if "sensor_id" not in df.columns and "sensor" in df.columns:
        df.rename(columns={"sensor": "sensor_id"}, inplace=True)

    out = os.path.join(out_dir, "vibration_dataset_clean.csv")
    df.to_csv(out, index=False)
    print("Wrote cleaned vibration dataset to", out)


if __name__ == "__main__":
    main()
