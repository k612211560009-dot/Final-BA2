"""
Clean and enrich `market_pipe_thickness_loss_dataset.csv` with corrosion_rate and remaining life.
Writes to data/processed/market_pipe_thickness_loss_dataset_clean.csv
"""
import os
import pandas as pd
import numpy as np
import pathlib


def ensure_dir(d):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)


def main():
    base = os.path.join(os.path.dirname(__file__), "..")
    src = os.path.join(base, "market_pipe_thickness_loss_dataset.csv")
    out_dir = os.path.join(base, "data", "processed")
    ensure_dir(out_dir)
    if not os.path.exists(src):
        print("market_pipe_thickness_loss_dataset.csv not found at", src)
        return
    df = pd.read_csv(src)

    # normalize column names
    cols = {c: c.strip().lower().replace(' ', '_') for c in df.columns}
    df.rename(columns=cols, inplace=True)

    # expected fields: measurement_date, original_thickness, current_thickness
    date_cols = [c for c in df.columns if 'date' in c]
    if date_cols:
        df['measurement_date'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
    # compute thickness loss and corrosion rate if possible
    if 'original_thickness' in df.columns and 'current_thickness' in df.columns:
        df['thickness_loss_mm'] = df['original_thickness'] - df['current_thickness']
        # compute corrosion_rate if measurement_date and original_date known; fallback to NaN
        df = df.sort_values('measurement_date')
        # approximate corrosion rate by differencing over time per segment/location if available
        if 'location' in df.columns:
            df['corrosion_rate_mm_year'] = np.nan
            for loc, g in df.groupby('location'):
                g = g.sort_values('measurement_date')
                dt = g['measurement_date'].diff().dt.total_seconds() / (3600*24*365)
                dth = g['thickness_loss_mm'].diff()
                rate = dth / (dt.replace(0, np.nan))
                df.loc[rate.index, 'corrosion_rate_mm_year'] = rate.values
        else:
            df['corrosion_rate_mm_year'] = np.nan

    # remaining life heuristic: (current_thickness - min_required)/corrosion_rate
    min_req = 5.0
    if 'current_thickness' in df.columns and 'corrosion_rate_mm_year' in df.columns:
        df['remaining_life_years'] = (df['current_thickness'] - min_req) / (df['corrosion_rate_mm_year'].abs() + 1e-12)
        df.loc[df['remaining_life_years'] < 0, 'remaining_life_years'] = 0

    out = os.path.join(out_dir, 'market_pipe_thickness_loss_dataset_clean.csv')
    df.to_csv(out, index=False)
    print('Wrote cleaned pipe/corrosion dataset to', out)


if __name__ == '__main__':
    main()
