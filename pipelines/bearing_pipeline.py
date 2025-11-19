"""
Bearing Feature Engineering Pipeline

Reads windowed vibration features from converted_data/extracted/cwru/*.csv,
joins with metadata and operational context, computes health indices and anomaly flags,
and writes bearing_features.csv to data/features/.

Output schema:
- equipment_id, file_source, start_sample, window_length
- rms, peak, peak_to_peak, kurtosis, crest_factor
- rolling_mean_rms, rolling_std_rms, rms_trend_slope
- health_index, is_anomaly
- operating_speed_rpm, load_percent (from operational_context if available)
"""
import pandas as pd
import numpy as np
import glob
import os
import pathlib
from scipy.stats import linregress


def ensure_dir(d):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)


def map_file_to_equipment_id(filename):
    """
    Map CWRU file prefix to equipment_id using metadata.
    File naming: B007_0_features.csv -> equipment_subtype B007
    """
    basename = os.path.basename(filename)
    # Extract prefix before first underscore or digit
    parts = basename.replace('_features.csv', '').split('_')
    subtype = parts[0]  # e.g., B007, IR014, Normal
    # Look up in equipment_master
    return subtype


def load_equipment_master():
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'metadata', 'equipment_master.csv')
    if not os.path.exists(path):
        print(f"Warning: equipment_master.csv not found at {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def load_operational_context():
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'metadata', 'operational_context.csv')
    if not os.path.exists(path):
        print(f"Warning: operational_context.csv not found at {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def compute_rolling_features(df, window=10):
    """
    Compute rolling mean and std for rms over a sliding window of rows.
    """
    df = df.sort_values('start_sample').reset_index(drop=True)
    df['rolling_mean_rms'] = df['rms'].rolling(window=window, min_periods=1).mean()
    df['rolling_std_rms'] = df['rms'].rolling(window=window, min_periods=1).std()
    return df


def compute_trend_slope(df, window=20):
    """
    Compute linear trend slope of rms over the last 'window' samples.
    """
    slopes = []
    for i in range(len(df)):
        start_idx = max(0, i - window + 1)
        subset = df.iloc[start_idx:i+1]
        if len(subset) < 3:
            slopes.append(0.0)
        else:
            x = np.arange(len(subset))
            y = subset['rms'].values
            try:
                slope, _, _, _, _ = linregress(x, y)
                slopes.append(slope)
            except:
                slopes.append(0.0)
    df['rms_trend_slope'] = slopes
    return df


def compute_health_index(df):
    """
    Bearing health index formula (example):
    health_index = 1.0 - min(1.0, (rms/rms_threshold) * (kurtosis/kurtosis_threshold))
    
    Thresholds (typical for bearings):
    - rms_threshold = 4.5 mm/s (ISO 10816 vibration severity)
    - kurtosis_threshold = 4.0 (normal distribution kurtosis ~3, elevated indicates impulses)
    
    Lower health_index = worse condition.
    """
    rms_threshold = 4.5
    kurtosis_threshold = 4.0
    
    rms_factor = np.clip(df['rms'] / rms_threshold, 0, 1)
    kurt_factor = np.clip(df['kurtosis'] / kurtosis_threshold, 0, 1)
    
    df['health_index'] = 1.0 - (rms_factor * 0.6 + kurt_factor * 0.4)  # weighted combination
    df['health_index'] = df['health_index'].clip(0, 1)
    return df


def flag_anomalies(df):
    """
    Flag anomalies based on health_index and kurtosis thresholds.
    """
    df['is_anomaly'] = ((df['health_index'] < 0.5) | (df['kurtosis'] > 5.0)).astype(int)
    return df


def process_bearing_data():
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    processed_dir = os.path.join(base_dir, 'converted_data', 'processed')
    out_dir = os.path.join(base_dir, 'data', 'features')
    ensure_dir(out_dir)
    
    # Load metadata
    eq_master = load_equipment_master()
    op_context = load_operational_context()
    
    # Load consolidated bearing features from processed data
    bearing_file = os.path.join(processed_dir, 'bearing_features_all.csv')
    if not os.path.exists(bearing_file):
        print(f"Bearing processed file not found: {bearing_file}")
        print("Falling back to extracted data...")
        # Fallback to original logic
        cwru_dir = os.path.join(base_dir, 'converted_data', 'extracted', 'cwru')
        files = glob.glob(os.path.join(cwru_dir, '*_features.csv'))
        if not files:
            print("No bearing feature CSVs found")
            return
        
        all_parts = []
        for f in files:
            print(f"Processing {os.path.basename(f)}...")
            df = pd.read_csv(f)
            subtype = map_file_to_equipment_id(f)
            df['equipment_id'] = f"BEARING_{subtype}"
            df['file_source'] = os.path.basename(f)
            all_parts.append(df)
        
        combined = pd.concat(all_parts, ignore_index=True)
    else:
        print(f"Loading processed bearing data from {bearing_file}...")
        combined = pd.read_csv(bearing_file)
        print(f"Loaded {len(combined)} rows")
        
        # Map fault_type to equipment_id if not already present
        if 'equipment_id' not in combined.columns:
            combined['equipment_id'] = 'BEARING_' + combined['fault_type'].astype(str) + '_' + combined['load'].astype(str)
        
        if 'file_source' not in combined.columns:
            combined['file_source'] = combined.get('source_file', combined['fault_type'])
    
    # Compute rolling features if not already present
    if 'rolling_mean_rms' not in combined.columns:
        print("Computing rolling features...")
        combined = compute_rolling_features(combined, window=10)
    
    # Compute trend slope if not already present  
    if 'rms_trend_slope' not in combined.columns:
        print("Computing trend slope...")
        combined = compute_trend_slope(combined, window=20)
    
    # Compute health index if not already present
    if 'health_index' not in combined.columns:
        print("Computing health index...")
        combined = compute_health_index(combined)
    
    # Flag anomalies if not already present
    if 'is_anomaly' not in combined.columns:
        print("Flagging anomalies...")
        combined = flag_anomalies(combined)
    
    # Join with operational context if available
    if not op_context.empty and 'equipment_id' in combined.columns:
        # Merge operating context data
        combined = combined.merge(
            op_context[['equipment_id', 'operating_speed_rpm', 'load_percent']],
            on='equipment_id',
            how='left'
        )
    else:
        combined['operating_speed_rpm'] = np.nan
        combined['load_percent'] = np.nan
    
    # Reorder columns
    cols = ['equipment_id', 'file_source', 'start_sample', 'window_length',
            'rms', 'peak', 'peak_to_peak', 'kurtosis', 'crest_factor',
            'rolling_mean_rms', 'rolling_std_rms', 'rms_trend_slope',
            'health_index', 'is_anomaly',
            'operating_speed_rpm', 'load_percent']
    combined = combined[cols]
    
    out_file = os.path.join(out_dir, 'bearing_features.csv')
    combined.to_csv(out_file, index=False)
    print(f"\nWrote {len(combined)} rows to {out_file}")
    print(f"Anomalies detected: {combined['is_anomaly'].sum()} ({combined['is_anomaly'].mean()*100:.1f}%)")
    print(f"Mean health index: {combined['health_index'].mean():.3f}")


def main():
    process_bearing_data()


if __name__ == '__main__':
    main()
