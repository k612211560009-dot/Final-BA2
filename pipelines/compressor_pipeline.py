"""
Compressor Feature Engineering Pipeline

This pipeline processes screw compressor data and extracts features for predictive maintenance:
- Motor performance metrics (speed, power, efficiency)
- Pressure metrics (discharge, suction, pressure ratio)
- Flow rate analysis
- Vibration analysis (RMS, peak, frequency components)
- Temperature monitoring
- Bearing condition scoring
- Seal degradation assessment
- Health index computation
- RUL estimation

Input: raw_data/equipment_predictive_maintenance_dataset.csv (Screw Compressor records)
Output: data/features/compressor_features.csv

Author: Generated for PdM Multi-Equipment System
Date: November 14, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
FEATURES_DIR = BASE_DIR / "data/features"
METADATA_DIR = BASE_DIR / "supplement_data/metadata"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print(" "*20 + "COMPRESSOR PIPELINE")
print("="*70)

# 1. LOAD RAW DATA

print("\n[1/7] Loading compressor data...")

# Load full dataset
df = pd.read_csv(RAW_DATA_DIR / "equipment_predictive_maintenance_dataset.csv", low_memory=False)

# Filter for Screw Compressor only
compressor_df = df[df['equipment_type'] == 'Screw Compressor'].copy()

print(f"  Total records: {len(compressor_df)}")
print(f"  Unique compressors: {compressor_df['equipment_id'].nunique()}")
print(f"  Equipment IDs: {compressor_df['equipment_id'].unique()}")
print(f"  Date range: {compressor_df['timestamp'].min()} to {compressor_df['timestamp'].max()}")

# Convert timestamp
compressor_df['timestamp'] = pd.to_datetime(compressor_df['timestamp'])

# 2. FEATURE ENGINEERING - OPERATIONAL METRICS

print("\n[2/7] Engineering operational features...")

# Pressure ratio (discharge / suction)
compressor_df['pressure_ratio'] = compressor_df['discharge_pressure_bar'] / (compressor_df['suction_pressure_bar'] + 0.1)

# Specific power (power per unit flow rate) - kW/(m3/h)
compressor_df['specific_power'] = compressor_df['motor_power_kw'] / (compressor_df['flow_rate_m3h'] + 1)

# Efficiency proxy (flow rate * pressure ratio / power)
compressor_df['efficiency_proxy'] = (compressor_df['flow_rate_m3h'] * compressor_df['pressure_ratio']) / (compressor_df['motor_power_kw'] + 1)

# Normalize efficiency (0-1 scale)
eff_min = compressor_df['efficiency_proxy'].quantile(0.05)
eff_max = compressor_df['efficiency_proxy'].quantile(0.95)
compressor_df['efficiency_normalized'] = np.clip(
    (compressor_df['efficiency_proxy'] - eff_min) / (eff_max - eff_min + 1e-6),
    0, 1
)

# Load factor (actual speed / max speed observed)
max_speed = compressor_df['motor_speed_rpm'].quantile(0.99)
compressor_df['load_factor'] = np.clip(compressor_df['motor_speed_rpm'] / max_speed, 0, 1)

print(f"  Engineered operational features:")
print(f"    - pressure_ratio: mean={compressor_df['pressure_ratio'].mean():.2f}")
print(f"    - specific_power: mean={compressor_df['specific_power'].mean():.3f} kW/(m3/h)")
print(f"    - efficiency_normalized: mean={compressor_df['efficiency_normalized'].mean():.3f}")
print(f"    - load_factor: mean={compressor_df['load_factor'].mean():.3f}")

# 3. FEATURE ENGINEERING - VIBRATION ANALYSIS

print("\n[3/7] Engineering vibration features...")

# Vibration severity ratio (peak / RMS)
compressor_df['vibration_severity'] = compressor_df['vibration_peak_mms'] / (compressor_df['vibration_rms_mms'] + 0.01)

# Frequency component health
# High bearing frequency amplitude indicates bearing issues
compressor_df['bearing_health_indicator'] = 1.0 - np.clip(
    compressor_df['bearing_freq_amp'].fillna(0) / compressor_df['bearing_freq_amp'].quantile(0.95),
    0, 1
)

# Vibration trend (calculate slope over time for each equipment)
def calculate_vibration_trend(group):
    """Calculate vibration RMS trend slope using linear regression"""
    if len(group) < 10:
        return pd.Series(0, index=group.index)
    
    x = np.arange(len(group))
    y = group['vibration_rms_mms'].fillna(method='ffill').fillna(method='bfill').values
    
    # Linear regression
    if np.std(y) > 0:
        slope = np.polyfit(x, y, 1)[0]
    else:
        slope = 0
    
    return pd.Series(slope, index=group.index)

compressor_df['vibration_trend_slope'] = compressor_df.groupby('equipment_id').apply(
    calculate_vibration_trend
).reset_index(level=0, drop=True)

print(f"  Vibration features:")
print(f"    - vibration_severity: mean={compressor_df['vibration_severity'].mean():.2f}")
print(f"    - bearing_health_indicator: mean={compressor_df['bearing_health_indicator'].mean():.3f}")
print(f"    - vibration_trend_slope: mean={compressor_df['vibration_trend_slope'].mean():.6f}")

# 4. ROLLING STATISTICS

print("\n[4/7] Computing rolling statistics...")

# 10-record rolling window (2.5 hours at 15-min intervals)
rolling_window = 10

for col in ['motor_speed_rpm', 'temperature_c', 'vibration_rms_mms', 'efficiency_normalized']:
    compressor_df[f'rolling_mean_{col}'] = compressor_df.groupby('equipment_id')[col].transform(
        lambda x: x.rolling(window=rolling_window, min_periods=1).mean()
    )
    compressor_df[f'rolling_std_{col}'] = compressor_df.groupby('equipment_id')[col].transform(
        lambda x: x.rolling(window=rolling_window, min_periods=1).std()
    )

print(f"  Rolling statistics computed for 4 key metrics (10-record window)")

# 5. BEARING & SEAL CONDITION SCORING

print("\n[5/7] Scoring bearing and seal condition...")

# Map bearing_condition to numeric score
bearing_map = {'Normal': 1.0, 'Marginal': 0.6, 'Critical': 0.2}
compressor_df['bearing_condition_score'] = compressor_df['bearing_condition'].map(bearing_map).fillna(0.8)

# Map seal_degradation to numeric score
seal_map = {'Low': 1.0, 'Medium': 0.6, 'High': 0.2}
compressor_df['seal_condition_score'] = compressor_df['seal_degradation'].map(seal_map).fillna(0.8)

print(f"  Bearing condition score: mean={compressor_df['bearing_condition_score'].mean():.3f}")
print(f"  Seal condition score: mean={compressor_df['seal_condition_score'].mean():.3f}")

# 6. HEALTH INDEX COMPUTATION

print("\n[6/7] Computing health index...")

# Composite health index (weighted average of multiple factors)
# Higher score = healthier equipment
compressor_df['health_index'] = (
    compressor_df['efficiency_normalized'] * 0.30 +      # Efficiency contribution
    compressor_df['bearing_health_indicator'] * 0.25 +   # Bearing health
    compressor_df['bearing_condition_score'] * 0.20 +    # Bearing condition
    compressor_df['seal_condition_score'] * 0.15 +       # Seal condition
    (1 - np.clip(compressor_df['vibration_rms_mms'] / compressor_df['vibration_rms_mms'].quantile(0.95), 0, 1)) * 0.10  # Vibration (inverted)
)

# Ensure health_index is in [0, 1]
compressor_df['health_index'] = np.clip(compressor_df['health_index'], 0, 1)

print(f"  Health index statistics:")
print(f"    Mean: {compressor_df['health_index'].mean():.3f}")
print(f"    Min:  {compressor_df['health_index'].min():.3f}")
print(f"    Max:  {compressor_df['health_index'].max():.3f}")

# 7. RUL ESTIMATION & ANOMALY DETECTION

print("\n[7/7] Estimating RUL and detecting anomalies...")

# Use days_to_failure if available, else estimate from health
compressor_df['rul_days'] = compressor_df['days_to_failure'].fillna(
    compressor_df['health_index'] * 365  # Healthy equipment ~1 year RUL
)

# Anomaly detection (critical conditions)
compressor_df['is_anomaly'] = (
    (compressor_df['health_index'] < 0.4) |                                    # Low health
    (compressor_df['vibration_rms_mms'] > compressor_df['vibration_rms_mms'].quantile(0.95)) |  # High vibration
    (compressor_df['temperature_c'] > compressor_df['temperature_c'].quantile(0.95)) |          # High temperature
    (compressor_df['bearing_condition'] == 'Critical') |                       # Critical bearing
    (compressor_df['seal_degradation'] == 'High')                              # High seal degradation
)

anomaly_count = compressor_df['is_anomaly'].sum()
anomaly_pct = (anomaly_count / len(compressor_df)) * 100

print(f"  RUL statistics:")
print(f"    Mean: {compressor_df['rul_days'].mean():.1f} days")
print(f"    Median: {compressor_df['rul_days'].median():.1f} days")
print(f"  Anomalies detected: {anomaly_count} ({anomaly_pct:.2f}%)")

# Count compressors with anomalies
compressors_with_anomalies = compressor_df[compressor_df['is_anomaly'] == True]['equipment_id'].nunique()
print(f"  Compressors with anomalies: {compressors_with_anomalies} / {compressor_df['equipment_id'].nunique()}")

# 8. ENRICH WITH METADATA

print("\n[8/8] Enriching with equipment metadata...")

# Load equipment master
equipment_master = pd.read_csv(METADATA_DIR / "equipment_master.csv")

# Filter for compressor entries (if they exist)
comp_master = equipment_master[equipment_master['equipment_id'].str.startswith('COMP')].copy()

if len(comp_master) == 0:
    print("  No compressor entries in equipment_master.csv, creating synthetic metadata...")
    # Create metadata from raw data
    comp_metadata = compressor_df[['equipment_id', 'installation_date', 'design_lifetime_years', 'criticality_level']].drop_duplicates('equipment_id')
    comp_metadata['location'] = comp_metadata['equipment_id'].map({
        'COMP_001': 'Building A - Compressor Room 1',
        'COMP_002': 'Building B - Compressor Room 2',
        'COMP_003': 'Building C - Utility Area'
    })
    comp_metadata['manufacturer'] = 'Atlas Copco'
    comp_metadata['model'] = 'GA 75 VSD+'
else:
    comp_metadata = comp_master

# Merge metadata
compressor_enriched = compressor_df.merge(
    comp_metadata[['equipment_id', 'location', 'manufacturer', 'model']],
    on='equipment_id',
    how='left'
)

print(f"  Enriched {len(compressor_enriched)} records with metadata")

# 9. SELECT FINAL FEATURES & SAVE

print("\n[9/9] Selecting features and saving...")

# Select final feature columns
feature_cols = [
    # Identifiers
    'equipment_id', 'timestamp', 'operating_hours',
    
    # Raw operational metrics
    'motor_speed_rpm', 'flow_rate_m3h', 
    'discharge_pressure_bar', 'suction_pressure_bar', 'motor_power_kw',
    'temperature_c',
    
    # Vibration metrics
    'vibration_rms_mms', 'vibration_peak_mms', 
    'freq1_amp', 'freq2_amp', 'bearing_freq_amp',
    
    # Engineered features
    'pressure_ratio', 'specific_power', 'efficiency_proxy', 'efficiency_normalized',
    'load_factor', 'vibration_severity', 'bearing_health_indicator',
    
    # Rolling statistics (8 features)
    'rolling_mean_motor_speed_rpm', 'rolling_std_motor_speed_rpm',
    'rolling_mean_temperature_c', 'rolling_std_temperature_c',
    'rolling_mean_vibration_rms_mms', 'rolling_std_vibration_rms_mms',
    'rolling_mean_efficiency_normalized', 'rolling_std_efficiency_normalized',
    
    # Condition scores
    'bearing_condition_score', 'seal_condition_score',
    
    # Degradation trends
    'vibration_trend_slope',
    
    # Health & RUL
    'health_index', 'rul_days',
    
    # Flags
    'is_anomaly',
    
    # Metadata
    'location', 'manufacturer', 'model'
]

# Create final output
output_df = compressor_enriched[feature_cols].copy()

# Save to CSV
output_path = FEATURES_DIR / "compressor_features.csv"
output_df.to_csv(output_path, index=False)

print(f"\n  Output saved: {output_path}")
print(f"  Shape: {output_df.shape}")
print(f"  Features: {len(feature_cols)} columns")

# 10. SUMMARY STATISTICS

print("\n" + "="*70)
print(" "*20 + "PIPELINE SUMMARY")
print("="*70)

print(f"\nInput Data:")
print(f"  Total records: {len(compressor_df):,}")
print(f"  Unique compressors: {compressor_df['equipment_id'].nunique()}")
print(f"  Date range: {compressor_df['timestamp'].min()} to {compressor_df['timestamp'].max()}")
print(f"  Duration: {(compressor_df['timestamp'].max() - compressor_df['timestamp'].min()).days} days")

print(f"\nOperational Metrics:")
print(f"  Motor speed: {compressor_df['motor_speed_rpm'].mean():.1f} ± {compressor_df['motor_speed_rpm'].std():.1f} RPM")
print(f"  Flow rate: {compressor_df['flow_rate_m3h'].mean():.1f} ± {compressor_df['flow_rate_m3h'].std():.1f} m3/h")
print(f"  Discharge pressure: {compressor_df['discharge_pressure_bar'].mean():.1f} ± {compressor_df['discharge_pressure_bar'].std():.1f} bar")
print(f"  Temperature: {compressor_df['temperature_c'].mean():.1f} ± {compressor_df['temperature_c'].std():.1f} °C")

print(f"\nHealth Metrics:")
print(f"  Health index: {compressor_df['health_index'].mean():.3f} (min: {compressor_df['health_index'].min():.3f}, max: {compressor_df['health_index'].max():.3f})")
print(f"  Efficiency normalized: {compressor_df['efficiency_normalized'].mean():.3f}")
print(f"  Bearing condition score: {compressor_df['bearing_condition_score'].mean():.3f}")
print(f"  Seal condition score: {compressor_df['seal_condition_score'].mean():.3f}")

print(f"\nRUL & Anomalies:")
print(f"  Mean RUL: {compressor_df['rul_days'].mean():.1f} days (~{compressor_df['rul_days'].mean()/365:.2f} years)")
print(f"  Median RUL: {compressor_df['rul_days'].median():.1f} days")
print(f"  Anomalies: {anomaly_count} ({anomaly_pct:.2f}%) across {compressors_with_anomalies} compressors")

print(f"\nCritical Equipment:")
critical_count = (compressor_df['health_index'] < 0.4).sum()
print(f"  Records with health < 0.4: {critical_count} ({critical_count/len(compressor_df)*100:.2f}%)")

print(f"\nOutput:")
print(f"  File: {output_path}")
print(f"  Records: {len(output_df):,}")
print(f"  Features: {len(feature_cols)}")

print("\n" + "="*70)
print("COMPRESSOR PIPELINE COMPLETE")
print("="*70)
