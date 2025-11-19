"""
Pump Pipeline - Feature Engineering for Pump Vibration & Temperature Data

This pipeline:
1. Loads pump vibration measurements (ISO, DEMO, ACC, P2P) and temperature
2. Creates timestamp from datetime fields
3. Maps Machine_ID to equipment_id via equipment_master
4. Computes efficiency metrics (ISO/DEMO ratio)
5. Detects seal condition from temperature correlation
6. Computes rolling statistics and health index
7. Estimates RUL from degradation trends
8. Flags anomalies
9. Outputs pump_features.csv

Author: Generated from PdM Architecture Design
Date: November 13, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = BASE_DIR / "converted_data/extracted/pumps"
METADATA_DIR = BASE_DIR / "supplement_data/metadata"
FEATURES_DIR = BASE_DIR / "data/features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print(" "*25 + "PUMP PIPELINE")
print("="*70)

# 1. LOAD DATA

print("\n Loading data...")

# Load pump data
pump_df = pd.read_csv(EXTRACTED_DIR / "pumps.csv")
print(f" Pump data: {pump_df.shape}")
print(f" Columns: {pump_df.columns.tolist()}")

# Load metadata
equipment_master = pd.read_csv(METADATA_DIR / "equipment_master.csv")
operational_context = pd.read_csv(METADATA_DIR / "operational_context.csv")

print(f" Equipment master: {equipment_master.shape}")
print(f" Operational context: {operational_context.shape}")

# 2. CREATE TIMESTAMP & MAP TO EQUIPMENT ID

print("\n Creating timestamp and mapping equipment IDs...")

# Create timestamp from datetime fields
pump_df['timestamp'] = pd.to_datetime(
    pump_df[['year', 'month', 'day', 'hour', 'minute', 'second']]
)
print(f" Created timestamp: {pump_df['timestamp'].min()} to {pump_df['timestamp'].max()}")

# Map Machine_ID to equipment_id
# Get pump equipment from master
pump_equipment = equipment_master[equipment_master['equipment_type'] == 'Pump'].copy()
print(f" Found {len(pump_equipment)} pumps in equipment master")

# Create mapping dictionary (Machine_ID -> equipment_id)
# In production, this would come from asset registry
def map_machine_to_equipment(machine_id):
    """Map Machine_ID to equipment_id"""
    # Simple mapping: PUMP-001, PUMP-002, etc.
    return f"PUMP-{machine_id:03d}"

pump_df['equipment_id'] = pump_df['Machine_ID'].apply(map_machine_to_equipment)
print(f" Mapped {pump_df['equipment_id'].nunique()} unique equipment IDs")

# 3. FEATURE ENGINEERING - Efficiency Metrics

print("\n Engineering efficiency features...")

# 3.1 Efficiency Score (ISO/DEMO ratio)
# Higher ISO relative to DEMO indicates better efficiency
# Lower values suggest energy loss or mechanical issues
pump_df['efficiency_score'] = pump_df['value_ISO'] / (pump_df['value_DEMO'] + 1e-9)  # Avoid division by zero

# Normalize to 0-1 scale (using percentile-based normalization)
eff_min, eff_max = pump_df['efficiency_score'].quantile([0.05, 0.95])
pump_df['efficiency_normalized'] = (
    (pump_df['efficiency_score'] - eff_min) / (eff_max - eff_min)
).clip(0, 1)

print(f" Efficiency score: mean={pump_df['efficiency_score'].mean():.2f}")

# 3.2 Vibration Severity Index (composite of ACC and P2P)
# Combines acceleration and peak-to-peak measurements
pump_df['vibration_severity'] = np.sqrt(
    pump_df['value_ACC']**2 + (pump_df['value_P2P'] / 10)**2  # Scale P2P down
)

print(f" Vibration severity: mean={pump_df['vibration_severity'].mean():.4f}")

# 4. SEAL CONDITION DETECTION

print("\n Detecting seal condition...")

# Seal degradation manifests as:
# 1. High temperature (>50°C for industrial pumps)
# 2. Correlation between temperature rise and vibration increase

# 4.1 Temperature trend
pump_df['temp_above_threshold'] = pump_df['valueTEMP'] > 50

# 4.2 Temperature-Vibration correlation (per equipment)
def compute_seal_condition(group):
    """
    Compute seal condition score based on temp-vibration correlation
    Score: 0 (bad) to 1 (good)
    """
    if len(group) < 10:  # Need minimum samples
        return pd.Series([0.5] * len(group), index=group.index)
    
    # Compute correlation
    corr = group['valueTEMP'].corr(group['value_ISO'])
    
    # High positive correlation = seal degradation
    # Score: inverse of correlation (capped at 1)
    if pd.isna(corr):
        seal_score = 0.5
    else:
        seal_score = 1.0 - min(max(corr, 0), 1.0)  # Invert positive correlation
    
    return pd.Series([seal_score] * len(group), index=group.index)

pump_df['seal_condition_score'] = pump_df.groupby('equipment_id', group_keys=False).apply(
    compute_seal_condition
).values

print(f" Seal condition score: mean={pump_df['seal_condition_score'].mean():.3f}")

# 5. ROLLING STATISTICS (Time-Window Features)

print("\n Computing rolling statistics...")

# Sort by equipment and time
pump_df = pump_df.sort_values(['equipment_id', 'timestamp'])

# Define rolling windows (30 days, 7 days)
def compute_rolling_features(group):
    """Compute rolling mean and std for key metrics"""
    
    # Set timestamp as index for rolling
    group = group.set_index('timestamp').sort_index()
    
    # 30-day rolling features
    group['iso_roll_mean_30d'] = group['value_ISO'].rolling('30D', min_periods=1).mean()
    group['iso_roll_std_30d'] = group['value_ISO'].rolling('30D', min_periods=1).std()
    group['temp_roll_mean_30d'] = group['valueTEMP'].rolling('30D', min_periods=1).mean()
    
    # 7-day rolling features
    group['iso_roll_mean_7d'] = group['value_ISO'].rolling('7D', min_periods=1).mean()
    group['vibration_roll_mean_7d'] = group['vibration_severity'].rolling('7D', min_periods=1).mean()
    
    return group.reset_index()

pump_df = pump_df.groupby('equipment_id', group_keys=False).apply(compute_rolling_features)

print(f" Created 5 rolling features (7-day and 30-day windows)")

# 6. HEALTH INDEX COMPUTATION

print("\n Computing health index...")

def compute_health_index(row):
    """
    Composite health index (0-1 scale, 1=healthy, 0=critical)
    Based on:
    - Efficiency normalized (40%)
    - Seal condition score (30%)
    - Vibration severity (20%, inverted)
    - Temperature stability (10%, based on rolling std)
    """
    # Vibration component (inverted: lower vibration = healthier)
    vib_max = 0.1  # Threshold for critical vibration
    vib_component = 1.0 - min(row['vibration_severity'] / vib_max, 1.0)
    
    # Temperature stability component (lower std = more stable = healthier)
    temp_std = row.get('iso_roll_std_30d', 0)
    temp_stability = 1.0 - min(temp_std / 5.0, 1.0)  # Normalize by threshold
    
    # Weighted sum
    health = (
        row['efficiency_normalized'] * 0.40 +
        row['seal_condition_score'] * 0.30 +
        vib_component * 0.20 +
        temp_stability * 0.10
    )
    
    return max(min(health, 1.0), 0.0)

pump_df['health_index'] = pump_df.apply(compute_health_index, axis=1)

print(f" Health index: mean={pump_df['health_index'].mean():.3f}, "
      f"min={pump_df['health_index'].min():.3f}, max={pump_df['health_index'].max():.3f}")

# 7. RUL ESTIMATION

print("\n Estimating Remaining Useful Life (RUL)...")

def estimate_rul_per_equipment(health_series):
    """
    Estimate RUL based on health degradation rate
    Uses linear regression on health_index over time
    Returns a single RUL value broadcasted to all rows
    """
    if len(health_series) < 30:  # Need minimum samples for trend
        return 365 * 5
    
    # Get last 90 days of data
    recent = health_series.tail(min(90, len(health_series)))
    
    # Compute degradation rate (slope of health index)
    x = np.arange(len(recent))
    y = recent.values
    
    if len(x) > 1 and y.std() > 0:
        slope = np.polyfit(x, y, 1)[0]  # Linear fit
        
        # If degrading (negative slope), estimate days to failure
        if slope < 0:
            current_health = recent.iloc[-1]
            failure_threshold = 0.3  # Health below 0.3 = failure
            
            if current_health > failure_threshold:
                rul_days = (current_health - failure_threshold) / abs(slope)
                rul_days = min(rul_days, 365 * 5)  # Cap at 5 years
            else:
                rul_days = 0
        else:
            rul_days = 365 * 5  # Not degrading, assume 5 years
    else:
        rul_days = 365 * 5  # Insufficient data
    
    return rul_days

# Calculate RUL per equipment and broadcast to all rows
pump_df['rul_days'] = pump_df.groupby('equipment_id')['health_index'].transform(
    estimate_rul_per_equipment
)

print(f" RUL estimated: mean={pump_df['rul_days'].mean():.0f} days")

# 8. ANOMALY DETECTION

print("\n Flagging anomalies...")

def flag_anomalies_for_group(group):
    """
    Flag anomalies based on:
    1. ISO vibration exceeds 3 std deviations
    2. Temperature spike (>80°C)
    3. Sudden efficiency drop (>30% from rolling mean)
    """
    if len(group) < 10:
        return pd.Series([False] * len(group), index=group.index)
    
    # ISO vibration anomaly
    iso_mean = group['value_ISO'].mean()
    iso_std = group['value_ISO'].std()
    iso_threshold = iso_mean + 3 * iso_std
    iso_anomaly = group['value_ISO'] > iso_threshold
    
    # Temperature spike
    temp_anomaly = group['valueTEMP'] > 80
    
    # Efficiency drop (compare to rolling mean)
    eff_drop = (group['efficiency_normalized'] - group['iso_roll_mean_30d']) < -0.30
    
    # Combine
    anomalies = iso_anomaly | temp_anomaly | eff_drop
    
    return anomalies

# Apply anomaly detection per equipment group
anomaly_results = []
for equipment_id, group in pump_df.groupby('equipment_id'):
    anomalies = flag_anomalies_for_group(group)
    anomaly_results.append(anomalies)

pump_df['is_anomaly'] = pd.concat(anomaly_results)

print(f" Anomalies detected: {pump_df['is_anomaly'].sum()} ({pump_df['is_anomaly'].sum()/len(pump_df)*100:.2f}%)")

# 9. ENRICH WITH OPERATIONAL CONTEXT

print("\n Enriching with operational context...")

operational_context['timestamp'] = pd.to_datetime(operational_context['timestamp'])

# Join with operational context (by equipment_id and nearest timestamp)
pump_enriched = pd.merge_asof(
    pump_df.sort_values('timestamp'),
    operational_context[['equipment_id', 'timestamp', 'operating_speed_rpm', 'load_percent', 'operating_mode']].sort_values('timestamp'),
    on='timestamp',
    by='equipment_id',
    direction='nearest',
    tolerance=pd.Timedelta('1H')  # Match within 1 hour
)

print(f" Joined with operational context")
print(f"    - Matched {pump_enriched['operating_speed_rpm'].notna().sum()} records")

# 10. SELECT FINAL FEATURES

print("\n Selecting final features...")

final_features = [
    # Identifiers
    'equipment_id',
    'timestamp',
    
    # Raw measurements
    'value_ISO',
    'value_DEMO',
    'value_ACC',
    'value_P2P',
    'valueTEMP',
    
    # Engineered features
    'efficiency_score',
    'efficiency_normalized',
    'vibration_severity',
    'seal_condition_score',
    'temp_above_threshold',
    
    # Rolling features
    'iso_roll_mean_30d',
    'iso_roll_std_30d',
    'temp_roll_mean_30d',
    'iso_roll_mean_7d',
    'vibration_roll_mean_7d',
    
    # Health metrics
    'health_index',
    'rul_days',
    'is_anomaly',
    
    # Operational context
    'operating_speed_rpm',
    'load_percent',
    'operating_mode'
]

pump_features = pump_enriched[final_features].copy()

print(f" Selected {len(final_features)} features")

# 11. SAVE OUTPUT

print("\n Saving output...")

output_path = FEATURES_DIR / "pump_features.csv"
pump_features.to_csv(output_path, index=False)

print(f" Saved to: {output_path}")
print(f" Shape: {pump_features.shape}")

# 12. SUMMARY STATISTICS

print("\n Summary Statistics:")
print(f"  Total records: {len(pump_features)}")
print(f"  Unique equipment: {pump_features['equipment_id'].nunique()}")
print(f"  Date range: {pump_features['timestamp'].min()} to {pump_features['timestamp'].max()}")

print(f"\n  Health Index Distribution:")
print(f"    Mean: {pump_features['health_index'].mean():.3f}")
print(f"    Median: {pump_features['health_index'].median():.3f}")
print(f"    Min: {pump_features['health_index'].min():.3f}")
print(f"    Max: {pump_features['health_index'].max():.3f}")

print(f"\n  Critical Pumps (health_index < 0.4):")
critical_pumps = pump_features[pump_features['health_index'] < 0.4]
print(f"    Count: {len(critical_pumps)} ({len(critical_pumps)/len(pump_features)*100:.1f}%)")

print(f"\n  Anomaly Summary:")
print(f"    Total anomalies: {pump_features['is_anomaly'].sum()}")
print(f"    Equipment with anomalies: {pump_features[pump_features['is_anomaly']]['equipment_id'].nunique()}")

print("\n" + "="*70)
print("PUMP PIPELINE COMPLETE")
print("="*70)
