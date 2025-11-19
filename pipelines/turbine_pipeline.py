"""
Turbine/Engine Pipeline - CMAPS Turbofan Feature Engineering

This pipeline:
1. Loads CMAPS turbofan engine data (train/test FD001-FD004)
2. Parses 26 sensor columns (temperature, pressure, speed, etc.)
3. Maps unit_id to equipment_id
4. Computes rolling statistics and degradation features
5. Calculates health index from sensor trends
6. Estimates RUL (Remaining Useful Life) from actual labels
7. Outputs turbine_features.csv for dashboard

Author: Generated from PdM Architecture Design
Date: November 14, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = BASE_DIR / "converted_data/extracted/cmaps"
PROCESSED_DIR = BASE_DIR / "converted_data/processed"
FEATURES_DIR = BASE_DIR / "data/features"
METADATA_DIR = BASE_DIR / "supplement_data/metadata"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print(" "*23 + "TURBINE PIPELINE")
print("="*70)

# 1. LOAD DATA

print("\nLoading data...")

# CMAPS sensor column names (from NASA documentation)
# Load processed data (consolidated from all FD files)
print("   Loading processed turbine data...")
train_df = pd.read_csv(PROCESSED_DIR / "turbine_train_data.csv")
print(f"   Training data: {train_df.shape}")

test_df = pd.read_csv(PROCESSED_DIR / "turbine_test_data.csv")
print(f"   Test data: {test_df.shape}")

rul_df = pd.read_csv(PROCESSED_DIR / "turbine_rul_data.csv")
print(f"   RUL labels: {rul_df.shape}")

# Filter to use only FD001 dataset (single operating condition)
train_df = train_df[train_df['dataset'] == 'train_FD001'].copy()
test_df = test_df[test_df['dataset'] == 'test_FD001'].copy()
rul_df = rul_df[rul_df['dataset'] == 'RUL_FD001'].copy()

print(f"   Filtered to FD001: train={train_df.shape}, test={test_df.shape}")

# Rename columns if needed (processed data has numeric column names from CSV)
if '0' in str(train_df.columns[0]):
    # Columns: 0=unit_id, 1=time_cycles, 2-4=op_settings, 5-26=sensors (21 sensors), 'dataset'
    col_names = ['unit_id', 'time_cycles'] + \
                [f'op_setting_{i}' for i in range(1, 4)] + \
                [f'sensor_{i}' for i in range(1, 22)] + ['dataset']
    train_df.columns = col_names
    test_df.columns = col_names

# Calculate RUL if not already present
if 'rul_actual' not in train_df.columns:
    train_df['max_cycles'] = train_df.groupby('unit_id')['time_cycles'].transform('max')
    train_df['rul_actual'] = train_df['max_cycles'] - train_df['time_cycles']

# Calculate max_cycles for test set as well
test_df['max_cycles'] = test_df.groupby('unit_id')['time_cycles'].transform('max')

# Get unique unit IDs from test set (sorted)
test_units_list = sorted(test_df['unit_id'].unique())
print(f"   Test set has {len(test_units_list)} unique units")

# Rename RUL data columns if needed
if '0' in str(rul_df.columns[0]):
    rul_df.columns = ['rul_actual', 'dataset']

# Create RUL mapping (one RUL per unit, take first N RUL values)
rul_mapping = pd.DataFrame({
    'unit_id': test_units_list,
    'rul_at_last_cycle': rul_df['rul_actual'].values[:len(test_units_list)]
})

# Merge RUL back to test_df
test_df = test_df.merge(rul_mapping, on='unit_id', how='left')
test_df['rul_actual'] = test_df['rul_at_last_cycle'] + (test_df['max_cycles'] - test_df['time_cycles'])
test_df['dataset'] = 'test'

# Combine
turbine_df = pd.concat([train_df, test_df], ignore_index=True)
print(f"   Combined dataset: {turbine_df.shape}")

# Load metadata
equipment_master = pd.read_csv(METADATA_DIR / "equipment_master.csv")
print(f"   Equipment master: {equipment_master.shape}")

# 2. MAP UNIT_ID TO EQUIPMENT_ID

print("\n Mapping unit IDs to equipment IDs...")

# Map unit_id to equipment_id
# In production, this would come from asset registry
def map_unit_to_equipment(unit_id):
    """Map unit_id to equipment_id"""
    return f"TURBINE-{unit_id:03d}"

turbine_df['equipment_id'] = turbine_df['unit_id'].apply(map_unit_to_equipment)
print(f"   Mapped {turbine_df['equipment_id'].nunique()} unique turbines")

# Create timestamp (synthetic - assume 1 cycle = 1 hour)
turbine_df['timestamp'] = pd.Timestamp('2023-01-01') + pd.to_timedelta(turbine_df['time_cycles'], unit='h')
print(f"   Created timestamp: {turbine_df['timestamp'].min()} to {turbine_df['timestamp'].max()}")

# 3. SENSOR FEATURE ENGINEERING

print("\n Engineering sensor features...")

# Key sensors for turbofan health (domain knowledge from NASA CMAPS)
# sensor_2: Total temperature at fan inlet (T2)
# sensor_3: Total temperature at LPC outlet (T24)
# sensor_4: Total temperature at HPC outlet (T30)
# sensor_7: Total pressure at fan inlet (P2)
# sensor_11: Physical fan speed (Nf)
# sensor_12: Physical core speed (Nc)
# sensor_13: Engine pressure ratio (P30/P2)
# sensor_17: Bypass ratio

# 3.1 Temperature Gradient (T30 - T2)
turbine_df['temp_gradient'] = turbine_df['sensor_4'] - turbine_df['sensor_2']

# 3.2 Pressure Ratio Normalized
turbine_df['pressure_ratio_norm'] = turbine_df['sensor_13'] / turbine_df['sensor_13'].max()

# 3.3 Speed Ratio (core/fan)
turbine_df['speed_ratio'] = turbine_df['sensor_12'] / (turbine_df['sensor_11'] + 1e-9)

# 3.4 Composite efficiency proxy
# Higher temp gradient + lower pressure ratio = degradation
turbine_df['efficiency_proxy'] = turbine_df['sensor_13'] / (turbine_df['temp_gradient'] + 1)

print(f"   Engineered 4 composite sensor features")

# 4. ROLLING STATISTICS (Per Equipment)

print("\n Computing rolling statistics...")

def compute_rolling_features(group):
    """Compute rolling statistics for key sensors"""
    group = group.sort_values('time_cycles')
    
    # 10-cycle rolling windows
    for sensor in ['sensor_2', 'sensor_4', 'sensor_11', 'sensor_12', 'sensor_13']:
        group[f'{sensor}_roll_mean'] = group[sensor].rolling(window=10, min_periods=1).mean()
        group[f'{sensor}_roll_std'] = group[sensor].rolling(window=10, min_periods=1).std()
    
    # 5-cycle rolling for efficiency proxy
    group['efficiency_roll_mean'] = group['efficiency_proxy'].rolling(window=5, min_periods=1).mean()
    
    return group

turbine_df = turbine_df.groupby('equipment_id', group_keys=False).apply(compute_rolling_features)
print(f"   Created rolling features (10-cycle and 5-cycle windows)")

# 5. DEGRADATION TREND FEATURES

print("\n Computing degradation trends...")

def compute_trend_slope(group):
    """Compute linear trend slope for sensor degradation"""
    group = group.sort_values('time_cycles')
    
    # Temperature trend (increasing temp = degradation)
    if len(group) >= 5:
        cycles = group['time_cycles'].values
        temp = group['sensor_4'].values
        
        # Simple linear regression slope
        n = len(cycles)
        slope = (n * np.sum(cycles * temp) - np.sum(cycles) * np.sum(temp)) / \
                (n * np.sum(cycles**2) - np.sum(cycles)**2 + 1e-9)
        
        group['temp_trend_slope'] = slope
    else:
        group['temp_trend_slope'] = 0.0
    
    return group

turbine_df = turbine_df.groupby('equipment_id', group_keys=False).apply(compute_trend_slope)
print(f"   Temperature trend slope computed")

# 6. HEALTH INDEX COMPUTATION

print("\n Computing health index...")

def compute_health_index(row):
    """
    Composite health index (0-1 scale, 1=healthy, 0=critical)
    Based on:
    - Efficiency proxy (40%)
    - Temperature stability (30%, lower std = healthier)
    - Pressure ratio (20%, higher = healthier)
    - Speed stability (10%)
    """
    # Normalize components
    eff_norm = row['efficiency_roll_mean'] / (row['efficiency_roll_mean'] + 5)  # Sigmoid-like
    
    # Temperature stability (lower std = better)
    temp_std = row.get('sensor_4_roll_std', 1.0)
    temp_stability = 1.0 - min(temp_std / 10.0, 1.0)
    
    # Pressure ratio component
    pressure_component = row['pressure_ratio_norm']
    
    # Speed stability
    speed_std = row.get('sensor_12_roll_std', 1.0)
    speed_stability = 1.0 - min(speed_std / 50.0, 1.0)
    
    # Weighted health
    health = (
        eff_norm * 0.40 +
        temp_stability * 0.30 +
        pressure_component * 0.20 +
        speed_stability * 0.10
    )
    
    return max(min(health, 1.0), 0.0)

turbine_df['health_index'] = turbine_df.apply(compute_health_index, axis=1)

print(f"   Health index: mean={turbine_df['health_index'].mean():.3f}, "
      f"min={turbine_df['health_index'].min():.3f}, max={turbine_df['health_index'].max():.3f}")

# 7. ANOMALY DETECTION

print("\n Flagging anomalies...")

def flag_anomalies_for_group(group):
    """Flag anomalies based on sensor thresholds and health degradation"""
    group = group.sort_values('time_cycles')
    
    # Anomaly conditions:
    # 1. Temperature spike (sensor_4 > mean + 3*std)
    temp_mean = group['sensor_4'].mean()
    temp_std = group['sensor_4'].std()
    temp_threshold = temp_mean + 3 * temp_std
    
    # 2. Low health index
    health_threshold = 0.4
    
    # 3. Rapid health degradation
    group['health_change'] = group['health_index'].diff()
    rapid_degradation = group['health_change'] < -0.05
    
    group['is_anomaly'] = (
        (group['sensor_4'] > temp_threshold) |
        (group['health_index'] < health_threshold) |
        rapid_degradation
    )
    
    return group

turbine_df = turbine_df.groupby('equipment_id', group_keys=False).apply(flag_anomalies_for_group)

print(f"   Anomalies detected: {turbine_df['is_anomaly'].sum()} "
      f"({turbine_df['is_anomaly'].sum()/len(turbine_df)*100:.1f}%)")

# 8. ENRICH WITH METADATA

print("\n Enriching with metadata...")

# Get turbine equipment from master
turbine_equipment = equipment_master[equipment_master['equipment_type'] == 'Turbine'].copy()
print(f"   Found {len(turbine_equipment)} turbines in equipment master")

# Merge with equipment master
turbine_enriched = turbine_df.merge(
    turbine_equipment[['equipment_id', 'location', 'manufacturer', 'model', 'installation_date']],
    on='equipment_id',
    how='left'
)

print(f"   Enriched with metadata: {turbine_enriched.shape}")

# 9. SELECT FINAL FEATURES

print("\n Selecting final features...")

# Final feature set
final_features = [
    # Identifiers
    'equipment_id',
    'timestamp',
    'time_cycles',
    'dataset',
    
    # Operational settings
    'op_setting_1',
    'op_setting_2',
    'op_setting_3',
    
    # Key sensors (select most important)
    'sensor_2',  # T2 - Fan inlet temp
    'sensor_4',  # T30 - HPC outlet temp
    'sensor_7',  # P2 - Fan inlet pressure
    'sensor_11', # Nf - Fan speed
    'sensor_12', # Nc - Core speed
    'sensor_13', # P30/P2 - Pressure ratio
    'sensor_17', # Bypass ratio
    
    # Engineered features
    'temp_gradient',
    'pressure_ratio_norm',
    'speed_ratio',
    'efficiency_proxy',
    
    # Rolling statistics (select key ones)
    'sensor_4_roll_mean',
    'sensor_4_roll_std',
    'sensor_13_roll_mean',
    'efficiency_roll_mean',
    
    # Degradation indicators
    'temp_trend_slope',
    
    # Health & RUL
    'health_index',
    'rul_actual',
    'is_anomaly',
    
    # Metadata
    'location',
    'manufacturer',
    'model'
]

# Filter to available columns
available_features = [f for f in final_features if f in turbine_enriched.columns]
turbine_features = turbine_enriched[available_features].copy()

print(f"   Selected {len(available_features)} features")

# 10. SAVE OUTPUT

print("\n Saving output...")

output_path = FEATURES_DIR / "turbine_features.csv"
turbine_features.to_csv(output_path, index=False)

print(f"   Saved to: {output_path}")
print(f"   Shape: {turbine_features.shape}")

# 11. SUMMARY STATISTICS

print("\n Summary Statistics:")
print(f"  Total records: {len(turbine_features):,}")
print(f"  Unique turbines: {turbine_features['equipment_id'].nunique()}")
print(f"  Date range: {turbine_features['timestamp'].min()} to {turbine_features['timestamp'].max()}")
print(f"  Cycle range: {turbine_features['time_cycles'].min()} to {turbine_features['time_cycles'].max()}")

print(f"\n  Health Index Distribution:")
print(f"    Mean: {turbine_features['health_index'].mean():.3f}")
print(f"    Median: {turbine_features['health_index'].median():.3f}")
print(f"    Min: {turbine_features['health_index'].min():.3f}")
print(f"    Max: {turbine_features['health_index'].max():.3f}")

print(f"\n  RUL Distribution:")
print(f"    Mean: {turbine_features['rul_actual'].mean():.1f} cycles")
print(f"    Median: {turbine_features['rul_actual'].median():.1f} cycles")
print(f"    Max: {turbine_features['rul_actual'].max():.0f} cycles")

print(f"\n  Critical Turbines (health_index < 0.4):")
critical_turbines = turbine_features[turbine_features['health_index'] < 0.4]
print(f"    Count: {len(critical_turbines):,} ({len(critical_turbines)/len(turbine_features)*100:.1f}%)")

print(f"\n  Anomaly Summary:")
print(f"    Total anomalies: {turbine_features['is_anomaly'].sum():,}")
print(f"    Turbines with anomalies: {turbine_features[turbine_features['is_anomaly']]['equipment_id'].nunique()}")

print(f"\n  Dataset Split:")
for ds in turbine_features['dataset'].unique():
    count = (turbine_features['dataset'] == ds).sum()
    print(f"    {ds}: {count:,} records ({count/len(turbine_features)*100:.1f}%)")

print("\n" + "="*70)
print(" TURBINE PIPELINE COMPLETE")
print("="*70)
