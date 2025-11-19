#!/usr/bin/env python3
"""
Run Multi-Equipment EDA Analysis
Executes the notebook sections programmatically to avoid kernel issues
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set styles
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
pd.set_option('display.max_columns', None)

print(" "*25 + "MULTI-EQUIPMENT EDA")

# Load all feature datasets
BASE_DIR = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
FEATURES_DIR = BASE_DIR / "data/features"

print("\n[1/7] Loading datasets...")
# Load corrosion features
corrosion_df = pd.read_csv(FEATURES_DIR / "corrosion_features.csv")
print(f"  Pipeline Corrosion: {corrosion_df.shape}")

# Load bearing features
bearing_df = pd.read_csv(FEATURES_DIR / "bearing_features.csv")
print(f"  Bearing: {bearing_df.shape}")

# Load pump features
pump_df = pd.read_csv(FEATURES_DIR / "pump_features.csv")
pump_df['timestamp'] = pd.to_datetime(pump_df['timestamp'])
print(f"  Pump: {pump_df.shape}")

# Load turbine features
turbine_df = pd.read_csv(FEATURES_DIR / "turbine_features.csv")
turbine_df['timestamp'] = pd.to_datetime(turbine_df['timestamp'])
print(f"  Turbine: {turbine_df.shape}")

# Load compressor features
compressor_df = pd.read_csv(FEATURES_DIR / "compressor_features.csv")
compressor_df['timestamp'] = pd.to_datetime(compressor_df['timestamp'])
print(f"  Compressor: {compressor_df.shape}")

print("\nAll datasets loaded successfully!")

# Create summary dataframe
print("\n[2/8] Creating equipment overview...")
equipment_summary = pd.DataFrame({
    'Equipment Type': ['Pipeline Corrosion', 'Bearing', 'Pump', 'Turbine', 'Compressor'],
    'Total Records': [
        len(corrosion_df),
        len(bearing_df),
        len(pump_df),
        len(turbine_df),
        len(compressor_df)
    ],
    'Unique Equipment': [
        corrosion_df['equipment_id'].nunique(),
        bearing_df['equipment_id'].nunique(),
        pump_df['equipment_id'].nunique(),
        turbine_df['equipment_id'].nunique(),
        compressor_df['equipment_id'].nunique()
    ],
    'Mean Health Index': [
        1 - corrosion_df['risk_score'].mean()/100,
        bearing_df['health_index'].mean(),
        pump_df['health_index'].mean(),
        turbine_df['health_index'].mean(),
        compressor_df['health_index'].mean()
    ],
    'Anomaly Rate (%)': [
        (corrosion_df['condition'] == 'Critical').sum() / len(corrosion_df) * 100,
        bearing_df['is_anomaly'].sum() / len(bearing_df) * 100,
        pump_df['is_anomaly'].sum() / len(pump_df) * 100,
        turbine_df['is_anomaly'].sum() / len(turbine_df) * 100,
        compressor_df['is_anomaly'].sum() / len(compressor_df) * 100
    ]
})

print(" "*25 + "EQUIPMENT OVERVIEW")
print(equipment_summary.to_string(index=False))

# Pipeline Corrosion Analysis
print("\n[3/8] Analyzing Pipeline Corrosion...")
print(" "*20 + "PIPELINE CORROSION EDA")
print(f"\nDataset: {corrosion_df.shape}")
print(f"Equipment: {corrosion_df['equipment_id'].nunique()} pipeline segments")
print(f"\nKey Features:")
print(f"  - Corrosion rate: {corrosion_df['corrosion_rate_mm_year'].mean():.3f} mm/year")
print(f"  - Remaining life: {corrosion_df['remaining_life_years'].mean():.1f} years")
print(f"  - Risk score: {corrosion_df['risk_score'].mean():.1f}/100")
print(f"  - Safety margin: {corrosion_df['safety_margin_percent'].mean():.1f}%")

condition_counts = corrosion_df['condition'].value_counts()
print(f"\nCondition Distribution:")
for cond, count in condition_counts.items():
    print(f"  {cond}: {count} ({count/len(corrosion_df)*100:.1f}%)")

# Bearing Analysis
print("\n[4/8] Analyzing Bearings...")
print(" "*25 + "BEARING EDA")
print(f"\nDataset: {bearing_df.shape}")
print(f"Equipment: {bearing_df['equipment_id'].nunique()} bearings")
print(f"\nKey Features:")
print(f"  - Mean RMS: {bearing_df['rms'].mean():.3f}")
print(f"  - Mean Health Index: {bearing_df['health_index'].mean():.3f}")
print(f"  - Anomalies: {bearing_df['is_anomaly'].sum()} ({bearing_df['is_anomaly'].sum()/len(bearing_df)*100:.1f}%)")

equipment_health = bearing_df.groupby('equipment_id')['health_index'].mean().sort_values()
print(f"\nHealth Index by Bearing:")
for eq, health in equipment_health.items():
    status = "CRITICAL" if health < 0.5 else "WARNING" if health < 0.7 else "GOOD"
    print(f"  {eq}: {health:.3f} [{status}]")

# Pump Analysis
print("\n[5/8] Analyzing Pumps...")
print(" "*27 + "PUMP EDA")
print(f"\nDataset: {pump_df.shape}")
print(f"Equipment: {pump_df['equipment_id'].nunique()} pumps")
print(f"Date range: {pump_df['timestamp'].min()} to {pump_df['timestamp'].max()}")
print(f"\nKey Features:")
print(f"  - Mean Efficiency: {pump_df['efficiency_normalized'].mean():.3f}")
print(f"  - Mean Health Index: {pump_df['health_index'].mean():.3f}")
print(f"  - Mean RUL: {pump_df['rul_days'].mean():.0f} days")
print(f"  - Anomalies: {pump_df['is_anomaly'].sum()} ({pump_df['is_anomaly'].sum()/len(pump_df)*100:.1f}%)")

for pump_id in pump_df['equipment_id'].unique():
    pump_data = pump_df[pump_df['equipment_id'] == pump_id]
    print(f"\n{pump_id} Summary:")
    print(f"  Health: {pump_data['health_index'].mean():.3f}")
    print(f"  Efficiency: {pump_data['efficiency_normalized'].mean():.3f}")
    print(f"  Avg RUL: {pump_data['rul_days'].mean():.0f} days")
    print(f"  Anomalies: {pump_data['is_anomaly'].sum()} ({pump_data['is_anomaly'].sum()/len(pump_data)*100:.1f}%)")

# Turbine Analysis
print("\n[6/8] Analyzing Turbines...")
print(" "*23 + "TURBINE/ENGINE EDA")
print(f"\nDataset: {turbine_df.shape}")
print(f"Equipment: {turbine_df['equipment_id'].nunique()} turbofan engines")
print(f"Cycle range: {turbine_df['time_cycles'].min()} to {turbine_df['time_cycles'].max()}")
print(f"\nKey Features:")
print(f"  - Mean Health Index: {turbine_df['health_index'].mean():.3f}")
print(f"  - Mean RUL: {turbine_df['rul_actual'].mean():.1f} cycles")
print(f"  - Mean Temperature (T30): {turbine_df['sensor_4'].mean():.1f}")
print(f"  - Mean Pressure Ratio: {turbine_df['sensor_13'].mean():.2f}")
print(f"  - Anomalies: {turbine_df['is_anomaly'].sum()} ({turbine_df['is_anomaly'].sum()/len(turbine_df)*100:.1f}%)")

# Dataset split
dataset_counts = turbine_df['dataset'].value_counts()
print(f"\nDataset Split:")
for dataset, count in dataset_counts.items():
    print(f"  {dataset}: {count} records ({count/len(turbine_df)*100:.1f}%)")

# RUL distribution stats
print(f"\nRUL Statistics:")
print(f"  Min: {turbine_df['rul_actual'].min():.0f} cycles")
print(f"  Q1: {turbine_df['rul_actual'].quantile(0.25):.0f} cycles")
print(f"  Median: {turbine_df['rul_actual'].median():.0f} cycles")
print(f"  Q3: {turbine_df['rul_actual'].quantile(0.75):.0f} cycles")
print(f"  Max: {turbine_df['rul_actual'].max():.0f} cycles")

# Compressor Analysis
print("\n[7/8] Analyzing Compressors...")
print(" "*23 + "COMPRESSOR EDA")
print(f"\nDataset: {compressor_df.shape}")
print(f"Equipment: {compressor_df['equipment_id'].nunique()} screw compressors")
print(f"Date range: {compressor_df['timestamp'].min()} to {compressor_df['timestamp'].max()}")
print(f"\nKey Features:")
print(f"  - Mean Health Index: {compressor_df['health_index'].mean():.3f}")
print(f"  - Mean Efficiency: {compressor_df['efficiency_normalized'].mean():.3f}")
print(f"  - Mean RUL: {compressor_df['rul_days'].mean():.1f} days (~{compressor_df['rul_days'].mean()/365:.1f} years)")
print(f"  - Mean Motor Speed: {compressor_df['motor_speed_rpm'].mean():.1f} RPM")
print(f"  - Mean Vibration: {compressor_df['vibration_rms_mms'].mean():.2f} mm/s")
print(f"  - Anomalies: {compressor_df['is_anomaly'].sum()} ({compressor_df['is_anomaly'].sum()/len(compressor_df)*100:.1f}%)")

print(f"\nOperational Metrics:")
print(f"  - Flow rate: {compressor_df['flow_rate_m3h'].mean():.1f} m3/h")
print(f"  - Discharge pressure: {compressor_df['discharge_pressure_bar'].mean():.1f} bar")
print(f"  - Temperature: {compressor_df['temperature_c'].mean():.1f} °C")
print(f"  - Load factor: {compressor_df['load_factor'].mean():.3f}")

for comp_id in compressor_df['equipment_id'].unique():
    comp_data = compressor_df[compressor_df['equipment_id'] == comp_id]
    print(f"\n{comp_id} Summary:")
    print(f"  Records: {len(comp_data):,}")
    print(f"  Health: {comp_data['health_index'].mean():.3f}")
    print(f"  Efficiency: {comp_data['efficiency_normalized'].mean():.3f}")
    print(f"  Avg RUL: {comp_data['rul_days'].mean():.0f} days")
    print(f"  Anomalies: {comp_data['is_anomaly'].sum()} ({comp_data['is_anomaly'].sum()/len(comp_data)*100:.1f}%)")

# Cross-Equipment Comparison
print("\n[8/8] Cross-Equipment Comparison...")
print(" "*20 + "CROSS-EQUIPMENT ANALYSIS")

# Critical equipment summary
print("\nCritical Equipment Summary:")
print(f"  Pipeline: {(corrosion_df['condition'] == 'Critical').sum()} critical segments")
print(f"  Bearing: {(bearing_df['health_index'] < 0.4).sum()} critical bearings")
print(f"  Pump: {(pump_df['health_index'] < 0.4).sum()} critical pump records")
print(f"  Turbine: {(turbine_df['health_index'] < 0.4).sum()} critical turbine records")
print(f"  Compressor: {(compressor_df['health_index'] < 0.4).sum()} critical compressor records")

# Key Findings
print(" "*25 + "KEY FINDINGS")

print("\n1. PIPELINE CORROSION:")
print(f"   - {(corrosion_df['condition']=='Critical').sum()} critical segments requiring immediate attention")
print(f"   - Mean remaining life: {corrosion_df['remaining_life_years'].mean():.1f} years")
highest_risk = corrosion_df.nlargest(1, 'risk_score')
print(f"   - Highest risk: {highest_risk['equipment_id'].values[0]} (risk: {highest_risk['risk_score'].values[0]:.1f})")

print("\n2. BEARING:")
print(f"   - {(bearing_df['health_index'] < 0.5).sum()} bearings below health threshold")
print(f"   - {bearing_df['is_anomaly'].sum()} anomalies detected across {bearing_df['equipment_id'].nunique()} bearings")
worst_bearing = bearing_df.groupby('equipment_id')['health_index'].mean().idxmin()
print(f"   - Worst performing: {worst_bearing} (health: {bearing_df.groupby('equipment_id')['health_index'].mean()[worst_bearing]:.3f})")

print("\n3. PUMP:")
print(f"   - {(pump_df['health_index'] < 0.4).sum()} critical pump records")
print(f"   - Mean efficiency: {pump_df['efficiency_normalized'].mean():.3f}")
print(f"   - Average RUL: {pump_df['rul_days'].mean():.0f} days (~{pump_df['rul_days'].mean()/365:.1f} years)")

print("\n4. TURBINE:")
print(f"   - {turbine_df['equipment_id'].nunique()} turbofan engines monitored")
print(f"   - Mean RUL: {turbine_df['rul_actual'].mean():.1f} cycles")
print(f"   - {turbine_df['is_anomaly'].sum()} anomalies detected ({turbine_df['is_anomaly'].sum()/len(turbine_df)*100:.2f}%)")
turbines_with_anomalies = turbine_df[turbine_df['is_anomaly'] == True]['equipment_id'].nunique()
print(f"   - {turbines_with_anomalies} turbines have anomalies")

print("\n5. COMPRESSOR:")
print(f"   - {compressor_df['equipment_id'].nunique()} screw compressors monitored")
print(f"   - Mean RUL: {compressor_df['rul_days'].mean():.1f} days (~{compressor_df['rul_days'].mean()/365:.1f} years)")
print(f"   - {compressor_df['is_anomaly'].sum()} anomalies detected ({compressor_df['is_anomaly'].sum()/len(compressor_df)*100:.2f}%)")
print(f"   - Mean efficiency: {compressor_df['efficiency_normalized'].mean():.3f}")
print(f"   - Critical records: {(compressor_df['health_index'] < 0.4).sum()}")

print(" "*25 + "RECOMMENDATIONS")

print("\n1. IMMEDIATE ACTIONS:")
print("   - Inspect critical pipeline segments with risk_score > 70")
print("   - Replace bearings with health_index < 0.4")
print("   - Schedule pump maintenance for units with RUL < 180 days")
print("   - Monitor turbines with frequent anomalies")

print("\n2. PREDICTIVE MODELING:")
print("   - Pipeline: Classification model (Critical/Moderate/Normal)")
print("   - Bearing: Binary classification (Healthy/Faulty)")
print("   - Pump: Regression for RUL prediction")
print("   - Turbine: RUL regression using LSTM/LightGBM")
print("   - Compressor: Efficiency degradation prediction + RUL estimation")

print("\n3. MONITORING PRIORITIES:")
priority_order = equipment_summary.sort_values('Anomaly Rate (%)', ascending=False)
print("   Equipment priority (by anomaly rate):")
for idx, row in priority_order.iterrows():
    print(f"   {idx+1}. {row['Equipment Type']}: {row['Anomaly Rate (%)']:.1f}% anomaly rate")

print(f"\nTotal Equipment Monitored: {equipment_summary['Unique Equipment'].sum()}")
print(f"Total Records Analyzed: {equipment_summary['Total Records'].sum():,}")
print("\nAll 5 equipment types analyzed successfully!")
