"""
Corrosion Pipeline - Feature Engineering for Pipeline Corrosion Data

This pipeline:
1. Loads cleaned pipeline corrosion data
2. Maps to equipment IDs from equipment_master.csv
3. Enriches with weather and operational context
4. Computes domain-specific features (corrosion rate, remaining life, risk score)
5. Classifies condition (Critical/Moderate/Normal)
6. Outputs corrosion_features.csv

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
PROCESSED_DIR = BASE_DIR / "converted_data/processed"
METADATA_DIR = BASE_DIR / "supplement_data/metadata"
FEATURES_DIR = BASE_DIR / "data/features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print(" "*20 + "CORROSION PIPELINE")
print("="*70)

# 1. LOAD DATA

print("\n Loading data...")

# Load corrosion data
corrosion_df = pd.read_csv(PROCESSED_DIR / "market_pipe_thickness_loss_dataset_clean.csv")
print(f"  Corrosion data: {corrosion_df.shape}")

# Load metadata
equipment_master = pd.read_csv(METADATA_DIR / "equipment_master.csv")
weather_data = pd.read_csv(METADATA_DIR / "weather_data.csv")
operational_context = pd.read_csv(METADATA_DIR / "operational_context.csv")

print(f"  Equipment master: {equipment_master.shape}")
print(f"  Weather data: {weather_data.shape}")
print(f"  Operational context: {operational_context.shape}")

# 2. MAP TO EQUIPMENT IDs

print("\n Mapping to equipment IDs...")

# Get pipeline equipment from master
pipeline_equipment = equipment_master[equipment_master['equipment_type'] == 'Pipeline Segment'].copy()
print(f"  Found {len(pipeline_equipment)} pipeline segments in equipment master")

# Create mapping function based on material and grade
def map_to_equipment_id(row):
    """Map corrosion record to equipment_id based on material and grade"""
    # Simple hash-based mapping (in production, use actual asset IDs)
    material_map = {
        'carbon steel': 'PIPE-001',
        'stainless steel': 'PIPE-002', 
        'pvc': 'PIPE-003',
        'hdpe': 'PIPE-004',
        'fiberglass': 'PIPE-005'
    }
    # Handle lowercase column names
    material = str(row['material']).lower() if 'material' in row else ''
    return material_map.get(material, 'PIPE-001')

corrosion_df['equipment_id'] = corrosion_df.apply(map_to_equipment_id, axis=1)
print(f"  Mapped {corrosion_df['equipment_id'].nunique()} unique equipment IDs")

# 3. FEATURE ENGINEERING - Domain-Specific Features

print("\n Engineering features...")

# 3.1 Corrosion Rate (mm/year)
corrosion_df['corrosion_rate_mm_year'] = corrosion_df['thickness_loss_mm'] / corrosion_df['time_years']

# 3.2 Remaining Thickness
corrosion_df['remaining_thickness_mm'] = corrosion_df['thickness_mm'] - corrosion_df['thickness_loss_mm']

# 3.3 Remaining Life Estimate (years)
# Assuming minimum safe thickness = 30% of original thickness
min_safe_thickness = corrosion_df['thickness_mm'] * 0.3
corrosion_df['remaining_life_years'] = (
    (corrosion_df['remaining_thickness_mm'] - min_safe_thickness) / 
    corrosion_df['corrosion_rate_mm_year']
)
corrosion_df['remaining_life_years'] = corrosion_df['remaining_life_years'].clip(lower=0)

# 3.4 Safety Margin (%)
corrosion_df['safety_margin_percent'] = (
    corrosion_df['remaining_thickness_mm'] / corrosion_df['thickness_mm']
) * 100

# 3.5 Pressure to Thickness Ratio (risk indicator)
corrosion_df['pressure_thickness_ratio'] = (
    corrosion_df['max_pressure_psi'] / corrosion_df['remaining_thickness_mm']
)

# 3.6 Loss Rate Severity (categorical)
def classify_loss_rate(rate):
    if rate < 0.5:
        return 'Low'
    elif rate < 1.5:
        return 'Moderate'
    else:
        return 'High'

corrosion_df['loss_rate_severity'] = corrosion_df['corrosion_rate_mm_year'].apply(classify_loss_rate)

# 3.7 Risk Score (0-100 composite score)
# Formula: weighted combination of multiple factors
def calculate_risk_score(row):
    """
    Composite risk score based on:
    - Corrosion rate (30%)
    - Safety margin (25%)
    - Remaining life (20%)
    - Pressure ratio (15%)
    - Material loss % (10%)
    """
    # Normalize each factor to 0-1 scale
    corr_rate_norm = min(row['corrosion_rate_mm_year'] / 3.0, 1.0)  # 3 mm/year = max
    safety_margin_norm = 1.0 - (row['safety_margin_percent'] / 100.0)  # Lower margin = higher risk
    remaining_life_norm = 1.0 - min(row['remaining_life_years'] / 20.0, 1.0)  # <20 years = risk
    pressure_ratio_norm = min(row['pressure_thickness_ratio'] / 200.0, 1.0)  # 200 = threshold
    material_loss_norm = row['material_loss_percent'] / 100.0
    
    # Weighted sum
    risk_score = (
        corr_rate_norm * 0.30 +
        safety_margin_norm * 0.25 +
        remaining_life_norm * 0.20 +
        pressure_ratio_norm * 0.15 +
        material_loss_norm * 0.10
    ) * 100
    
    return min(risk_score, 100.0)

corrosion_df['risk_score'] = corrosion_df.apply(calculate_risk_score, axis=1)

print(f"  Created 7 engineered features")
print(f"    - corrosion_rate_mm_year: mean={corrosion_df['corrosion_rate_mm_year'].mean():.2f}")
print(f"    - remaining_life_years: mean={corrosion_df['remaining_life_years'].mean():.2f}")
print(f"    - safety_margin_percent: mean={corrosion_df['safety_margin_percent'].mean():.2f}")
print(f"    - risk_score: mean={corrosion_df['risk_score'].mean():.2f}")

# 4. ENRICH WITH EXTERNAL DATA

print("\n Enriching with external data...")

# 4.1 Create synthetic timestamp for joining (based on time_years)
# Assuming data collection started on 2020-01-01
base_date = pd.Timestamp('2020-01-01')
corrosion_df['timestamp'] = base_date + pd.to_timedelta(corrosion_df['time_years'] * 365, unit='D')
corrosion_df['date'] = corrosion_df['timestamp'].dt.date

# 4.2 Join with weather data (by date)
weather_data['date'] = pd.to_datetime(weather_data['date']).dt.date
corrosion_enriched = corrosion_df.merge(
    weather_data[['date', 'ambient_temp_c', 'humidity_percent', 'rainfall_mm']],
    on='date',
    how='left'
)

print(f"  Joined with weather data")
print(f"    - Matched {corrosion_enriched['ambient_temp_c'].notna().sum()} records")

# 4.3 Join with operational context (by equipment_id and approximate timestamp)
operational_context['timestamp'] = pd.to_datetime(operational_context['timestamp'])
operational_context['date'] = operational_context['timestamp'].dt.date

# Aggregate operational context to daily level
ops_daily = operational_context.groupby(['equipment_id', 'date']).agg({
    'operating_speed_rpm': 'mean',
    'load_percent': 'mean',
    'operating_mode': lambda x: x.mode()[0] if len(x) > 0 else 'Normal'
}).reset_index()

corrosion_enriched = corrosion_enriched.merge(
    ops_daily,
    on=['equipment_id', 'date'],
    how='left'
)

print(f"  Joined with operational context")
print(f"    - Matched {corrosion_enriched['operating_speed_rpm'].notna().sum()} records")

# 5. CLASSIFY CONDITION BASED ON RISK SCORE

print("\n Classifying pipeline condition...")

def classify_condition_from_risk(risk_score):
    """
    Classify condition based on risk score:
    - Critical: risk_score >= 70
    - Moderate: 40 <= risk_score < 70
    - Normal: risk_score < 40
    """
    if risk_score >= 70:
        return 'Critical'
    elif risk_score >= 40:
        return 'Moderate'
    else:
        return 'Normal'

corrosion_enriched['condition_predicted'] = corrosion_enriched['risk_score'].apply(classify_condition_from_risk)

# Compare with actual condition
condition_comparison = pd.crosstab(
    corrosion_enriched['condition'], 
    corrosion_enriched['condition_predicted']
)
print(f"  Condition classification complete")
print(f"\n  Actual vs Predicted Condition:")
print(condition_comparison)

# 6. SELECT FINAL FEATURES

print("\n Selecting final features...")

final_features = [
    # Identifiers
    'equipment_id',
    'timestamp',
    
    # Original features (lowercase)
    'material',
    'grade',
    'pipe_size_mm',
    'thickness_mm',
    'max_pressure_psi',
    'time_years',
    'thickness_loss_mm',
    'material_loss_percent',
    'condition',  # Actual condition
    
    # Engineered features
    'corrosion_rate_mm_year',
    'remaining_thickness_mm',
    'remaining_life_years',
    'safety_margin_percent',
    'pressure_thickness_ratio',
    'loss_rate_severity',
    'risk_score',
    'condition_predicted',
    
    # External data
    'ambient_temp_c',
    'humidity_percent',
    'rainfall_mm',
    'operating_speed_rpm',
    'load_percent',
    'operating_mode'
]

corrosion_features = corrosion_enriched[final_features].copy()

print(f"  Selected {len(final_features)} features")

# 7. SAVE OUTPUT

print("\n Saving output...")

output_path = FEATURES_DIR / "corrosion_features.csv"
corrosion_features.to_csv(output_path, index=False)

print(f"  Saved to: {output_path}")
print(f"  Shape: {corrosion_features.shape}")

# 8. SUMMARY STATISTICS

print("\n Summary Statistics:")
print(f"  Total records: {len(corrosion_features)}")
print(f"  Unique equipment: {corrosion_features['equipment_id'].nunique()}")
print(f"  Date range: {corrosion_features['timestamp'].min()} to {corrosion_features['timestamp'].max()}")

print(f"\n  Condition Distribution:")
for condition, count in corrosion_features['condition'].value_counts().items():
    pct = count / len(corrosion_features) * 100
    print(f"    {condition}: {count} ({pct:.1f}%)")

print(f"\n  Risk Score Distribution:")
print(f"    Mean: {corrosion_features['risk_score'].mean():.2f}")
print(f"    Median: {corrosion_features['risk_score'].median():.2f}")
print(f"    Min: {corrosion_features['risk_score'].min():.2f}")
print(f"    Max: {corrosion_features['risk_score'].max():.2f}")

print(f"\n  High-Risk Pipelines (risk_score >= 70):")
high_risk = corrosion_features[corrosion_features['risk_score'] >= 70]
print(f"    Count: {len(high_risk)} ({len(high_risk)/len(corrosion_features)*100:.1f}%)")
if len(high_risk) > 0:
    print(f"    Equipment IDs: {high_risk['equipment_id'].unique().tolist()}")

print("\n" + "="*70)
print(" CORROSION PIPELINE COMPLETE")
print("="*70)
