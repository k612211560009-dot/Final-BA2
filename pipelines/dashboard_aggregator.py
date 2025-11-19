"""
Dashboard Aggregator - Multi-Equipment Summary for PdM Dashboard

This pipeline:
1. Reads all feature files (bearing, pump, corrosion, turbine if exists)
2. Extracts latest health metrics for each equipment
3. Maps health/risk metrics to standardized risk levels (Critical/High/Medium/Low)
4. Estimates days to maintenance from degradation trends
5. Outputs equipment_summary.csv and alerts_summary.csv

Author: Generated from PdM Architecture Design  
Date: November 13, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE_DIR / "data/features"
DASHBOARD_DIR = BASE_DIR / "data/dashboard"
METADATA_DIR = BASE_DIR / "supplement_data/metadata"
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print(" "*20 + "DASHBOARD AGGREGATOR")
print("="*70)

# 1. LOAD ALL FEATURE FILES

print("\n Loading feature files...")

# Dictionary to store all equipment data
equipment_data = {}

# Load bearing features
bearing_path = FEATURES_DIR / "bearing_features.csv"
if bearing_path.exists():
    bearing_df = pd.read_csv(bearing_path)
    # Bearing data doesn't have timestamp - use file_source or create synthetic
    if 'timestamp' not in bearing_df.columns:
        # Create synthetic timestamp based on row order (assuming chronological)
        bearing_df['timestamp'] = pd.Timestamp('2023-01-01') + pd.to_timedelta(bearing_df.index * 10, unit='m')
    else:
        bearing_df['timestamp'] = pd.to_datetime(bearing_df['timestamp'])
    equipment_data['bearing'] = bearing_df
    print(f"  Bearing features: {bearing_df.shape} - "
          f"{bearing_df['equipment_id'].nunique()} equipment")
else:
    print("  Bearing features not found")

# Load pump features
pump_path = FEATURES_DIR / "pump_features.csv"
if pump_path.exists():
    pump_df = pd.read_csv(pump_path)
    pump_df['timestamp'] = pd.to_datetime(pump_df['timestamp'])
    equipment_data['pump'] = pump_df
    print(f"  Pump features: {pump_df.shape} - "
          f"{pump_df['equipment_id'].nunique()} equipment")
else:
    print("  Pump features not found")

# Load corrosion features
corrosion_path = FEATURES_DIR / "corrosion_features.csv"
if corrosion_path.exists():
    corrosion_df = pd.read_csv(corrosion_path)
    corrosion_df['timestamp'] = pd.to_datetime(corrosion_df['timestamp'])
    equipment_data['corrosion'] = corrosion_df
    print(f"  Corrosion features: {corrosion_df.shape} - "
          f"{corrosion_df['equipment_id'].nunique()} equipment")
else:
    print("  Corrosion features not found")

# Load turbine features (if exists)
turbine_path = FEATURES_DIR / "turbine_features.csv"
if turbine_path.exists():
    turbine_df = pd.read_csv(turbine_path)
    turbine_df['timestamp'] = pd.to_datetime(turbine_df['timestamp'])
    equipment_data['turbine'] = turbine_df
    print(f"  Turbine features: {turbine_df.shape} - "
          f"{turbine_df['equipment_id'].nunique()} equipment")
else:
    print("  Turbine features not found (optional)")

# Load compressor features (if exists)
compressor_path = FEATURES_DIR / "compressor_features.csv"
if compressor_path.exists():
    compressor_df = pd.read_csv(compressor_path)
    compressor_df['timestamp'] = pd.to_datetime(compressor_df['timestamp'])
    equipment_data['compressor'] = compressor_df
    print(f"  Compressor features: {compressor_df.shape} - "
          f"{compressor_df['equipment_id'].nunique()} equipment")
else:
    print("  Compressor features not found (optional)")

# Load equipment master for metadata
equipment_master = pd.read_csv(METADATA_DIR / "equipment_master.csv")
print(f"  Equipment master: {equipment_master.shape}")

# 2. EXTRACT LATEST METRICS PER EQUIPMENT

print("\n Extracting latest metrics...")

summary_records = []

# Process bearing equipment
if 'bearing' in equipment_data:
    bearing_latest = equipment_data['bearing'].sort_values('timestamp').groupby('equipment_id').last()
    
    for equipment_id, row in bearing_latest.iterrows():
        # Estimate RUL from health_index if not available
        health = row.get('health_index', 0.5)
        if 'rul_days' in row and pd.notna(row['rul_days']):
            rul = row['rul_days']
        else:
            # Estimate RUL from health: health 1.0 = 365 days, health 0.0 = 0 days
            rul = health * 365
        
        summary_records.append({
            'equipment_id': equipment_id,
            'equipment_type': 'Bearing',
            'current_health': health,
            'primary_metric': health,
            'secondary_metric': row.get('rms', np.nan),
            'rul_days': rul,
            'is_anomaly': row.get('is_anomaly', False),
            'last_updated': row['timestamp']
        })

# Process pump equipment
if 'pump' in equipment_data:
    pump_latest = equipment_data['pump'].sort_values('timestamp').groupby('equipment_id').last()
    
    for equipment_id, row in pump_latest.iterrows():
        summary_records.append({
            'equipment_id': equipment_id,
            'equipment_type': 'Pump',
            'current_health': row.get('health_index', np.nan),
            'primary_metric': row.get('efficiency_normalized', np.nan),
            'secondary_metric': row.get('seal_condition_score', np.nan),
            'rul_days': row.get('rul_days', np.nan),
            'is_anomaly': row.get('is_anomaly', False),
            'last_updated': row['timestamp']
        })

# Process corrosion equipment
if 'corrosion' in equipment_data:
    corrosion_latest = equipment_data['corrosion'].sort_values('timestamp').groupby('equipment_id').last()
    
    for equipment_id, row in corrosion_latest.iterrows():
        # Convert risk_score to health_index (invert: high risk = low health)
        health_from_risk = 1.0 - (row.get('risk_score', 50) / 100.0)
        
        summary_records.append({
            'equipment_id': equipment_id,
            'equipment_type': 'Pipeline',
            'current_health': health_from_risk,
            'primary_metric': row.get('risk_score', np.nan),
            'secondary_metric': row.get('safety_margin_percent', np.nan),
            'rul_days': row.get('remaining_life_years', np.nan) * 365 if pd.notna(row.get('remaining_life_years')) else np.nan,
            'is_anomaly': (row.get('condition', 'Normal') == 'Critical'),
            'last_updated': row['timestamp']
        })

# Process turbine equipment (if exists)
if 'turbine' in equipment_data:
    turbine_latest = equipment_data['turbine'].sort_values('timestamp').groupby('equipment_id').last()
    
    for equipment_id, row in turbine_latest.iterrows():
        # Convert RUL from cycles to days (assuming 1 cycle = 1 hour flight)
        rul_cycles = row.get('rul_actual', np.nan)
        rul_days = (rul_cycles / 24.0) if pd.notna(rul_cycles) else np.nan
        
        summary_records.append({
            'equipment_id': equipment_id,
            'equipment_type': 'Turbine',
            'current_health': row.get('health_index', np.nan),
            'primary_metric': row.get('health_index', np.nan),
            'secondary_metric': row.get('rul_actual', np.nan),  # Store RUL cycles as secondary metric
            'rul_days': rul_days,
            'is_anomaly': row.get('is_anomaly', False),
            'last_updated': row['timestamp']
        })

# Process compressor equipment (if exists)
if 'compressor' in equipment_data:
    compressor_latest = equipment_data['compressor'].sort_values('timestamp').groupby('equipment_id').last()
    
    for equipment_id, row in compressor_latest.iterrows():
        summary_records.append({
            'equipment_id': equipment_id,
            'equipment_type': 'Compressor',
            'current_health': row.get('health_index', np.nan),
            'primary_metric': row.get('efficiency_normalized', np.nan),
            'secondary_metric': row.get('vibration_rms_mms', np.nan),
            'rul_days': row.get('rul_days', np.nan),
            'is_anomaly': row.get('is_anomaly', False),
            'last_updated': row['timestamp']
        })

# Create summary DataFrame
summary_df = pd.DataFrame(summary_records)
print(f"  Extracted metrics for {len(summary_df)} equipment")

# 3. ENRICH WITH EQUIPMENT METADATA

print("\n Enriching with equipment metadata...")

# Map location and other metadata from equipment_master
summary_enriched = summary_df.merge(
    equipment_master[['equipment_id', 'location', 'manufacturer', 'installation_date']],
    on='equipment_id',
    how='left'
)

print(f"  Enriched with location and manufacturer info")

# 4. MAP TO STANDARDIZED RISK LEVELS

print("\n Mapping to standardized risk levels...")

def map_health_to_risk_level(health):
    """
    Map health index (0-1) to risk level
    - Critical: health < 0.3
    - High: 0.3 <= health < 0.5
    - Medium: 0.5 <= health < 0.7
    - Low: health >= 0.7
    """
    if pd.isna(health):
        return 'Unknown'
    elif health < 0.3:
        return 'Critical'
    elif health < 0.5:
        return 'High'
    elif health < 0.7:
        return 'Medium'
    else:
        return 'Low'

summary_enriched['risk_level'] = summary_enriched['current_health'].apply(map_health_to_risk_level)

print(f"  Risk level distribution:")
for level, count in summary_enriched['risk_level'].value_counts().items():
    pct = count / len(summary_enriched) * 100
    print(f"    {level}: {count} ({pct:.1f}%)")

# 5. ESTIMATE DAYS TO MAINTENANCE

print("\n Estimating days to maintenance...")

def estimate_days_to_maintenance(row):
    """
    Estimate days until maintenance required
    Priority: use rul_days if available, else estimate from health
    """
    if pd.notna(row['rul_days']) and row['rul_days'] > 0:
        return min(row['rul_days'], 365 * 5)  # Cap at 5 years
    else:
        # Estimate based on health index
        health = row['current_health']
        if pd.isna(health):
            return 365  # Default to 1 year
        elif health < 0.3:
            return 7  # Critical: within 1 week
        elif health < 0.5:
            return 30  # High: within 1 month
        elif health < 0.7:
            return 90  # Medium: within 3 months
        else:
            return 180  # Low: within 6 months

summary_enriched['days_to_maintenance'] = summary_enriched.apply(
    estimate_days_to_maintenance, axis=1
)

print(f"  Average days to maintenance: {summary_enriched['days_to_maintenance'].mean():.0f}")

# 6. SELECT FINAL FEATURES FOR EQUIPMENT SUMMARY

print("\n Preparing equipment summary...")

equipment_summary = summary_enriched[[
    'equipment_id',
    'equipment_type',
    'location',
    'manufacturer',
    'installation_date',
    'current_health',
    'risk_level',
    'days_to_maintenance',
    'primary_metric',
    'secondary_metric',
    'is_anomaly',
    'last_updated'
]].copy()

# Round numeric columns
equipment_summary['current_health'] = equipment_summary['current_health'].round(3)
equipment_summary['primary_metric'] = equipment_summary['primary_metric'].round(2)
equipment_summary['secondary_metric'] = equipment_summary['secondary_metric'].round(2)
equipment_summary['days_to_maintenance'] = equipment_summary['days_to_maintenance'].round(0).astype(int)

# Sort by risk level (Critical first)
risk_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3, 'Unknown': 4}
equipment_summary['_risk_sort'] = equipment_summary['risk_level'].map(risk_order)
equipment_summary = equipment_summary.sort_values('_risk_sort').drop(columns=['_risk_sort'])

print(f"  Equipment summary prepared: {equipment_summary.shape}")

# 7. CREATE ALERTS SUMMARY (HIGH PRIORITY ONLY)

print("\n Creating alerts summary...")

# Filter for Critical and High risk
alerts_df = equipment_summary[
    equipment_summary['risk_level'].isin(['Critical', 'High'])
].copy()

# Add alert priority and recommended action
def generate_alert_priority(row):
    """Generate alert priority based on risk and anomaly status"""
    if row['risk_level'] == 'Critical':
        return 'P1 - Immediate' if row['is_anomaly'] else 'P2 - Urgent'
    elif row['risk_level'] == 'High':
        return 'P2 - Urgent' if row['is_anomaly'] else 'P3 - High'
    else:
        return 'P4 - Medium'

def generate_recommended_action(row):
    """Generate recommended maintenance action"""
    if row['risk_level'] == 'Critical':
        if row['equipment_type'] == 'Bearing':
            return 'Replace bearing immediately - high vibration detected'
        elif row['equipment_type'] == 'Pump':
            return 'Inspect seal and bearings - efficiency degraded'
        elif row['equipment_type'] == 'Pipeline':
            return 'Emergency inspection - critical corrosion level'
        else:
            return 'Immediate inspection and repair required'
    elif row['risk_level'] == 'High':
        if row['equipment_type'] == 'Bearing':
            return 'Schedule bearing replacement within 30 days'
        elif row['equipment_type'] == 'Pump':
            return 'Schedule seal inspection and lubrication check'
        elif row['equipment_type'] == 'Pipeline':
            return 'Plan pipeline segment replacement or coating repair'
        else:
            return 'Schedule preventive maintenance'
    else:
        return 'Monitor condition'

alerts_df['alert_priority'] = alerts_df.apply(generate_alert_priority, axis=1)
alerts_df['recommended_action'] = alerts_df.apply(generate_recommended_action, axis=1)

# Reorder columns
alerts_summary = alerts_df[[
    'alert_priority',
    'equipment_id',
    'equipment_type',
    'location',
    'risk_level',
    'current_health',
    'days_to_maintenance',
    'is_anomaly',
    'recommended_action',
    'last_updated'
]]

# Sort by priority
priority_order = {'P1 - Immediate': 0, 'P2 - Urgent': 1, 'P3 - High': 2, 'P4 - Medium': 3}
alerts_summary['_priority_sort'] = alerts_summary['alert_priority'].map(priority_order)
alerts_summary = alerts_summary.sort_values('_priority_sort').drop(columns=['_priority_sort'])

print(f"  Alerts summary created: {len(alerts_summary)} high-priority alerts")

# 8. SAVE OUTPUTS

print("\n Saving outputs...")

# Save equipment summary
equipment_summary_path = DASHBOARD_DIR / "equipment_summary.csv"
equipment_summary.to_csv(equipment_summary_path, index=False)
print(f"  Equipment summary: {equipment_summary_path}")

# Save alerts summary
alerts_summary_path = DASHBOARD_DIR / "alerts_summary.csv"
alerts_summary.to_csv(alerts_summary_path, index=False)
print(f"  Alerts summary: {alerts_summary_path}")

# 9. GENERATE DASHBOARD STATISTICS

print("\n" + "="*70)
print(" "*20 + "DASHBOARD STATISTICS")
print("="*70)

print(f"\n Overall Equipment Status:")
print(f"  Total equipment monitored: {len(equipment_summary)}")
print(f"  Equipment types: {equipment_summary['equipment_type'].nunique()}")
print(f"    - {', '.join(equipment_summary['equipment_type'].unique())}")

print(f"\n Risk Distribution:")
for level, count in equipment_summary['risk_level'].value_counts().items():
    pct = count / len(equipment_summary) * 100
    print(f"  {level}: {count} equipment ({pct:.1f}%)")

print(f"\n Active Anomalies:")
anomaly_count = equipment_summary['is_anomaly'].sum()
print(f"  Equipment with anomalies: {anomaly_count} ({anomaly_count/len(equipment_summary)*100:.1f}%)")

print(f"\n Maintenance Schedule:")
maintenance_7d = len(equipment_summary[equipment_summary['days_to_maintenance'] <= 7])
maintenance_30d = len(equipment_summary[equipment_summary['days_to_maintenance'] <= 30])
maintenance_90d = len(equipment_summary[equipment_summary['days_to_maintenance'] <= 90])

print(f"  Within 7 days: {maintenance_7d} equipment")
print(f"  Within 30 days: {maintenance_30d} equipment")
print(f"  Within 90 days: {maintenance_90d} equipment")

print(f"\n Alert Summary:")
print(f"  Total alerts: {len(alerts_summary)}")
if len(alerts_summary) > 0:
    for priority, count in alerts_summary['alert_priority'].value_counts().items():
        print(f"    {priority}: {count} alerts")

print(f"\n Top 3 Critical Equipment:")
critical_equipment = equipment_summary[equipment_summary['risk_level'] == 'Critical'].head(3)
if len(critical_equipment) > 0:
    for idx, row in critical_equipment.iterrows():
        print(f"  {row['equipment_id']} ({row['equipment_type']}) - "
              f"Health: {row['current_health']:.3f}, "
              f"Location: {row['location']}, "
              f"Days to maintenance: {row['days_to_maintenance']}")
else:
    print("  None - all equipment in good condition!")

print("\n" + "="*70)
print(" DASHBOARD AGGREGATION COMPLETE")
print("="*70)
print(f"\n Output files:")
print(f"  - {equipment_summary_path}")
print(f"  - {alerts_summary_path}")
print("\n Use these files for PdM dashboard visualization")
