# Multi-Layer Data Architecture for PdM

This document defines the **layered data organization** for the Predictive Maintenance system, following the guidance to **NOT merge everything into one table** but instead use **equipment-specific data marts** with unified dashboard aggregation.

## Architecture layers

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 3: DASHBOARD AGGREGATION (Unified Health View)              │
│  - equipment_summary.csv (equipment_id, health_index, risk_level)   │
│  - alerts_summary.csv (alert_id, equipment_id, severity, action)    │
└─────────────────────────────────────────────────────────────────────┘
                              ▲
                              │ (aggregation scripts)
                              │
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 2: EQUIPMENT-SPECIFIC FEATURES (Per Equipment Type)         │
│  - bearing_features.csv (equipment_id, vibration_rms, health_index)│
│  - pump_features.csv (equipment_id, efficiency, cavitation_index)  │
│  - corrosion_features.csv (equipment_id, corrosion_rate, RUL)      │
│  - turbine_features.csv (equipment_id, RUL, operating_mode)        │
└─────────────────────────────────────────────────────────────────────┘
                              ▲
                              │ (feature engineering pipelines)
                              │
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1: RAW & CONVERTED DATA + METADATA                          │
│  - converted_data/extracted/ (per-file vibration features)          │
│  - converted_data/processed/ (cleaned vibration, corrosion CSVs)    │
│  - data/metadata/ (equipment_master, sensor_metadata, etc.)         │
└─────────────────────────────────────────────────────────────────────┘
```

## Directory structure

```
d:/Final BA2/
├── raw_data/                     # Original .mat, .txt, .xlsx
├── converted_data/
│   ├── extracted/                # Per-file feature CSVs (cwru, cwru2, cmaps, pumps)
│   └── processed/                # Cleaned master CSVs (vibration_dataset_clean.csv, etc.)
├── data/
│   ├── metadata/                 # Supplementary CSVs (equipment_master, sensor_metadata, maintenance_schedule, failure_history, weather, operational_context)
│   ├── features/                 # LAYER 2: Equipment-specific feature datasets
│   │   ├── bearing_features.csv
│   │   ├── pump_features.csv
│   │   ├── corrosion_features.csv
│   │   └── turbine_features.csv
│   └── dashboard/                # LAYER 3: Aggregated views
│       ├── equipment_summary.csv
│       └── alerts_summary.csv
├── pipelines/                    # Feature engineering scripts per equipment
│   ├── bearing_pipeline.py
│   ├── pump_pipeline.py
│   ├── corrosion_pipeline.py
│   ├── turbine_pipeline.py
│   └── dashboard_aggregator.py
├── models/                       # Trained models per equipment type
│   ├── bearing_anomaly_model.pkl
│   ├── pump_rul_model.pkl
│   └── corrosion_classifier.pkl
└── dashboard/                    # Streamlit/Dash app
    └── app.py
```

---

## Pipeline contracts (per equipment type)

### Bearing pipeline

- **Input**: `converted_data/extracted/cwru/*.csv`, `data/metadata/equipment_master.csv`, `data/metadata/sensor_metadata.csv`, `data/metadata/operational_context.csv`
- **Processing**:
  1. Map file prefixes (B007, IR014, etc.) to `equipment_id` using metadata.
  2. Join with operational_context to add `operating_speed_rpm`, `load_percent`.
  3. Compute rolling statistics (rolling_mean_rms, rolling_std_rms over 10 windows).
  4. Compute trend features (rms_trend_slope via linear regression over last 20 windows).
  5. Compute bearing health index: `health_index = 1.0 - min(1.0, (rms / rms_threshold) * (kurtosis / kurtosis_threshold))`.
  6. Flag anomalies: `is_anomaly = (health_index < 0.5) | (kurtosis > 5.0)`.
- **Output**: `data/features/bearing_features.csv` with columns: `equipment_id, timestamp, rms, peak, kurtosis, crest_factor, rolling_mean_rms, rolling_std_rms, rms_trend_slope, health_index, is_anomaly, operating_speed_rpm, load_percent`

### Pump pipeline

- **Input**: `converted_data/extracted/pumps/*.csv`, metadata tables
- **Processing**:
  1. Map `Machine_ID` to `equipment_id`.
  2. Join with operational_context for speed/load.
  3. Compute derived metrics:
     - `efficiency_score = (flow_rate * discharge_pressure) / motor_power` (if available; else use proxy from value_ISO/value_DEMO).
     - `cavitation_index = NPSH_available / NPSH_required` (synthetic if NPSH not available).
     - `seal_condition_score = f(valueTEMP, vibration)` — higher temp + higher vibration = lower score.
  4. Compute health index from efficiency + seal condition + vibration.
  5. Estimate RUL via linear extrapolation of health_index degradation trend.
- **Output**: `data/features/pump_features.csv` with columns: `equipment_id, timestamp, efficiency_score, cavitation_index, seal_condition_score, health_index, estimated_rul_days, vibration_rms, temperature`

### Corrosion pipeline

- **Input**: `converted_data/processed/market_pipe_thickness_loss_dataset_clean.csv`, metadata
- **Processing**:
  1. Map rows to `equipment_id` (synthetic pipeline segment IDs).
  2. Join with weather_data to add `ambient_temp_c`, `humidity_percent`.
  3. Compute:
     - `corrosion_rate_mm_year = thickness_loss_mm / time_years`.
     - `remaining_life_years = (thickness_mm - min_thickness_required) / corrosion_rate_mm_year`.
     - `safety_margin_percent = (remaining_thickness_mm / original_thickness_mm) * 100`.
     - `risk_score = f(corrosion_rate, remaining_life, humidity, temperature)` — higher corrosion rate + lower remaining life + higher humidity = higher risk.
  4. Classify condition: `Critical` if remaining_life < 2 years, `Moderate` if 2-5 years, `Low` otherwise.
- **Output**: `data/features/corrosion_features.csv` with columns: `equipment_id, measurement_date, thickness_mm, thickness_loss_mm, corrosion_rate_mm_year, remaining_life_years, safety_margin_percent, risk_score, condition, ambient_temp_c, humidity_percent`

### Turbine pipeline

- **Input**: `converted_data/extracted/cmaps/*.csv`, RUL labels, metadata
- **Processing**:
  1. Parse sensor columns (0..25) and map to meaningful names (temp1, pressure1, etc. from CMAPSS readme).
  2. Map unit_id to `equipment_id`.
  3. Join RUL labels from `RUL_FD*.txt`.
  4. Compute cycle-based features (rolling mean/std over cycles).
  5. Convert cycles to approximate timestamps using operational_context.
  6. Train/load RUL regression model (LSTM or RandomForest).
- **Output**: `data/features/turbine_features.csv` with columns: `equipment_id, timestamp, cycle, sensor_readings (26 cols), rul_actual, rul_predicted, health_index`

---

## Dashboard aggregation

- **Input**: all `data/features/*.csv`
- **Processing**:
  1. Compute per-equipment current health_index (latest timestamp).
  2. Map health_index to risk_level: `Critical` (<0.3), `High` (0.3-0.5), `Medium` (0.5-0.7), `Low` (>0.7).
  3. Estimate `days_to_maintenance` from health degradation rate or RUL.
  4. Generate alerts for equipment with risk_level >= `High`.
- **Output**:
  - `data/dashboard/equipment_summary.csv`: `equipment_id, equipment_type, current_health_index, risk_level, days_to_maintenance, last_updated`
  - `data/dashboard/alerts_summary.csv`: `alert_id, equipment_id, equipment_type, severity, triggered_by (feature), recommended_action, timestamp`

---

## Next implementation steps

1. Build `pipelines/bearing_pipeline.py` (highest priority — most complete raw data).
2. Build `pipelines/corrosion_pipeline.py` (second — already has clean CSV).
3. Build `pipelines/pump_pipeline.py` (third — pump CSV available).
4. Build `pipelines/turbine_pipeline.py` (fourth — requires CMAPSS column mapping).
5. Build `pipelines/dashboard_aggregator.py` (final — reads all feature CSVs and produces summary).

## Usage

```bash
# Run pipelines in order
python pipelines/bearing_pipeline.py
python pipelines/corrosion_pipeline.py
python pipelines/pump_pipeline.py
python pipelines/turbine_pipeline.py

# Aggregate for dashboard
python pipelines/dashboard_aggregator.py

# Launch dashboard
streamlit run dashboard/app.py
```
