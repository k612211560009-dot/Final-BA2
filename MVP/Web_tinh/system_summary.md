# System Summary - Multi-Equipment PdM Platform

**Generated**: November 15, 2025  
**Status**: Fully Operational

---

## Quick Stats

| Metric               | Value                          |
| -------------------- | ------------------------------ |
| **Total Equipment**  | 121 units                      |
| **Equipment Types**  | 5 types                        |
| **Total Records**    | 253,076                        |
| **Active Pipelines** | 6 (5 equipment + 1 aggregator) |
| **Anomaly Rate**     | 37.2% overall                  |
| **Critical Alerts**  | 3 high-priority                |
| **Dashboard Status** | Live with real data            |

## Equipment Breakdown

| Type       | Units | Records | Anomaly % | Key Metrics                               |
| ---------- | ----- | ------- | --------- | ----------------------------------------- |
| Bearing    | 10    | 2,993   | 46.2%     | Vibration RMS, FFT, Health Index          |
| Pump       | 2     | 5,114   | 28.3%     | Efficiency, Seal Condition, RUL           |
| Pipeline   | 5     | 1,000   | 48.7%     | Corrosion Rate, Risk Score, Safety Margin |
| Turbine    | 101   | 33,729  | 1.3%      | 26 Sensors, RUL (cycles), Health Index    |
| Compressor | 3     | 210,240 | 9.3%      | Motor Performance, Efficiency, Vibration  |

## Data Flow Architecture

```
Raw Data Sources
    ↓
[5 Equipment Pipelines]
    ├─ bearing_pipeline.py      → bearing_features.csv (2,993 rows)
    ├─ pump_pipeline.py          → pump_features.csv (5,114 rows)
    ├─ corrosion_pipeline.py     → corrosion_features.csv (1,000 rows)
    ├─ turbine_pipeline.py       → turbine_features.csv (33,729 rows)
    └─ compressor_pipeline.py    → compressor_features.csv (210,240 rows)
    ↓
[Dashboard Aggregator]
    ├─ equipment_summary.csv (121 equipment × 12 metrics)
    └─ alerts_summary.csv (3 high-priority alerts)
    ↓
[Web Dashboard]
    ├─ load_data.py (CSV → JavaScript converter)
    ├─ data.js (real equipment data)
    └─ web.htm (visualization)
```

## System Components

### Completed Components

1. **Feature Engineering** (5 pipelines)

   - Bearing: Vibration analysis + FFT features
   - Pump: Efficiency + seal monitoring
   - Pipeline: Risk scoring + corrosion modeling
   - Turbine: Multi-sensor health index + RUL
   - Compressor: Motor performance + vibration + efficiency

2. **Dashboard System**

   - Unified aggregator (common schema)
   - Equipment summary (121 units)
   - Alert system (P1/P2/P3 priorities)
   - Web visualization (web.htm + data.js)
   - Real-time data integration

3. **Modeling & Analysis**

   - Multi-Equipment EDA (script + notebook)
   - Pipeline Corrosion Model (92.3% accuracy)
   - Turbine RUL Model (R²=0.847, RMSE~30 cycles)
   - Compressor Modeling Notebook (3 models ready)

4. **Automation**
   - RUN_ALL_PIPELINES.py orchestrator
   - Dashboard data refresh automation
   - Common schema across equipment types

### Planned Enhancements

- Model deployment (FastAPI/Flask)
- Real-time data ingestion
- Automated alerting (email/SMS)
- React-based dashboard
- Bearing/Pump supervised learning (need failure labels)

## Current System Performance

### Risk Distribution

- **Critical**: 1 equipment (0.8%)
- **High**: 2 equipment (1.7%)
- **Medium**: 115 equipment (95.0%)
- **Low**: 3 equipment (2.5%)

### Maintenance Schedule

- **Within 7 days**: 16 equipment
- **Within 30 days**: 17 equipment
- **Within 90 days**: 103 equipment

### Model Performance

| Model                | Equipment  | Metric    | Value            | Status   |
| -------------------- | ---------- | --------- | ---------------- | -------- |
| LightGBM Classifier  | Pipeline   | Accuracy  | 92.3%            | Deployed |
| LightGBM Regressor   | Turbine    | R² / RMSE | 0.847 / 30       | Deployed |
| LightGBM Multi-Model | Compressor | Models    | 3 (Eff/RUL/Anom) | Ready    |

---

## How to Use

### Quick Start (Demo)

```bash
# 1. Run all pipelines
python RUN_ALL_PIPELINES.py

# 2. Update dashboard
cd MVP/Web_tinh
python load_data.py

# 3. Open dashboard
# Open web.htm in browser or use Live Server
```

### Individual Pipeline Execution

```bash
cd pipelines

# Run specific equipment pipeline
python bearing_pipeline.py
python pump_pipeline.py
python corrosion_pipeline.py
python turbine_pipeline.py
python compressor_pipeline.py

# Aggregate for dashboard
python dashboard_aggregator.py
```

### Analysis & Modeling

```bash
# Run EDA
cd notebooks
python run_multi_equipment_eda.py

# Or open notebooks in Jupyter/VS Code
# - Multi_Equipment_EDA.ipynb
# - Turbine_RUL_Modeling.ipynb
# - Compressor_Modeling.ipynb
# - Pipeline_Corrosion_Modeling.ipynb
```

## Key Output Files

### Feature Files (`data/features/`)

- `bearing_features.csv` (2,993 × 16)
- `pump_features.csv` (5,114 × 23)
- `corrosion_features.csv` (1,000 × 25)
- `turbine_features.csv` (33,729 × 29)
- `compressor_features.csv` (210,240 × 38)

### Dashboard Files (`data/dashboard/`)

- `equipment_summary.csv` (121 × 12)
- `alerts_summary.csv` (3 × 10)

### Web Dashboard (`MVP/Web_tinh/`)

- `web.htm` - Main dashboard page
- `data.js` - Real equipment data (121 units)
- `script.js` - Visualization logic
- `style.css` - Styling
- `load_data.py` - Data refresh script

### Models (`models/`)

- `pipeline_corrosion_model.pkl`
- `turbine_rul_model.txt`
- `{model}_evaluation/` folders with metrics, plots, SHAP

## Demo Workflows

### Option A: Quick (5-10 min)

1. Open `MVP/Web_tinh/web.htm`
2. Show KPI cards and risk distribution
3. Demonstrate filtering and alerts
4. Show model evaluation plots

### Option B: Detailed (15-20 min)

1. Explain system architecture
2. Run `RUN_ALL_PIPELINES.py` live
3. Execute `run_multi_equipment_eda.py`
4. Open modeling notebook (Turbine or Compressor)

### Option C: Full (30+ min)

- Combine A + B
- Deep dive into feature engineering
- SHAP analysis walkthrough
- Q&A and discussion

## Documentation

- **README.md**: Main project overview
- **docs/README_PdM_System.md**: Detailed system documentation
- **docs/MODEL_EVALUATION.md**: Comprehensive model evaluation (9000+ lines)
- **SYSTEM_SUMMARY.md**: This file (quick reference)

## Verification Checklist

- [x] All 5 equipment pipelines executable
- [x] Dashboard aggregator produces valid output
- [x] Equipment summary has 121 records
- [x] Web dashboard displays real data
- [x] RUN_ALL_PIPELINES.py runs without errors
- [x] EDA script produces comprehensive statistics
- [x] Modeling notebooks are complete
- [x] Documentation is up-to-date

## System Requirements

- Python 3.8+
- Key libraries: pandas, numpy, scikit-learn, lightgbm, shap
- Optional: Jupyter for notebooks
- Browser with JavaScript support for dashboard

---

## Contributors

- **Dashboard**: Downstream machine management (MVP/Web_tinh/)
- **Modeling**: LightGBM, SHAP analysis
- **Architecture**: Multi-layer pipeline design

**For detailed technical information, see `docs/MODEL_EVALUATION.md`**  
**For usage instructions, see `docs/README_PdM_System.md`**
