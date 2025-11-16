# Models Directory

This directory contains all trained models, evaluation results, and modeling notebooks.

## Directory Structure

```
models/
├── saved_models/          # Trained model files (.pkl, .txt)
│   ├── pipeline/          # Pipeline corrosion risk model
│   ├── turbine/           # Turbine RUL prediction model
│   ├── compressor/        # Compressor efficiency/RUL/anomaly models
│   ├── bearing/           # Bearing anomaly detection (planned)
│   └── pump/              # Pump health prediction (planned)
├── evaluation_plots/      # Model performance visualizations
│   ├── pipeline/          # Confusion matrix, ROC curve, SHAP
│   ├── turbine/           # Actual vs Predicted, SHAP, feature importance
│   ├── compressor/        # 3 model evaluations
│   ├── bearing/           # Anomaly plots
│   └── pump/              # Health prediction plots
├── metrics/               # Performance metrics (JSON)
│   ├── pipeline/          # Accuracy, precision, recall, F1
│   ├── turbine/           # R², RMSE, MAE, MAPE
│   ├── compressor/        # Multi-model metrics
│   ├── bearing/           # Anomaly detection metrics
│   └── pump/              # Health prediction metrics
└── notebooks/             # Modeling notebooks
    ├── Pipeline_Corrosion_Modeling.ipynb
    ├── Turbine_RUL_Modeling.ipynb
    └── Compressor_Modeling.ipynb
```

## Model Summary

### Pipeline Corrosion Risk Classifier

- **Algorithm**: LightGBM
- **Task**: 3-class classification (Critical/Moderate/Normal)
- **Performance**: 92.3% accuracy
- **Status**: ✅ Deployed

### Turbine RUL Predictor

- **Algorithm**: LightGBM
- **Task**: Regression (predict remaining useful life in cycles)
- **Performance**: R²=0.847, RMSE=30.2 cycles
- **Status**: ✅ Deployed

### Compressor Multi-Model System

- **Models**: 3 (Efficiency, RUL, Anomaly)
- **Algorithms**: LightGBM for all 3
- **Performance**:
  - Efficiency: R²=0.82
  - RUL: RMSE=245 days
  - Anomaly: F1=0.91
- **Status**: Ready (notebook complete)

### Bearing Anomaly Detector

- **Algorithm**: Rule-based (supervised learning planned)
- **Status**: Needs failure labels

### Pump Health Predictor

- **Algorithm**: Rule-based (supervised learning planned)
- **Status**: Needs failure labels

## Usage

### Load a Trained Model

```python
import pickle

# For PKL files
with open('saved_models/pipeline/pipeline_corrosion_model.pkl', 'rb') as f:
    model = pickle.load(f)

# For LightGBM text files
import lightgbm as lgb
model = lgb.Booster(model_file='saved_models/turbine/turbine_rul_model.txt')
```

### View Evaluation Metrics

```python
import json

with open('metrics/pipeline/model_info.json', 'r') as f:
    metrics = json.load(f)
print(f"Accuracy: {metrics['accuracy']:.2%}")
```

### Run Modeling Notebooks

```bash
# Open in Jupyter or VS Code
jupyter notebook notebooks/Turbine_RUL_Modeling.ipynb
```

## Model Artifacts

Each model directory contains:

- **saved_models/**: Serialized model files
- **evaluation_plots/**:
  - `actual_vs_predicted.png`
  - `feature_importance.png`
  - `shap_summary.png`
  - `confusion_matrix.png` (classification)
  - `residual_plot.png` (regression)
- **metrics/**:
  - `model_info.json` (metadata)
  - `metrics.json` (performance)
  - `predictions.csv` (test set predictions)

## Retraining Models

To retrain a model:

1. Open the corresponding notebook in `notebooks/`
2. Run all cells to retrain with updated data
3. Models are saved automatically to `saved_models/`
4. Evaluation plots saved to `evaluation_plots/`
5. Metrics updated in `metrics/`

## Adding New Models

1. Create subdirectory in each category (saved_models, evaluation_plots, metrics)
2. Create modeling notebook in `notebooks/`
3. Follow existing naming convention: `{EquipmentType}_Modeling.ipynb`
4. Save model artifacts to appropriate subdirectories

---

**Last Updated**: November 15, 2025
**Total Models**: 5 (3 deployed, 2 planned)
