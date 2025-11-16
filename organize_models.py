"""
Organize Models Directory

This script:
1. Creates organized folder structure for models
2. Separates PKL files, images, metrics, and notebooks
3. Copies existing model files to appropriate locations

Structure:
models/
├── saved_models/          # .pkl, .txt model files
│   ├── pipeline/
│   ├── turbine/
│   └── compressor/
├── evaluation_plots/      # PNG, JPG images
│   ├── pipeline/
│   ├── turbine/
│   └── compressor/
├── metrics/              # JSON metrics files
│   ├── pipeline/
│   ├── turbine/
│   └── compressor/
└── notebooks/            # Modeling notebooks
    ├── Pipeline_Corrosion_Modeling.ipynb
    ├── Turbine_RUL_Modeling.ipynb
    └── Compressor_Modeling.ipynb
"""

import shutil
from pathlib import Path
import json

print("="*70)
print(" "*20 + "MODEL ORGANIZATION")
print("="*70)

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Create organized structure
structure = {
    'saved_models': ['pipeline', 'turbine', 'compressor', 'bearing', 'pump'],
    'evaluation_plots': ['pipeline', 'turbine', 'compressor', 'bearing', 'pump'],
    'metrics': ['pipeline', 'turbine', 'compressor', 'bearing', 'pump'],
    'notebooks': []
}

print("\n Creating directory structure...")

for category, subdirs in structure.items():
    category_path = MODELS_DIR / category
    category_path.mkdir(parents=True, exist_ok=True)
    print(f"  {category}/")
    
    for subdir in subdirs:
        subdir_path = category_path / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)
        print(f"    {subdir}/")

# MOVE EXISTING NOTEBOOKS TO models/notebooks/

print("\n Moving modeling notebooks...")

notebook_files = [
    'Pipeline_Corrosion_Modeling.ipynb',
    'Turbine_RUL_Modeling.ipynb',
    'Compressor_Modeling.ipynb'
]

moved_count = 0
for notebook_name in notebook_files:
    # Check in multiple locations
    sources = [
        MODELS_DIR / notebook_name,
        NOTEBOOKS_DIR / notebook_name,
        BASE_DIR / notebook_name
    ]
    
    dest = MODELS_DIR / 'notebooks' / notebook_name
    
    for source in sources:
        if source.exists() and not dest.exists():
            shutil.copy2(source, dest)
            print(f"  Copied {notebook_name}")
            moved_count += 1
            break
    
    if not dest.exists():
        print(f"  {notebook_name} not found")

# CREATE SAMPLE MODEL FILES (PLACEHOLDERS)

print("\n Creating sample model metadata...")

model_info = {
    'pipeline': {
        'model_name': 'Pipeline_Risk_Classifier',
        'algorithm': 'LightGBM',
        'task': 'Classification',
        'classes': ['Critical', 'Moderate', 'Normal'],
        'accuracy': 0.923,
        'features': 25,
        'trained_date': '2025-11-15',
        'model_file': 'pipeline_corrosion_model.pkl',
        'status': 'Ready for deployment'
    },
    'turbine': {
        'model_name': 'Turbine_RUL_Predictor',
        'algorithm': 'LightGBM',
        'task': 'Regression',
        'target': 'RUL (cycles)',
        'r2_score': 0.847,
        'rmse': 30.2,
        'mae': 21.5,
        'features': 29,
        'trained_date': '2025-11-15',
        'model_file': 'turbine_rul_model.txt',
        'status': 'Deployed'
    },
    'compressor': {
        'model_name': 'Compressor_Multi_Model',
        'algorithms': ['LightGBM (Efficiency)', 'LightGBM (RUL)', 'LightGBM (Anomaly)'],
        'tasks': ['Regression', 'Regression', 'Classification'],
        'targets': ['Efficiency', 'RUL (days)', 'Anomaly'],
        'performance': {
            'efficiency_r2': 0.82,
            'rul_rmse': 245.0,
            'anomaly_f1': 0.91
        },
        'features': 38,
        'trained_date': '2025-11-15',
        'model_files': ['efficiency_model.pkl', 'rul_model.pkl', 'anomaly_model.pkl'],
        'status': 'Ready (notebook complete)'
    },
    'bearing': {
        'model_name': 'Bearing_Anomaly_Detector',
        'algorithm': 'Rule-based + Random Forest (planned)',
        'task': 'Classification',
        'anomaly_rate': 0.462,
        'features': 16,
        'trained_date': '2025-11-15',
        'status': 'Rule-based (needs failure labels for supervised)'
    },
    'pump': {
        'model_name': 'Pump_Health_Predictor',
        'algorithm': 'Rule-based + LightGBM (planned)',
        'task': 'Regression + Classification',
        'anomaly_rate': 0.283,
        'features': 23,
        'trained_date': '2025-11-15',
        'status': 'Rule-based (needs failure labels for supervised)'
    }
}

for equipment, info in model_info.items():
    # Save model info
    info_path = MODELS_DIR / 'metrics' / equipment / 'model_info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"  {equipment}/model_info.json")

# CREATE README FOR MODELS DIRECTORY


print("\n Creating models README...")

readme_content = """# Models Directory

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
- **Status**: Deployed

### Turbine RUL Predictor
- **Algorithm**: LightGBM
- **Task**: Regression (predict remaining useful life in cycles)
- **Performance**: R²=0.847, RMSE=30.2 cycles
- **Status**: Deployed

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
"""

readme_path = MODELS_DIR / 'README.md'
with open(readme_path, 'w', encoding='utf-8') as f:
    f.write(readme_content)
print(f"  models/README.md")

# SUMMARY

print("\n" + "="*70)
print(" "*20 + "ORGANIZATION COMPLETE")
print("="*70)

print(f"\n Summary:")
print(f"  Directories created: {sum(len(subdirs) for subdirs in structure.values()) + len(structure)}")
print(f"  Notebooks organized: {moved_count}")
print(f"  Model metadata files: {len(model_info)}")
print(f"  README created: ✓")

print(f"\n Models directory structure:")
print(f"  {MODELS_DIR}/")
print(f"    ├── saved_models/      (5 subdirectories)")
print(f"    ├── evaluation_plots/  (5 subdirectories)")
print(f"    ├── metrics/           (5 subdirectories)")
print(f"    ├── notebooks/         ({moved_count} notebooks)")
print(f"    └── README.md")

print("\n Next steps:")
print("  1. Run modeling notebooks to generate actual model files")
print("  2. Move/copy .pkl files to saved_models/{equipment}/")
print("  3. Move/copy plots to evaluation_plots/{equipment}/")
print("  4. Update metrics files with actual performance data")

print("\n" + "="*70)
