# BA2 Project Structure

## 📁 Cấu trúc Project

```
Final BA2/
├── pipelines/              # Pipeline cho từng thiết bị (tích hợp EDA + Preprocessing + Training)
│   ├── bearing_pipeline.py
│   ├── compressor_pipeline.py
│   ├── corrosion_pipeline.py
│   ├── pump_pipeline.py
│   └── turbine_pipeline.py
│
├── notebooks/              # EDA notebooks (exploratory + preprocessing đặc thù)
│   ├── Multi_Equipment_EDA.ipynb
│   ├── Pipeline_Corrosion_EDA.ipynb
│   └── Pipeline_downtime.ipynb
│
├── models/                 # Saved models và evaluation
│   ├── saved_models/
│   ├── metrics/
│   └── evaluation_plots/
│
├── scripts/               # Utility scripts
│   ├── convert_*.py      # Data conversion scripts
│   ├── copy_to_processed.py
│   └── synthetic_data_generator.py
│
├── converted_data/
│   ├── extracted/        # Raw extracted data
│   └── processed/        # Processed data ready for modeling
│
└── MVP/                  # Web interface
    ├── Web_tinh/
    └── responsive/
```

---

## 🎯 Design Philosophy

### **Keep It Simple & Integrated**

Không tách preprocessing thành module riêng vì:

1. **Mỗi thiết bị có đặc thù riêng:**
   - Bearing: Vibration analysis, fault detection
   - Turbine: RUL calculation, degradation
   - Pipeline: Corrosion rate, thickness loss
   - Pump: Flow patterns, cavitation
   - Compressor: Pressure/temperature

2. **Pipeline tích hợp sẵn:**
   ```python
   # Mỗi pipeline đã tích hợp:
   Data Loading → EDA → Preprocessing → Feature Engineering → Training → Evaluation
   ```

3. **Tránh over-engineering:**
   - Không cần abstraction layer phức tạp
   - Preprocessing gắn liền với domain logic
   - Dễ đọc, dễ maintain hơn

---

## 🔄 Workflow cho mỗi Equipment

### 1. **EDA Phase** (Notebooks)
```
notebooks/Multi_Equipment_EDA.ipynb
├── Load data
├── Exploratory analysis
├── Identify patterns
└── Domain-specific insights
```

### 2. **Pipeline Execution**
```
pipelines/bearing_pipeline.py
├── Load processed data
├── Preprocessing (integrated)
│   ├── Handle missing values
│   ├── Feature engineering (domain-specific)
│   └── Scaling
├── Model training
└── Evaluation & save
```

### 3. **Prediction & Deployment**
```
generate_predictions.py
├── Load trained models
├── Generate predictions
└── Save results for dashboard
```

---

## 📊 So sánh với BA (Business Analytics) Project

| Aspect | BA Project | BA2 Project |
|--------|-----------|-------------|
| **Scope** | Single problem (time series forecast) | Multiple equipment types |
| **Data** | Homogeneous (sales data) | Heterogeneous (sensors, vibration, etc) |
| **Preprocessing** | General, reusable | Domain-specific per equipment |
| **Structure** | `src/` modular pipeline | Equipment-specific pipelines |
| **Complexity** | Simple, one pipeline | Multiple specialized pipelines |
| **Best approach** | Centralized preprocessing | Integrated preprocessing |

---

## ✅ Best Practices

### DO:
- ✅ Keep preprocessing integrated in pipelines
- ✅ Document domain-specific logic in notebooks
- ✅ Use utility scripts for data conversion only
- ✅ Each pipeline is self-contained

### DON'T:
- ❌ Don't create generic preprocessing module
- ❌ Don't over-abstract domain logic
- ❌ Don't duplicate code across pipelines (use small helper functions if needed)
- ❌ Don't force all equipment into same preprocessing flow

---

## 🚀 Running the Project

```bash
# 1. Convert raw data
python scripts/run_all_converters.py

# 2. Run EDA (optional, for exploration)
jupyter notebook notebooks/Multi_Equipment_EDA.ipynb

# 3. Run individual pipeline
python pipelines/bearing_pipeline.py

# 4. Run all pipelines
python RUN_ALL_PIPELINES.py

# 5. Generate predictions
python generate_predictions.py
```

---

## 📝 When to Create Helper Functions?

Only create small helper functions in `scripts/` for:
- ✅ Data format conversion (CSV, MAT, XLSX)
- ✅ File I/O operations
- ✅ Synthetic data generation
- ✅ Simple utilities (not business logic)

**NOT for:**
- ❌ Feature engineering (keep in pipelines)
- ❌ Domain-specific preprocessing
- ❌ Model training logic

---

## 🎓 Summary

**Philosophy:** "Integration over Abstraction"

- Simpler structure
- Clearer ownership (one pipeline = one equipment)
- Easier to understand and modify
- Less overhead, more maintainable
- Domain expertise stays with domain code

**Remember:** The best code is the code you don't have to write! 🎉
