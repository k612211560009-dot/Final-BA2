# Data Preprocessing Guide

## 📁 Cấu trúc Module

```
scripts/
├── preprocessing.py          # Module xử lý chung (GENERAL)
└── copy_to_processed.py     # Script consolidate data

notebooks/
├── Preprocessing_Example.ipynb    # Ví dụ sử dụng
├── Multi_Equipment_EDA.ipynb      # EDA đặc thù cho từng thiết bị
└── Pipeline_Corrosion_EDA.ipynb   # EDA đặc thù cho pipeline
```

---

## 🎯 Nguyên tắc thiết kế

### ✅ Module `preprocessing.py` - Xử lý CHUNG
Chứa các hàm có thể **tái sử dụng** cho mọi loại data:

1. **Basic Cleaning**
   - Remove duplicates
   - Handle missing values
   - Remove outliers

2. **Scaling**
   - StandardScaler
   - MinMaxScaler
   - RobustScaler

3. **General Feature Engineering**
   - Rolling features (mean, std, min, max)
   - Lag features
   - Time-based split

4. **Utilities**
   - Data info
   - Save/load functions

### ✅ Notebook - Xử lý ĐẶC THÙ
Mỗi notebook xử lý **domain-specific logic**:

| Equipment | Đặc thù xử lý |
|-----------|---------------|
| **Bearing** | Vibration analysis, fault patterns, load impact |
| **Turbine** | RUL calculation, degradation curves, sensor fusion |
| **Pipeline** | Corrosion rate, thickness loss, environmental factors |
| **Compressor** | Pressure/temperature patterns, efficiency metrics |
| **Pump** | Flow rate analysis, cavitation detection |

---

## 🚀 Cách sử dụng

### Option 1: Quick Clean (Nhanh)

```python
from scripts.preprocessing import quick_clean

# Tự động xử lý các bước cơ bản
df_clean = quick_clean(
    df,
    remove_duplicates=True,
    handle_missing='drop',    # hoặc 'fill'
    remove_outliers=True,
    outlier_method='iqr'
)
```

### Option 2: Step by Step (Chi tiết)

```python
from scripts.preprocessing import DataPreprocessor

# Khởi tạo
preprocessor = DataPreprocessor()

# Bước 1: Xem thông tin data
preprocessor.get_data_info(df)

# Bước 2: Remove duplicates
df = preprocessor.remove_duplicates(df)

# Bước 3: Handle missing values
df = preprocessor.handle_missing_values(
    df, 
    strategy='fill',
    fill_method='mean'
)

# Bước 4: Remove outliers
df = preprocessor.remove_outliers(
    df,
    columns=['col1', 'col2'],
    method='iqr',
    threshold=1.5
)

# Bước 5: Scale features
X_scaled = preprocessor.fit_transform(X, method='standard')

# Bước 6: Create rolling features
df = preprocessor.create_rolling_features(
    df,
    column='temperature',
    windows=[3, 5, 10],
    functions=['mean', 'std']
)

# Bước 7: Save
preprocessor.save_processed_data(df, 'output.csv')
```

---

## 📊 Workflow Example

### 1. Load và Clean (Module chung)
```python
import pandas as pd
from scripts.preprocessing import DataPreprocessor

df = pd.read_csv('data.csv')
preprocessor = DataPreprocessor()

# Clean cơ bản
df = preprocessor.remove_duplicates(df)
df = preprocessor.handle_missing_values(df, strategy='drop')
```

### 2. Domain-Specific Processing (Notebook)
```python
# VÍ DỤ: Bearing-specific
if 'fault_type' in df.columns:
    # Tạo binary label
    df['is_faulty'] = df['fault_type'].apply(
        lambda x: 0 if x == 'Normal' else 1
    )
    
    # Tạo fault severity
    severity_map = {
        'B007': 1,   # Small crack
        'B014': 2,   # Medium crack
        'B021': 3    # Large crack
    }
    df['severity'] = df['fault_type'].map(severity_map).fillna(0)

# VÍ DỤ: Turbine-specific
if 'cycle' in df.columns:
    # Calculate RUL
    max_cycles = df.groupby('engine_id')['cycle'].transform('max')
    df['RUL'] = max_cycles - df['cycle']
    
    # Create degradation stage
    df['degradation_stage'] = pd.cut(
        df['RUL'],
        bins=[0, 50, 100, float('inf')],
        labels=['critical', 'warning', 'healthy']
    )
```

### 3. Feature Engineering (Module chung + Notebook)
```python
# Module chung - rolling features
df = preprocessor.create_rolling_features(
    df,
    column='temperature',
    windows=[5, 10],
    functions=['mean', 'std']
)

# Notebook - domain features
df['temp_pressure_ratio'] = df['temperature'] / (df['pressure'] + 1e-10)
df['vibration_level'] = df['rms'] * df['peak']
```

### 4. Scale và Save (Module chung)
```python
# Scale
X_scaled = preprocessor.fit_transform(X, method='standard')

# Save
preprocessor.save_processed_data(df_final, 'processed_data.csv')
```

---

## 📝 Best Practices

### ✅ DO

1. **Sử dụng module cho các bước chung:**
   - Duplicates, missing values, outliers
   - Standard scaling, normalization
   - Rolling/lag features

2. **Viết code đặc thù trong notebook:**
   - Business logic cụ thể
   - Domain knowledge features
   - Exploratory analysis

3. **Document rõ ràng:**
   - Comment tại sao làm bước đó
   - Note các threshold và assumptions

### ❌ DON'T

1. **Không hard-code domain logic vào module:**
   ```python
   # ❌ BAD - trong preprocessing.py
   if equipment_type == 'bearing':
       df['fault_severity'] = ...
   
   # ✅ GOOD - trong notebook
   if equipment_type == 'bearing':
       df['fault_severity'] = ...
   ```

2. **Không duplicate code:**
   ```python
   # ❌ BAD - copy paste hàm remove_duplicates vào mỗi notebook
   
   # ✅ GOOD - import từ module
   from scripts.preprocessing import DataPreprocessor
   ```

3. **Không tạo quá nhiều functions trong module:**
   - Chỉ những gì thực sự reusable
   - Giữ module simple và clean

---

## 🔍 Khi nào tạo function mới trong module?

### ✅ Nên tạo khi:
- Function được dùng ≥ 3 lần trong các notebooks khác nhau
- Logic hoàn toàn general, không domain-specific
- Function test được độc lập
- Code dễ đọc và maintain

### ❌ Không nên tạo khi:
- Chỉ dùng 1-2 lần
- Logic phụ thuộc vào domain knowledge
- Quá specific cho một dataset
- Hay thay đổi logic

---

## 📚 Examples

Xem các notebook ví dụ:
- `notebooks/Preprocessing_Example.ipynb` - Cách sử dụng cơ bản
- `notebooks/Multi_Equipment_EDA.ipynb` - Domain-specific processing
- `notebooks/Pipeline_Corrosion_EDA.ipynb` - Pipeline-specific features

---

## 🎓 Tóm tắt

| Tiêu chí | Module | Notebook |
|----------|--------|----------|
| **Mục đích** | Tái sử dụng | Phân tích cụ thể |
| **Scope** | General | Domain-specific |
| **Code style** | Clean, documented | Exploratory, flexible |
| **Testing** | Unit tests | Manual validation |
| **When to use** | ≥3 times, general | 1-2 times, specific |

**Quy tắc vàng:** Nếu nghi ngờ có nên cho vào module không → Để trong notebook trước, sau khi dùng 3 lần mới refactor vào module!
