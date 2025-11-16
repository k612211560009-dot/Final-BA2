# Synthetic dataset generator

This script generates a synthetic predictive-maintenance time-series dataset named
`equipment_predictive_maintenance_dataset.csv` covering 2022-01-01 to 2023-12-31 at 15-minute intervals
for 10 equipment units (pumps, turbines, compressors). It injects missing data, outliers, seasonal
patterns, progressive degradation, maintenance events and failure events as described in the project prompt.

Files created:

- `equipment_predictive_maintenance_dataset.csv` — final combined CSV

How to run (bash on Windows):

```bash
python synthetic_data_generator.py
```

If pandas/numpy are missing, install requirements:

```bash
python -m pip install -r requirements.txt --user
```

Notes:

- The generator is self-contained and does not require the original raw files. It produces >50 features and covers the quality constraints.
