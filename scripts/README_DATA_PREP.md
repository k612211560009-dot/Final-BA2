# Data preparation & conversion

This folder contains small converter scripts that normalize and extract features from the raw datasets in this workspace.

Structure created by the scripts:

- data/
  - extracted/ # per-source extracted features (cwru, cwru2, cmaps, pumps)
  - processed/ # cleaned CSVs (vibration, pipeline/corrosion)

Scripts (in `scripts/`):

- `convert_cwru_mat_to_csv.py` — windowed feature extraction for `.mat` files in `cwru - bearing (vòng bi)/`.
- `convert_cwru2_to_csv.py` — copies existing CSV features and extracts from raw `.mat` files in `cwru2 - bearing, gearbox/raw/`.
- `convert_cmaps_rul_to_csv.py` — converts RUL `.txt` files in `CMaps - demo engine, turbin (RUL)/` to CSV.
- `convert_vibration_csv_clean.py` — normalizes `vibration_dataset.csv` and writes `data/processed/vibration_dataset_clean.csv`.
- `convert_pipeline_corrosion_csv.py` — cleans `market_pipe_thickness_loss_dataset.csv`, computes corrosion rate and remaining life where possible.
- `convert_pumps_xlsx.py` — converts `Centrifugal_pumps_measurements.xlsx` (if present) into CSV per-sheet.
- `run_all_converters.py` — runner to execute all converters.

How to run

From the workspace root (bash):

```bash
python scripts/run_all_converters.py
```

Each converter will print progress and skip if the expected source files aren't present.
