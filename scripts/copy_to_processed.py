"""
Script to copy and organize data from extracted folders to processed folder
"""
import os
import shutil
import pandas as pd
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).parent.parent
EXTRACTED_DIR = BASE_DIR / "converted_data" / "extracted"
PROCESSED_DIR = BASE_DIR / "converted_data" / "processed"

# Create processed directory if it doesn't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def copy_and_consolidate_data():
    """Copy data from extracted to processed folder"""
    
    print("Starting data consolidation...")
    
    # 1. Process CMAPS data (Turbine RUL)
    print("\n1. Processing CMAPS (Turbine) data...")
    cmaps_dir = EXTRACTED_DIR / "cmaps"
    if cmaps_dir.exists():
        # Combine all train files
        train_files = sorted(cmaps_dir.glob("train_FD*.csv"))
        if train_files:
            train_dfs = []
            for f in train_files:
                df = pd.read_csv(f)
                df['dataset'] = f.stem  # Add source info
                train_dfs.append(df)
            combined_train = pd.concat(train_dfs, ignore_index=True)
            combined_train.to_csv(PROCESSED_DIR / "turbine_train_data.csv", index=False)
            print(f"   ✓ Created turbine_train_data.csv ({len(combined_train)} rows)")
        
        # Combine all test files
        test_files = sorted(cmaps_dir.glob("test_FD*.csv"))
        if test_files:
            test_dfs = []
            for f in test_files:
                df = pd.read_csv(f)
                df['dataset'] = f.stem
                test_dfs.append(df)
            combined_test = pd.concat(test_dfs, ignore_index=True)
            combined_test.to_csv(PROCESSED_DIR / "turbine_test_data.csv", index=False)
            print(f"   ✓ Created turbine_test_data.csv ({len(combined_test)} rows)")
        
        # Combine RUL files
        rul_files = sorted(cmaps_dir.glob("RUL_FD*.csv"))
        if rul_files:
            rul_dfs = []
            for f in rul_files:
                df = pd.read_csv(f)
                df['dataset'] = f.stem
                rul_dfs.append(df)
            combined_rul = pd.concat(rul_dfs, ignore_index=True)
            combined_rul.to_csv(PROCESSED_DIR / "turbine_rul_data.csv", index=False)
            print(f"   ✓ Created turbine_rul_data.csv ({len(combined_rul)} rows)")
    
    # 2. Process CWRU data (Bearing)
    print("\n2. Processing CWRU (Bearing) data...")
    cwru_dir = EXTRACTED_DIR / "cwru"
    if cwru_dir.exists():
        cwru_files = sorted(cwru_dir.glob("*.csv"))
        if cwru_files:
            bearing_dfs = []
            for f in cwru_files:
                df = pd.read_csv(f)
                # Extract fault type and load from filename
                parts = f.stem.split('_')
                df['fault_type'] = parts[0]
                df['load'] = parts[1] if len(parts) > 1 else 'unknown'
                df['source_file'] = f.stem
                bearing_dfs.append(df)
            combined_bearing = pd.concat(bearing_dfs, ignore_index=True)
            combined_bearing.to_csv(PROCESSED_DIR / "bearing_features_all.csv", index=False)
            print(f"   ✓ Created bearing_features_all.csv ({len(combined_bearing)} rows)")
    
    # 3. Process CWRU2 data (Bearing + Gearbox)
    print("\n3. Processing CWRU2 (Bearing+Gearbox) data...")
    cwru2_dir = EXTRACTED_DIR / "cwru2"
    if cwru2_dir.exists():
        cwru2_files = sorted(cwru2_dir.glob("*.csv"))
        if cwru2_files:
            for f in cwru2_files:
                # Copy feature_time file directly
                if "feature_time" in f.name:
                    shutil.copy(f, PROCESSED_DIR / "bearing_gearbox_features.csv")
                    print(f"   ✓ Created bearing_gearbox_features.csv")
                else:
                    df = pd.read_csv(f)
                    if len(df) > 0:
                        # Add source info
                        df['source_file'] = f.stem
                        output_name = f"bearing_gearbox_{f.stem}.csv"
                        df.to_csv(PROCESSED_DIR / output_name, index=False)
                        print(f"   ✓ Created {output_name} ({len(df)} rows)")
    
    # 4. Process Pumps data
    print("\n4. Processing Pumps data...")
    pumps_dir = EXTRACTED_DIR / "pumps"
    if pumps_dir.exists():
        pumps_file = pumps_dir / "pumps.csv"
        if pumps_file.exists():
            df = pd.read_csv(pumps_file)
            df.to_csv(PROCESSED_DIR / "pumps_data_clean.csv", index=False)
            print(f"   ✓ Created pumps_data_clean.csv ({len(df)} rows)")
    
    # 5. Copy compressor data if exists (but not the large file)
    print("\n5. Processing Compressor data...")
    compressor_file = EXTRACTED_DIR / "compressor" / "equipment_predictive_maintenance_dataset.csv"
    if compressor_file.exists():
        # Only copy if file is not too large (< 100MB)
        file_size_mb = compressor_file.stat().st_size / (1024 * 1024)
        if file_size_mb < 100:
            shutil.copy(compressor_file, PROCESSED_DIR / "compressor_data.csv")
            print(f"   ✓ Created compressor_data.csv ({file_size_mb:.2f} MB)")
        else:
            print(f"   ⚠ Skipping compressor data (file too large: {file_size_mb:.2f} MB)")
            # Create a sample instead
            df = pd.read_csv(compressor_file, nrows=10000)
            df.to_csv(PROCESSED_DIR / "compressor_data_sample.csv", index=False)
            print(f"   ✓ Created compressor_data_sample.csv (10,000 rows sample)")
    
    print("\n" + "="*60)
    print("Data consolidation completed!")
    print(f"Processed files saved to: {PROCESSED_DIR}")
    print("="*60)

if __name__ == "__main__":
    copy_and_consolidate_data()
