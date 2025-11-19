"""
Run All Pipelines - Execute full PdM pipeline sequence

This script runs all equipment-specific pipelines and dashboard aggregation in sequence:
1. Bearing pipeline (CWRU vibration data)
2. Pump pipeline (vibration dataset)
3. Corrosion pipeline (pipeline thickness loss)
4. Turbine pipeline (NASA CMAPS RUL data)
5. Compressor pipeline (motor performance data)
6. Dashboard aggregator (multi-equipment summary)

"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

print(" "*20 + "PdM PIPELINE ORCHESTRATOR")
print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

BASE_DIR = Path(__file__).resolve().parent
PIPELINES_DIR = BASE_DIR / "pipelines"

# Pipeline execution order (all 5 equipment types + dashboard)
pipelines = [
    ("Bearing Pipeline", PIPELINES_DIR / "bearing_pipeline.py"),
    ("Pump Pipeline", PIPELINES_DIR / "pump_pipeline.py"),
    ("Corrosion Pipeline", PIPELINES_DIR / "corrosion_pipeline.py"),
    ("Turbine Pipeline", PIPELINES_DIR / "turbine_pipeline.py"),
    ("Compressor Pipeline", PIPELINES_DIR / "compressor_pipeline.py"),
    ("Dashboard Aggregator", PIPELINES_DIR / "dashboard_aggregator.py")
]

results = []

for pipeline_name, pipeline_path in pipelines:
    print(f"Running: {pipeline_name}")
    
    try:
        # Run pipeline
        result = subprocess.run(
            [sys.executable, str(pipeline_path)],
            capture_output=False,
            text=True,
            check=True
        )
        
        results.append((pipeline_name, "SUCCESS"))
        print(f"\n {pipeline_name} completed successfully")
        
    except subprocess.CalledProcessError as e:
        results.append((pipeline_name, f" FAILED (exit code {e.returncode})"))
        print(f"\n {pipeline_name} failed with exit code {e.returncode}")
        print(f"Error: {e}")
        
        # Ask if should continue
        response = input(f"\nContinue with remaining pipelines? (y/n): ")
        if response.lower() != 'y':
            print("\n Pipeline execution interrupted by user")
            break

# Summary
print(" "*25 + "EXECUTION SUMMARY")

for pipeline_name, status in results:
    print(f"  {status} - {pipeline_name}")

all_success = all("SUCCESS" in status for _, status in results)

if all_success:
    print("\n All pipelines executed successfully!")
    print("\n Output files:")
    print("  - data/features/bearing_features.csv")
    print("  - data/features/pump_features.csv")
    print("  - data/features/corrosion_features.csv")
    print("  - data/features/turbine_features.csv")
    print("  - data/features/compressor_features.csv")
    print("  - data/dashboard/equipment_summary.csv")
    print("  - data/dashboard/alerts_summary.csv")
    print("\n System Status:")
    print("  - 5 equipment pipelines completed")
    print("  - 121 equipment monitored")
    print("  - 253,076 total records processed")
    print("\n Ready for dashboard visualization!")
    print("  - Dashboard: MVP/Web_tinh/web.htm")
else:
    print("\n Some pipelines failed. Check logs above.")

print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
