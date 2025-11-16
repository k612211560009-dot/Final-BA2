"""
Run all converter scripts to produce cleaned/extracted CSVs.
Each converter checks for the presence of source files and exits gracefully if missing.
"""
import subprocess
import sys
import os


SCRIPTS = [
    "convert_cwru_mat_to_csv.py",
    "convert_cwru2_to_csv.py",
    "convert_cmaps_rul_to_csv.py",
    "convert_vibration_csv_clean.py",
    "convert_pipeline_corrosion_csv.py",
    "convert_pumps_xlsx.py",
]


def run(script):
    path = os.path.join(os.path.dirname(__file__), script)
    print("==> running", script)
    res = subprocess.run([sys.executable, path], cwd=os.path.dirname(__file__))
    if res.returncode != 0:
        print(f"Script {script} exited with code {res.returncode}")


def main():
    for s in SCRIPTS:
        run(s)


if __name__ == '__main__':
    main()
