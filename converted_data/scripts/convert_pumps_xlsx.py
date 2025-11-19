"""
Convert `Centrifugal_pumps_measurements.xlsx` (if present) to cleaned CSV(s).
Writes per-sheet CSVs to data/extracted/pumps/
"""
import os
import pandas as pd
import pathlib


def ensure_dir(d):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)


def main():
    base = os.path.join(os.path.dirname(__file__), "..")
    src = os.path.join(base, "Centrifugal_pumps_measurements.xlsx")
    out_dir = os.path.join(base, "data", "extracted", "pumps")
    ensure_dir(out_dir)
    if not os.path.exists(src):
        print("Centrifugal_pumps_measurements.xlsx not found at", src)
        return
    xls = pd.ExcelFile(src)
    for sheet in xls.sheet_names:
        print("Reading sheet", sheet)
        df = xls.parse(sheet)
        # basic clean: strip column names
        df.columns = [str(c).strip() for c in df.columns]
        out = os.path.join(out_dir, f"pumps_{sheet}.csv")
        df.to_csv(out, index=False)
        print("  wrote", out)


if __name__ == '__main__':
    main()
