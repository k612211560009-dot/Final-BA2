"""
Convert CMaps RUL-style text files into CSVs. Writes per-file CSVs under data/extracted/cmaps/
"""
import os
import glob
import pathlib
import pandas as pd


def ensure_dir(d):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)


def parse_space_separated_file(path):
    # Many RUL/CMAPSS files are space-separated with variable whitespace
    df = pd.read_csv(path, sep="\s+", header=None, engine='python')
    return df


def main():
    base = os.path.join(os.path.dirname(__file__), "..")
    src_dir = os.path.join(base, "CMaps - demo engine, turbin (RUL)")
    out_dir = os.path.join(base, "data", "extracted", "cmaps")
    ensure_dir(out_dir)

    files = glob.glob(os.path.join(src_dir, "*.txt"))
    if not files:
        print("No CMAPSS txt files found in", src_dir)
        return

    for f in files:
        print("Parsing", f)
        try:
            df = parse_space_separated_file(f)
        except Exception as e:
            print("  failed to parse:", e)
            continue
        out = os.path.join(out_dir, os.path.splitext(os.path.basename(f))[0] + ".csv")
        df.to_csv(out, index=False)
        print("  wrote", out)


if __name__ == "__main__":
    main()
