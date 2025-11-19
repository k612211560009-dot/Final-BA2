"""
Handle cwru2 folder: copy existing feature CSVs and extract .mat raw files similarly to cwru script.
Output: ../data/extracted/cwru2/
"""
import os
import glob
import pathlib
import shutil
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import kurtosis


def ensure_dir(d):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)


def windowed_features(sig, win=2048):
    n = len(sig)
    out = []
    for start in range(0, n, win):
        w = sig[start:start+win]
        if len(w) < 16:
            continue
        rms = np.sqrt(np.mean(w**2))
        peak = np.max(np.abs(w))
        pk2pk = np.ptp(w)
        kurt = float(kurtosis(w, fisher=False, bias=False))
        crest = peak / (rms + 1e-12)
        out.append({
            "start_sample": start,
            "window_length": len(w),
            "rms": rms,
            "peak": peak,
            "peak_to_peak": pk2pk,
            "kurtosis": kurt,
            "crest_factor": crest,
        })
    return pd.DataFrame(out)


def process_raw_mats(src_raw_dir, out_dir):
    mats = glob.glob(os.path.join(src_raw_dir, "*.mat"))
    for m in mats:
        try:
            mat = loadmat(m)
        except Exception as e:
            print("Failed to load", m, e)
            continue
        # pick first array-like
        sig = None
        for k, v in mat.items():
            if k.startswith("__"):
                continue
            if isinstance(v, np.ndarray) and v.size > 0:
                if v.ndim == 1:
                    sig = v
                    break
                if v.ndim == 2 and 1 in v.shape:
                    sig = v.ravel()
                    break
        if sig is None:
            print("No signal in", m)
            continue
        df = windowed_features(sig)
        out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(m))[0] + "_features.csv")
        df.to_csv(out_path, index=False)
        print("Wrote", out_path)


def main():
    base = os.path.join(os.path.dirname(__file__), "..")
    src_base = os.path.join(base, "cwru2 - bearing, gearbox")
    out_dir = os.path.join(base, "data", "extracted", "cwru2")
    ensure_dir(out_dir)

    # copy existing CSV feature files
    csvs = glob.glob(os.path.join(src_base, "*.csv"))
    for c in csvs:
        print("Copying", c)
        shutil.copy(c, out_dir)

    # handle raw/
    raw_dir = os.path.join(src_base, "raw")
    if os.path.isdir(raw_dir):
        process_raw_mats(raw_dir, out_dir)


if __name__ == "__main__":
    main()
