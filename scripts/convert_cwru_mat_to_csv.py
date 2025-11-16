"""
Convert .mat files in `cwru - bearing (vòng bi)/` to per-file CSV feature extracts.

Output directory: ../data/extracted/cwru/

This script attempts to find the first 1D/2D numeric array inside the .mat and computes
windowed features (RMS, peak, kurtosis, crest factor) per 2048-sample window.
"""
import os
import glob
import pathlib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import kurtosis


def ensure_dir(d):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)


def find_signal_in_mat(mat):
    # Return the first 1D or 2D numeric array that looks like a signal
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.size > 0:
            if v.ndim == 1:
                return v
            if v.ndim == 2:
                # prefer row/col vectors
                if 1 in v.shape:
                    return v.ravel()
                # else if many rows, skip unless only one column
    return None


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


def main():
    src_dir = os.path.join(os.path.dirname(__file__), "..", "cwru - bearing (vòng bi)")
    src_dir = os.path.normpath(src_dir)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "extracted", "cwru")
    ensure_dir(out_dir)

    mats = glob.glob(os.path.join(src_dir, "*.mat"))
    if not mats:
        print("No .mat files found in", src_dir)
        return

    for m in mats:
        print("Processing", m)
        try:
            mat = loadmat(m)
        except Exception as e:
            print("  failed to load:", e)
            continue
        sig = find_signal_in_mat(mat)
        if sig is None:
            print("  no suitable signal array found in", m)
            continue
        df = windowed_features(sig)
        fname = os.path.splitext(os.path.basename(m))[0] + "_features.csv"
        out_path = os.path.join(out_dir, fname)
        df.to_csv(out_path, index=False)
        print("  wrote", out_path)


if __name__ == "__main__":
    main()
