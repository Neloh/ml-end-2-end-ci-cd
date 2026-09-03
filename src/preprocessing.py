"""Data prep for the e2e MLOps proof.
Reads raw CSV from /opt/ml/processing/input, writes train/validation/test CSV
(XGBoost format: label first column, no header) to the respective output dirs.
"""
import os
import numpy as np
import pandas as pd

IN = "/opt/ml/processing/input"
OUT = "/opt/ml/processing"

def main():
    # If no raw file present, synthesize (keeps the proof self-contained)
    raw_path = os.path.join(IN, "raw.csv")
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
    else:
        rng = np.random.RandomState(42)
        n = 2000
        f1 = rng.rand(n) * 10 - 5
        f2 = rng.rand(n) * 10 - 5
        f3 = rng.rand(n) * 10 - 5
        label = ((f1**2 + f2**2) < 25).astype(int)
        df = pd.DataFrame({"label": label, "f1": f1, "f2": f2, "f3": f3})

    df = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    n = len(df)
    tr, va = int(0.7 * n), int(0.85 * n)
    splits = {"train": df.iloc[:tr], "validation": df.iloc[tr:va], "test": df.iloc[va:]}
    for name, part in splits.items():
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        # XGBoost CSV: label first, no header, no index
        part.to_csv(os.path.join(d, f"{name}.csv"), index=False, header=False)
        print(f"wrote {name}: {len(part)} rows -> {d}")

if __name__ == "__main__":
    main()
