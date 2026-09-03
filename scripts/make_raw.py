"""Generate a small synthetic raw.csv for the pipeline's DataPrep input. No sensitive data."""
import numpy as np, pandas as pd
rng = np.random.RandomState(42)
n = 2000
f1 = rng.rand(n) * 10 - 5
f2 = rng.rand(n) * 10 - 5
f3 = rng.rand(n) * 10 - 5
label = ((f1**2 + f2**2) < 25).astype(int)
pd.DataFrame({"label": label, "f1": f1, "f2": f2, "f3": f3}).to_csv("/tmp/raw.csv", index=False)
print("wrote /tmp/raw.csv")
