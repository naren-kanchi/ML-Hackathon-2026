"""Create the frozen 5-fold trip-grouped split manifest (run once).

Balanced by observation count: trips shuffled with a fixed seed, then sorted by
n_obs and dealt to folds in a serpentine order. Label-unreliable trips are
listed but excluded from every fold.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.io import list_trips  # noqa: E402
from pathsnap.splits import BAD_LABEL_TRIPS, N_FOLDS, SEED, SPLIT_PATH  # noqa: E402

if os.path.exists(SPLIT_PATH) and "--force" not in sys.argv:
    sys.exit(f"{SPLIT_PATH} exists; folds are frozen (pass --force to overwrite)")
inv = pd.read_csv(os.path.join(ROOT, "outputs", "eda", "inventory.csv"), dtype={"trip": str})
n_obs = dict(zip(inv.trip, inv.n_obs))
train = [t for t in list_trips("train") if t not in BAD_LABEL_TRIPS]
rng = np.random.default_rng(SEED)
rng.shuffle(train)
train.sort(key=lambda t: -n_obs[t])
folds = [[] for _ in range(N_FOLDS)]
for i, t in enumerate(train):
    r, c = divmod(i, N_FOLDS)
    folds[c if r % 2 == 0 else N_FOLDS - 1 - c].append(t)
man = dict(seed=SEED, n_folds=N_FOLDS, folds=[sorted(f) for f in folds],
           fold_obs=[int(sum(n_obs[t] for t in f)) for f in folds],
           excluded_label_unreliable=sorted(BAD_LABEL_TRIPS),
           student_test=list_trips("student_test"),
           evaluator_test="not present on this machine; never used")
os.makedirs(os.path.dirname(SPLIT_PATH), exist_ok=True)
json.dump(man, open(SPLIT_PATH, "w"), indent=1)
print(json.dumps({k: v for k, v in man.items() if k != "folds"}, indent=1))
