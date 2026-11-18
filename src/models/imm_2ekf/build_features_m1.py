"""Build Model 1 (variant C) candidate features for train + student_test.

python scripts/build_features_m1.py -> outputs/cache/m1/<partition>/<trip>.npz
The cache stores features (inference-safe) and, in separate keys, the derived
labels needed for training/evaluation (y_row, ambiguous, far).
"""
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from pathsnap.context import prepare  # noqa: E402
from pathsnap.io import DATA_ROOT, PARTITIONS, list_trips, load_trip  # noqa: E402
from src.models.imm_2ekf.model1 import model1_features  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "cache", "m1")


def label_rows(cands, lab):
    y = np.full(len(cands.ptr) - 1, -1)
    hit = np.nonzero(cands.arc == lab[cands.obs])[0]
    y[cands.obs[hit]] = hit
    return y


def one(part, tid):
    p = os.path.join(OUT, part, f"{tid}.npz")
    if os.path.exists(p):
        return tid, 0.0
    t0 = time.time()
    ctx = prepare(load_trip(part, tid))
    F = model1_features(ctx)
    L = np.load(os.path.join(ROOT, "outputs", "labels", part, f"{tid}.npz"))
    np.savez_compressed(p, F=F, ptr=ctx.cands.ptr, arc=ctx.cands.arc, obs=ctx.cands.obs,
                        y_row=label_rows(ctx.cands, L["label"]), ambiguous=L["ambiguous"], far=L["far"],
                        label=L["label"])
    return tid, time.time() - t0


if __name__ == "__main__":
    jobs = []
    for part in PARTITIONS:
        os.makedirs(os.path.join(OUT, part), exist_ok=True)
        jobs += [(part, t) for t in list_trips(part)]
    # largest first for load balance
    jobs.sort(key=lambda j: -os.path.getsize(os.path.join(DATA_ROOT, j[0], j[1], j[1] + ".track")))
    t0 = time.time()
    res = Parallel(n_jobs=14, verbose=5)(delayed(one)(*j) for j in jobs)
    print("total wall", round(time.time() - t0, 1), "s; sum cpu", round(sum(r[1] for r in res), 1))
