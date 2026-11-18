"""Cache Model 2 per-trip tensors (road subgraph + candidate geometry + history windows).

python -m src.models.road_gnn_rlcd.build_features_m2 -> outputs/cache/m2/<partition>/<trip>.pt
Labels (y_row, far, ambiguous) are stored under separate keys for training/eval only.
"""
import os
import sys

import numpy as np
import torch
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from pathsnap.context import prepare  # noqa: E402
from pathsnap.io import PARTITIONS, list_trips, load_trip  # noqa: E402
from src.models.road_gnn_rlcd.model import build_trip_tensors  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "cache", "m2")


def one(part, tid):
    p = os.path.join(OUT, part, f"{tid}.pt")
    if os.path.exists(p):
        return
    ctx = prepare(load_trip(part, tid))
    d = build_trip_tensors(ctx)
    L = np.load(os.path.join(ROOT, "outputs", "labels", part, f"{tid}.npz"))
    y = np.full(ctx.T, -1)
    hit = np.nonzero(ctx.cands.arc == L["label"][ctx.cands.obs])[0]
    y[ctx.cands.obs[hit]] = hit
    d.update(y_row=y, far=L["far"], ambiguous=L["ambiguous"], T=ctx.T, arc=ctx.cands.arc)
    torch.save(d, p)


if __name__ == "__main__":
    jobs = []
    for part in PARTITIONS:
        os.makedirs(os.path.join(OUT, part), exist_ok=True)
        jobs += [(part, t) for t in list_trips(part)]
    Parallel(n_jobs=14, verbose=5)(delayed(one)(*j) for j in jobs)
